// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include "spumePCG.H"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <span>
#include <vector>

#include "bridge/ldu_to_sell.hpp"
#include "core/amg_precond.hpp"
#include "core/equilibrate.hpp"
#include "core/precond.hpp"
#include "core/sell.hpp"
#include "core/solver.hpp"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(spumePCG, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<spumePCG>
        addspumePCGSymMatrixConstructorToTable_;
}

// * * * * * * * * * * * * * * * Local Helpers  * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace
{

//- True if any coupled interface (processor/cyclic/AMI) contributes to Amul.
// The interface-field ptr list holds only coupled interfaces; physical BC
// patches are null entries (their contribution is already baked into diag and
// source). So any set() entry is a coupling the SELL bridge does not capture.
bool hasCoupledInterface(const lduInterfaceFieldPtrsList& interfaces)
{
    forAll(interfaces, i)
    {
        if (interfaces.set(i))
        {
            return true;
        }
    }
    return false;
}

//- Amortised AMG-aggregation cache. The costly part of AMG setup is the greedy
// strength-based aggregation (a graph traversal over every level); the mesh
// sparsity is static across a PIMPLE run, so the aggregation STRUCTURE is too.
// Cache the per-level aggregations and rebuild only the coefficient-dependent
// operators (Galerkin products + smoother) from the CURRENT matrix each solve —
// exactly how stock GAMG reuses its GAMGAgglomeration MeshObject while
// recomputing coarse matrices. Crucially the preconditioner therefore stays
// CURRENT: caching the whole preconditioner instead (stale operators) blows up a
// transient run as the coefficients drift (measured: 28 -> 900 iters by
// timestep 3), which is exactly what this must not do.
//
// Single entry, keyed on the lduAddressing pointer (owned by the mesh, stable
// for a static mesh) plus cell count and preconditioner name. The pressure solve
// runs serially within PIMPLE and each MPI rank is its own process, so a
// process-static cache needs no locking. The cached aggregations own their data
// (no references into OpenFOAM), so they safely outlive the transient solver.
struct AmgHierarchyCache
{
    const void* key = nullptr;
    Foam::label ncells = -1;
    Foam::word name;
    std::vector<spume::Aggregation> aggs; // cached STRUCTURE (topology, static)
    bool valid = false;
};
AmgHierarchyCache g_amgCache;

} // End anonymous namespace
} // End namespace Foam

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * * //

Foam::spumePCG::spumePCG
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

Foam::solverPerformance Foam::spumePCG::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    // Correctness fallback (ADR-0004: reference by default). The SELL bridge
    // captures diag + upper/lower only; OpenFOAM's Amul also folds in coupled-
    // interface (processor/cyclic) contributions. For asymmetric matrices or
    // when a coupled interface is active, delegate to the reference PCG.
    // Bridging halo coupling is a later M2 slice.
    if (!matrix_.symmetric() || hasCoupledInterface(interfaces_))
    {
        return lduMatrix::solver::New
        (
            "PCG",
            fieldName_,
            matrix_,
            interfaceBouCoeffs_,
            interfaceIntCoeffs_,
            interfaces_,
            controlDict_
        )->solve(psi, source, cmpt);
    }

    solverPerformance solverPerf(typeName, fieldName_);

    const label nCells = psi.size();

    // Assemble the SELL-C operator from LDU (symmetric: lower == upper).
    const labelUList& lowerAddr = matrix_.lduAddr().lowerAddr();
    const labelUList& upperAddr = matrix_.lduAddr().upperAddr();
    const scalarField& diag = matrix_.diag();
    const scalarField& upper = matrix_.upper();

    // OpenFOAM assembles the pressure matrix negative-definite (diag < 0). The
    // SPUME core CG and diagonal Jacobi preconditioner require an SPD operator,
    // so solve the negated system (-A) x = (-b): identical solution x, SPD
    // operator (positive diagonal). Detect from the diagonal sign so a
    // positive-definite matrix is left untouched.
    const double sgn = (nCells > 0 && diag[0] < 0) ? -1.0 : 1.0;

    std::vector<double> sdiag(static_cast<std::size_t>(nCells));
    for (label i = 0; i < nCells; ++i)
    {
        sdiag[static_cast<std::size_t>(i)] = sgn*diag[i];
    }
    std::vector<double> supper(static_cast<std::size_t>(upper.size()));
    for (label f = 0; f < upper.size(); ++f)
    {
        supper[static_cast<std::size_t>(f)] = sgn*upper[f];
    }

    spume::Sell<double> a = spume::assemble_sell
    (
        std::span<const int>(lowerAddr.cdata(), lowerAddr.size()),
        std::span<const int>(upperAddr.cdata(), upperAddr.size()),
        std::span<const double>(sdiag),
        std::span<const double>(supper),
        std::span<const double>{},              // symmetric
        static_cast<int>(nCells)
    );

    // Initial residual in OpenFOAM's normalised convention (for logging and
    // the stopping check), matching the reference solvers.
    scalarField wA(nCells);
    scalarField pA(nCells);
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    scalarField rA(source - wA);
    const scalar normFactor = this->normFactor(psi, source, wA, pA);

    solverPerf.initialResidual() = gSumMag(rA, matrix().mesh().comm())/normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    if (minIter_ > 0 || !solverPerf.checkConvergence(tolerance_, relTol_, log_))
    {
        spume::SolveOptions opt;
        opt.tol = static_cast<double>(tolerance_);
        opt.max_iter = (maxIter_ > 0 ? static_cast<int>(maxIter_) : 1000);
        // Threaded (OpenMP) SELL kernels by default. openmp is bit-identical to
        // the reference path (SELL fixed-order accumulation), so this is pure
        // parallelisation, not a different numerical path (invariant #4 holds).
        const spume::Dispatch disp =
            controlDict_.getOrDefault<bool>("spumeThreaded", true)
                ? spume::Dispatch::openmp : spume::Dispatch::reference;
        opt.dispatch = disp;

        std::vector<double> x(psi.begin(), psi.begin() + nCells);
        std::vector<double> b(static_cast<std::size_t>(nCells));
        for (label i = 0; i < nCells; ++i)
        {
            b[static_cast<std::size_t>(i)] = sgn*source[i];  // (-A) x = (-b)
        }

        // Preconditioner selection. Default: FP64 exact diagonal Jacobi (the
        // reference-precision path, ADR-0004). Opt-in `chebyshevFP32` is the
        // ADR-0002 mixed-precision path — reduced precision ONLY inside the
        // preconditioner, under the flexible FP64 outer CG, with mandatory
        // diagonal equilibration (make_eq_operator<float> equilibrates in FP64
        // then narrows to FP32). The outer operator `a` stays FP64.
        const word pc =
            controlDict_.getOrDefault<word>("spumePreconditioner", "jacobi");

        const auto tSetup0 = std::chrono::steady_clock::now();

        // Amortisation policy (see AmgHierarchyCache): cache the aggregation
        // STRUCTURE (mesh-static topology) and rebuild the coefficient-dependent
        // operators + smoother from the CURRENT matrix on every solve. This is
        // how GAMG reuses its agglomeration while recomputing coarse matrices —
        // the preconditioner stays current instead of drifting toward the first
        // matrix, so a transient run's evolving coefficients cannot degrade it.
        const bool amgCache = controlDict_.getOrDefault<bool>("spumeAmgCache", true);
        const void* meshKey = static_cast<const void*>(&matrix_.lduAddr());

        std::unique_ptr<spume::Preconditioner> localPrecond;
        spume::Preconditioner* activePrecond = nullptr;

        if (pc == "jacobi")
        {
            spume::EqOperator<double> eq;
            eq.scale.resize(static_cast<std::size_t>(nCells));
            for (label i = 0; i < nCells; ++i)
            {
                eq.scale[static_cast<std::size_t>(i)] =
                    1.0/std::sqrt(sdiag[static_cast<std::size_t>(i)]);
            }
            localPrecond = std::make_unique<spume::JacobiPrecond<double>>(eq, disp);
        }
        else
        {
            // Non-Jacobi preconditioners consume the negated SPD operator as CSR
            // (equilibrated to FP32 for the mixed-precision paths, ADR-0002),
            // rebuilt from the current matrix each solve.
            const spume::Csr csr = spume::assemble_csr
            (
                std::span<const int>(lowerAddr.cdata(), lowerAddr.size()),
                std::span<const int>(upperAddr.cdata(), upperAddr.size()),
                std::span<const double>(sdiag),
                std::span<const double>(supper),
                std::span<const double>{},
                static_cast<int>(nCells)
            );

            // V-cycle smoother knobs, dictionary-tunable without a rebuild.
            // Defaults reproduce the previous behaviour (steps 5, eta 30).
            spume::ChebyshevOptions copt;
            copt.steps =
                static_cast<int>(controlDict_.getOrDefault<label>("chebyshevSteps", 5));
            copt.eta =
                static_cast<double>(controlDict_.getOrDefault<scalar>("chebyshevEta", 30.0));
            const double coarseTol =
                static_cast<double>(controlDict_.getOrDefault<scalar>("amgCoarseTol", 1e-2));
            // K-cycle: Krylov-accelerated coarse correction, GAMG-class rate on
            // unsmoothed aggregation. Default on (same FP64 answer, ADR-0002).
            const bool kcycle = controlDict_.getOrDefault<bool>("spumeKcycle", true);
            const int kcycleLevels =
                static_cast<int>(controlDict_.getOrDefault<label>("spumeKcycleLevels", 5));

            if (pc == "chebyshevFP32")
            {
                localPrecond = std::make_unique<spume::ChebyshevPrecond<float>>
                (
                    spume::make_eq_operator<float>(csr), copt, disp
                );
            }
            else if (pc == "amgFP32" || pc == "amgFP64")
            {
                // Self-coarsening AMG. Cache the aggregation hierarchy (static)
                // and build the operators fresh from `csr` each solve.
                std::vector<spume::Aggregation> freshAggs;
                const bool cacheHit = amgCache && g_amgCache.valid
                    && g_amgCache.key == meshKey
                    && g_amgCache.ncells == nCells
                    && g_amgCache.name == pc;
                if (amgCache && !cacheHit)
                {
                    g_amgCache.aggs = spume::aggregate_hierarchy(csr, 200, 20);
                    g_amgCache.key = meshKey;
                    g_amgCache.ncells = nCells;
                    g_amgCache.name = pc;
                    g_amgCache.valid = true;
                }
                else if (!amgCache)
                {
                    freshAggs = spume::aggregate_hierarchy(csr, 200, 20);
                }
                const std::vector<spume::Aggregation>& aggs =
                    amgCache ? g_amgCache.aggs : freshAggs;

                if (pc == "amgFP32")
                {
                    localPrecond = std::make_unique<spume::AmgPrecond<float>>(csr, aggs, copt, coarseTol, 500, disp, kcycle, kcycleLevels);
                }
                else
                {
                    localPrecond = std::make_unique<spume::AmgPrecond<double>>(csr, aggs, copt, coarseTol, 500, disp, kcycle, kcycleLevels);
                }
            }
            else if (pc == "gamgFP32" || pc == "gamgFP64")
            {
                // Reuse OpenFOAM's cached GAMGAgglomeration hierarchy (a
                // MeshObject) and run the SPUME FP32/FP64 V-cycle on it; the
                // operators are rebuilt from the current matrix each solve. This
                // is the "reuse the trunk, own the kernels" path (ADR-0001).
                const GAMGAgglomeration& gagg =
                    GAMGAgglomeration::New(matrix_, controlDict_);

                std::vector<spume::Aggregation> hierarchy;
                hierarchy.reserve(static_cast<std::size_t>(gagg.size()));
                for (label lev = 0; lev < gagg.size(); ++lev)
                {
                    const labelField& ra = gagg.restrictAddressing(lev);
                    spume::Aggregation agg;
                    agg.agg.assign(ra.begin(), ra.end());
                    agg.ncoarse = static_cast<spume::index_t>(gagg.nCells(lev));
                    hierarchy.push_back(std::move(agg));
                }

                if (pc == "gamgFP32")
                {
                    localPrecond = std::make_unique<spume::AmgPrecond<float>>(csr, hierarchy, copt, coarseTol, 500, disp, kcycle, kcycleLevels);
                }
                else
                {
                    localPrecond = std::make_unique<spume::AmgPrecond<double>>(csr, hierarchy, copt, coarseTol, 500, disp, kcycle, kcycleLevels);
                }
            }
            else
            {
                FatalErrorInFunction
                    << "Unknown spumePreconditioner '" << pc
                    << "' (expected jacobi | chebyshevFP32 | amgFP32 | amgFP64)"
                    << exit(FatalError);
            }
        }
        activePrecond = localPrecond.get();

        const auto tSetup1 = std::chrono::steady_clock::now();
        const spume::SolveResult res = spume::fcg
        (
            a,
            *activePrecond,
            std::span<const double>(b),
            std::span<double>(x),
            opt
        );
        const auto tSolve1 = std::chrono::steady_clock::now();

        // Per-phase attribution (setup = preconditioner/hierarchy build, solve =
        // the flexible-CG loop). Printed at log level >= 1 so a run self-reports
        // where its time went without post-hoc log forensics.
        if (log_ >= 1)
        {
            using ms = std::chrono::duration<double, std::milli>;
            Info<< "spumePCG[" << pc << "]: setup "
                << ms(tSetup1 - tSetup0).count() << " ms, solve "
                << ms(tSolve1 - tSetup1).count() << " ms, "
                << res.iterations << " iters ("
                << (disp == spume::Dispatch::openmp ? "openmp" : "reference")
                << ")" << endl;
        }

        std::copy(x.begin(), x.end(), psi.begin());

        // Final residual, same normalisation.
        matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
        rA = source - wA;
        solverPerf.finalResidual() = gSumMag(rA, matrix().mesh().comm())/normFactor;
        solverPerf.nIterations() = res.iterations;
    }

    return solverPerf;
}

// ************************************************************************* //
