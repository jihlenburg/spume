// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include "spumePCG.H"

#include <algorithm>
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

        std::unique_ptr<spume::Preconditioner> precond;
        if (pc == "jacobi")
        {
            spume::EqOperator<double> eq;
            eq.scale.resize(static_cast<std::size_t>(nCells));
            for (label i = 0; i < nCells; ++i)
            {
                eq.scale[static_cast<std::size_t>(i)] =
                    1.0/std::sqrt(sdiag[static_cast<std::size_t>(i)]);
            }
            precond = std::make_unique<spume::JacobiPrecond<double>>(eq);
        }
        else
        {
            // Non-Jacobi preconditioners consume the negated SPD operator as
            // CSR (equilibrated to FP32 for the mixed-precision paths, ADR-0002).
            const spume::Csr csr = spume::assemble_csr
            (
                std::span<const int>(lowerAddr.cdata(), lowerAddr.size()),
                std::span<const int>(upperAddr.cdata(), upperAddr.size()),
                std::span<const double>(sdiag),
                std::span<const double>(supper),
                std::span<const double>{},
                static_cast<int>(nCells)
            );

            if (pc == "chebyshevFP32")
            {
                spume::ChebyshevOptions copt;
                copt.steps =
                    static_cast<int>(controlDict_.getOrDefault<label>("chebyshevSteps", 5));
                copt.eta =
                    static_cast<double>(controlDict_.getOrDefault<scalar>("chebyshevEta", 30.0));
                precond = std::make_unique<spume::ChebyshevPrecond<float>>
                (
                    spume::make_eq_operator<float>(csr), copt
                );
            }
            else if (pc == "amgFP32")
            {
                // FP32 algebraic multigrid — the M2 lever.
                precond = std::make_unique<spume::AmgPrecond<float>>(csr);
            }
            else if (pc == "amgFP64")
            {
                precond = std::make_unique<spume::AmgPrecond<double>>(csr);
            }
            else
            {
                FatalErrorInFunction
                    << "Unknown spumePreconditioner '" << pc
                    << "' (expected jacobi | chebyshevFP32 | amgFP32 | amgFP64)"
                    << exit(FatalError);
            }
        }

        const spume::SolveResult res = spume::fcg
        (
            a,
            *precond,
            std::span<const double>(b),
            std::span<double>(x),
            opt
        );

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
