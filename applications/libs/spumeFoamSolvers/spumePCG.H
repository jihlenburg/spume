// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// spumePCG — a runtime-selectable SPUME linear solver for symmetric
// lduMatrices, the M2 integration seam (ADR-0001: runtime selection tables
// load SPUME-owned solvers with zero upstream core diff; ADR-0004: default
// behaviour is the reference — a case opts in with `solver spumePCG`).
//
// Stage 0 (this file): delegates to the upstream PCG through the solver runtime
// selection table, so a case selecting `spumePCG` is bit-for-bit identical to
// `PCG`. Stage 1 swaps the internals to the SPUME M0 CG over the SELL bridge
// (src/bridge/ldu_to_sell.hpp) under the reorder-tolerance equivalence gate.

#ifndef SPUME_SPUMEPCG_H
#define SPUME_SPUMEPCG_H

#include "compat/ldu.hpp"

namespace Foam
{

class spumePCG
:
    public lduMatrix::solver
{
    // Private Member Functions

        spumePCG(const spumePCG&) = delete;
        void operator=(const spumePCG&) = delete;

public:

    //- Runtime type information
    TypeName("spumePCG");

    // Constructors

        spumePCG
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

    //- Destructor
    virtual ~spumePCG() = default;

    // Member Functions

        //- Solve the matrix with this solver
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt = 0
        ) const;
};

} // End namespace Foam

#endif
