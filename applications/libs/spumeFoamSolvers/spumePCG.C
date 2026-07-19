// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include "spumePCG.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(spumePCG, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<spumePCG>
        addspumePCGSymMatrixConstructorToTable_;
}

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
    // Stage 0 passthrough: construct the reference PCG through the runtime
    // selection table with this solver's controls and delegate. The explicit
    // "PCG" name bypasses the dictionary's `solver spumePCG` entry, so the
    // result is bit-for-bit identical to selecting `solver PCG` directly.
    // Stage 1 replaces this body with the SPUME M0 CG over the SELL bridge.
    autoPtr<lduMatrix::solver> ref = lduMatrix::solver::New
    (
        "PCG",
        fieldName_,
        matrix_,
        interfaceBouCoeffs_,
        interfaceIntCoeffs_,
        interfaces_,
        controlDict_
    );

    return ref->solve(psi, source, cmpt);
}

// ************************************************************************* //
