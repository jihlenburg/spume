// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// SPUME <-> OpenFOAM compatibility shim: the single point at which SPUME solver
// libraries reach the upstream LDU matrix / linear-solver API (architecture
// invariant #2 — no direct vendor/ or upstream includes elsewhere in src/ or
// applications/).
//
// Unlike compat/foam.hpp (the finite-volume umbrella, fvCFD.H), this is the
// lower matrix layer: it pulls only lduMatrix.H, which declares lduMatrix,
// lduMatrix::solver (+ its runtime selection tables), solverPerformance,
// scalarField, and the FieldField/lduInterface types a solver constructor
// needs. A runtime-selectable SPUME solver (e.g. spumePCG) needs the matrix
// layer, not the whole FV framework, so this header deliberately does not
// include compat/foam.hpp — keeping solver libraries linkable against
// libOpenFOAM alone.

#ifndef SPUME_COMPAT_LDU_HPP
#define SPUME_COMPAT_LDU_HPP

// LDU matrix, the lduMatrix::solver base + symMatrix/asymMatrix runtime
// selection tables, solverPerformance, and scalarField.
#include "lduMatrix.H"

// Pin: SPUME targets the OpenFOAM v2606 API surface (ADR-0014). Defined by
// wmake (WM_VERSION = OPENFOAM=2606). See compat/foam.hpp for the rationale.
#if !defined(OPENFOAM)
#  error "SPUME compat shim: OPENFOAM version macro undefined — build via wmake against a v2606+ tree (ADR-0014)."
#elif OPENFOAM < 2606
#  error "SPUME requires the OpenFOAM v2606 API or newer (ADR-0014)."
#endif

#endif  // SPUME_COMPAT_LDU_HPP
