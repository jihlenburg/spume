// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// SPUME <-> OpenFOAM compatibility shim: the single point at which SPUME code
// includes the upstream OpenFOAM library API (architecture invariant #2 — no
// direct vendor/ or upstream includes elsewhere in src/). Upstream API churn
// is absorbed here and only here.
//
// Scope note: this rule governs upstream *library* headers — the `.H` files
// that declare OpenFOAM classes. The textual app-assembly fragments an
// OpenFOAM solver includes by convention (createFields.H, UEqn.H, pEqn.H,
// setRootCaseLists.H, ...) are the standard wmake solver-assembly mechanism,
// resolved via -I paths at build time; they are treated like vendored code,
// not as SPUME source subject to this rule. See src/compat/README.md.

#ifndef SPUME_COMPAT_FOAM_HPP
#define SPUME_COMPAT_FOAM_HPP

// OpenFOAM finite-volume umbrella: argList, Time, fvMesh, vol/surfaceFields,
// the fvm/fvc operators, Info/messageStream, and the FatalError machinery.
#include "fvCFD.H"

// Pin: SPUME targets the OpenFOAM v2606 API surface (ADR-0014). Upstream
// defines the integer macro OPENFOAM through wmake (WM_VERSION = OPENFOAM=2606
// in wmake/rules/General/general). The nightly merge canary builds against
// upstream develop, whose value is >= 2606.
#if !defined(OPENFOAM)
#  error "SPUME compat shim: OPENFOAM version macro undefined — build via wmake against a v2606+ tree (ADR-0014)."
#elif OPENFOAM < 2606
#  error "SPUME requires the OpenFOAM v2606 API or newer (ADR-0014)."
#endif

#endif  // SPUME_COMPAT_FOAM_HPP
