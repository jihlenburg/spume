// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// Upstream API surface for the incompressible PIMPLE solver family, consumed
// by the spumePimpleFoam leaf application. Adding another solver family means
// adding a sibling header here — never a raw upstream include in leaf code.

#ifndef SPUME_COMPAT_PIMPLE_HPP
#define SPUME_COMPAT_PIMPLE_HPP

#include "compat/foam.hpp"

#include "dynamicFvMesh.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"

#endif  // SPUME_COMPAT_PIMPLE_HPP
