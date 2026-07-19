// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <span>

#include "core/formats.hpp"
#include "core/sell.hpp"

namespace spume {

// Assemble a SELL-C-8 operator from OpenFOAM LDU addressing + coefficients.
//
// This is the M2 bridge between OpenFOAM's lduMatrix and the SPUME core. It
// takes plain spans (no OpenFOAM types) so it is unit-testable without a built
// OpenFOAM; the solver library adapts UList/scalarField to spans at the call
// site.
//
// LDU convention (OpenFOAM lduMatrix::Amul):
//   - diag[i]        = A[i][i]                                (nCells entries)
//   - upper[f]       = A[lowerAddr[f]][upperAddr[f]]          (nFaces entries)
//   - lower[f]       = A[upperAddr[f]][lowerAddr[f]]          (nFaces entries)
//   with lowerAddr[f] < upperAddr[f].
//
// If `lower` is empty the matrix is treated as symmetric (lower == upper) — the
// storage OpenFOAM uses for symmetric operators such as the pressure Poisson
// matrix. Coefficients are copied verbatim; no sign convention is imposed.
//
// Throws std::invalid_argument if the array sizes are inconsistent.
Sell<double> assemble_sell(std::span<const int> lowerAddr,
                           std::span<const int> upperAddr,
                           std::span<const double> diag,
                           std::span<const double> upper,
                           std::span<const double> lower,
                           int nCells);

// As assemble_sell, but returns the CSR form. Used to build the FP32
// equilibrated preconditioner operator (make_eq_operator<float> takes a Csr).
// Same LDU convention and validation.
Csr assemble_csr(std::span<const int> lowerAddr,
                 std::span<const int> upperAddr,
                 std::span<const double> diag,
                 std::span<const double> upper,
                 std::span<const double> lower,
                 int nCells);

} // namespace spume
