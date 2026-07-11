// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include "core/formats.hpp"
#include "core/types.hpp"

namespace spume::gen {

// Exact extreme eigenvalues of the 7-point operator below.
struct PoissonBounds {
    double lambda_min;
    double lambda_max;
};

// 7-point Poisson operator on the interior of a structured nx*ny*nz grid
// (homogeneous Dirichlet boundary, unit grid spacing scaling): diagonal 6,
// off-diagonals -1 toward each existing neighbor, lexicographic ordering
// idx = i + nx*(j + ny*k). SPD.
Coo poisson7(index_t nx, index_t ny, index_t nz);

// Exact extreme eigenvalues: sums over dimensions of 4 sin^2(pi/(2(n+1)))
// and 4 sin^2(n pi/(2(n+1))). Used by tests for rigorous error bounds.
PoissonBounds poisson7_bounds(index_t nx, index_t ny, index_t nz);

} // namespace spume::gen
