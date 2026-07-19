// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <vector>

#include "core/formats.hpp"
#include "core/types.hpp"

// Algebraic-multigrid coarsening — the setup half of the FP32-GAMG-under-FP64-
// Krylov preconditioner (roadmap M2, ADR-0002). This module builds, from a
// matrix alone, one coarse level: an aggregation of the fine unknowns and the
// Galerkin coarse operator A_c = P^T A P for the piecewise-constant tentative
// prolongation P (P[i][agg[i]] = 1). The V-cycle (smooth / restrict / recurse /
// prolong) is assembled on top of these in a later slice; the smoother is the
// existing Chebyshev polynomial (core/precond.hpp), the coarsest solve is FP64
// CG (core/solver.hpp).
//
// Everything here is FP64 and OpenFOAM-free: the hierarchy is built once per
// case in FP64 for correctness, and each level's operator is narrowed to FP32
// (via make_eq_operator<float>, the only sanctioned demotion) when the V-cycle
// is assembled. Coarsening structure is static across timesteps (the mesh is
// static), so this setup is amortised — only coefficients update.

namespace spume {

// Aggregation of the fine unknowns: agg[i] is the coarse aggregate index of
// fine row i, in [0, ncoarse).
struct Aggregation {
    std::vector<index_t> agg;
    index_t ncoarse = 0;
};

// Greedy strength-based aggregation. A seed unknown adopts its still-unaggregated
// neighbours whose coupling is strong relative to the strongest off-diagonal in
// the seed's row (|a_ij| >= theta * max_k|a_ik|). Deterministic (natural order),
// so the hierarchy is reproducible. theta in (0,1]; larger = smaller aggregates.
Aggregation aggregate(const Csr& a, double theta = 0.25);

// Galerkin coarse operator A_c = P^T A P for the piecewise-constant P defined by
// `agg`. For that P this is exactly the sum of A's entries grouped by aggregate:
// A_c[I][J] = sum_{i in I, j in J} a[i][j]. Symmetric if A is symmetric, and it
// preserves the constant near-null-space (A 1 = 0  =>  A_c 1 = 0), the property
// that makes it a valid coarse operator.
Csr galerkin(const Csr& a, const Aggregation& agg);

} // namespace spume
