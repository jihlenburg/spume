// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <span>

#include "core/precond.hpp"
#include "core/sell.hpp"
#include "core/types.hpp"

namespace spume {

struct SolveOptions {
    // Convergence: stop once the relative residual ||r||_2/||b||_2 drops to
    //   max(tol, rel_tol * initial relative residual),
    // mirroring OpenFOAM's max(tolerance, relTol*initialResidual) stop so a
    // leaf solver can honour a case's relTol (rel_tol = 0 => absolute tol only,
    // the previous behaviour). At least min_iter iterations always run.
    double tol = 1e-10;
    double rel_tol = 0.0;
    int min_iter = 0;
    int max_iter = 10000;
    // Policy defaults: reference dispatch, standard reductions. Optimized
    // paths and the deterministic debug mode are opt-in at runtime.
    Dispatch dispatch = Dispatch::reference;
    Reduction reduction = Reduction::standard;
};

struct SolveResult {
    int iterations = 0;
    double rel_residual = 0.0;
    bool converged = false;
};

// Reference solver: plain conjugate gradients, FP64 throughout. This is the
// baseline every mixed-precision result is judged against.
SolveResult cg(const Sell<double>& a, std::span<const double> b, std::span<double> x,
               const SolveOptions& opt = {});

// Flexible preconditioned conjugate gradients (FCG): CG with the
// Polak-Ribiere beta
//
//     beta_k = (z_{k+1}, r_{k+1} - r_k) / (z_k, r_k)
//            = -alpha_k (z_{k+1}, q_k) / (z_k, r_k),
//
// which reduces to the standard Fletcher-Reeves beta when M is constant but
// stays convergent when M varies slightly per application — exactly what a
// reduced-precision preconditioner is. Outer iteration is FP64; reduced
// precision enters only through M (numerics policy).
SolveResult fcg(const Sell<double>& a, const Preconditioner& m, std::span<const double> b,
                std::span<double> x, const SolveOptions& opt = {});

} // namespace spume
