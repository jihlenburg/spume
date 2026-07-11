// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <span>

#include "core/precond.hpp"
#include "core/sell.hpp"
#include "core/types.hpp"

namespace spume {

struct SolveOptions {
    double tol = 1e-10; // convergence: ||r||_2 <= tol * ||b||_2 (recursive residual)
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
