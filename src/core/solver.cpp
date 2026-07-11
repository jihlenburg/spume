// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include "core/solver.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "core/reduce.hpp"
#include "core/spmv.hpp"
#include "core/vecops.hpp"

namespace spume {

namespace {

void check_shapes(const Sell<double>& a, std::span<const double> b, std::span<double> x) {
    if (a.nrows != a.ncols) {
        throw std::invalid_argument("solver: matrix must be square");
    }
    if (b.size() != static_cast<std::size_t>(a.nrows) ||
        x.size() != static_cast<std::size_t>(a.nrows)) {
        throw std::invalid_argument("solver: vector size mismatch");
    }
}

} // namespace

SolveResult cg(const Sell<double>& a, std::span<const double> b, std::span<double> x,
               const SolveOptions& opt) {
    check_shapes(a, b, x);
    const auto n = static_cast<std::size_t>(a.nrows);
    const Dispatch d = opt.dispatch;
    const Reduction m = opt.reduction;

    const double bnorm = nrm2(b, d, m);
    if (bnorm == 0.0) {
        for (auto& v : x) {
            v = 0.0;
        }
        return SolveResult{0, 0.0, true};
    }

    std::vector<double> r(n), p(n), q(n);

    // r = b - A x
    spmv(a, std::span<const double>(x), std::span<double>(r), d);
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = b[i] - r[i];
    }

    double rho = dot(r, r, d, m);
    double relres = std::sqrt(rho) / bnorm;
    if (relres <= opt.tol) {
        return SolveResult{0, relres, true};
    }
    p = r;

    for (int it = 1; it <= opt.max_iter; ++it) {
        spmv(a, std::span<const double>(p), std::span<double>(q), d);
        const double pq = dot(p, q, d, m);
        if (!(pq > 0.0) || !std::isfinite(pq)) {
            return SolveResult{it, relres, false}; // not SPD or breakdown
        }
        const double alpha = rho / pq;
        axpy(alpha, std::span<const double>(p), x, d);
        axpy(-alpha, std::span<const double>(q), std::span<double>(r), d);

        const double rho_new = dot(r, r, d, m);
        relres = std::sqrt(rho_new) / bnorm;
        if (relres <= opt.tol) {
            return SolveResult{it, relres, true};
        }
        aypx(rho_new / rho, std::span<const double>(r), std::span<double>(p), d);
        rho = rho_new;
    }
    return SolveResult{opt.max_iter, relres, false};
}

SolveResult fcg(const Sell<double>& a, const Preconditioner& precond, std::span<const double> b,
                std::span<double> x, const SolveOptions& opt) {
    check_shapes(a, b, x);
    const auto n = static_cast<std::size_t>(a.nrows);
    const Dispatch d = opt.dispatch;
    const Reduction m = opt.reduction;

    const double bnorm = nrm2(b, d, m);
    if (bnorm == 0.0) {
        for (auto& v : x) {
            v = 0.0;
        }
        return SolveResult{0, 0.0, true};
    }

    std::vector<double> r(n), z(n), p(n), q(n);

    spmv(a, std::span<const double>(x), std::span<double>(r), d);
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = b[i] - r[i];
    }
    double relres = nrm2(r, d, m) / bnorm;
    if (relres <= opt.tol) {
        return SolveResult{0, relres, true};
    }

    precond.apply(r, z);
    p = z;
    double rho = dot(r, z, d, m); // (z_k, r_k)
    if (!(rho > 0.0) || !std::isfinite(rho)) {
        return SolveResult{0, relres, false}; // preconditioner not SPD
    }

    for (int it = 1; it <= opt.max_iter; ++it) {
        spmv(a, std::span<const double>(p), std::span<double>(q), d);
        const double pq = dot(p, q, d, m);
        if (!(pq > 0.0) || !std::isfinite(pq)) {
            return SolveResult{it, relres, false};
        }
        const double alpha = rho / pq;
        axpy(alpha, std::span<const double>(p), x, d);
        axpy(-alpha, std::span<const double>(q), std::span<double>(r), d);

        relres = std::sqrt(dot(r, r, d, m)) / bnorm;
        if (relres <= opt.tol) {
            return SolveResult{it, relres, true};
        }

        precond.apply(r, z);
        const double rho_new = dot(r, z, d, m);
        if (!(rho_new > 0.0) || !std::isfinite(rho_new)) {
            return SolveResult{it, relres, false};
        }
        // Flexible (Polak-Ribiere) beta: (z_{k+1}, r_{k+1} - r_k)/rho with
        // r_{k+1} - r_k = -alpha q, avoiding an extra stored vector.
        const double beta = -alpha * dot(z, q, d, m) / rho;
        aypx(beta, std::span<const double>(z), std::span<double>(p), d);
        rho = rho_new;
    }
    return SolveResult{opt.max_iter, relres, false};
}

} // namespace spume
