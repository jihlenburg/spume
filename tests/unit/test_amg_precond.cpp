// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <span>
#include <vector>

#include "core/amg_precond.hpp"
#include "core/equilibrate.hpp"
#include "core/formats.hpp"
#include "core/poisson.hpp"
#include "core/precond.hpp"
#include "core/sell.hpp"
#include "core/solver.hpp"

TEST_CASE("multi-level AMG preconditioner solves and accelerates fcg") {
    const spume::Csr a = spume::coo_to_csr(spume::gen::poisson7(32, 32, 32));
    const spume::Sell<double> A = spume::sell_from_csr(a);
    const auto n = static_cast<std::size_t>(a.nrows);

    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = 0.5 - static_cast<double>(i % 13) * 0.1; // deterministic RHS
    }

    spume::SolveOptions opt;
    opt.tol = 1e-8;
    opt.max_iter = 4000;

    // Baseline: flexible CG with the Chebyshev polynomial preconditioner alone.
    const auto eqop = spume::make_eq_operator<double>(a);
    const spume::ChebyshevPrecond<double> cheb(eqop);
    std::vector<double> x_cheb(n, 0.0);
    const spume::SolveResult r_cheb =
        spume::fcg(A, cheb, std::span<const double>(b), std::span<double>(x_cheb), opt);
    REQUIRE(r_cheb.converged);

    // Multi-level AMG (FP64 smoother).
    const spume::AmgPrecond<double> amg(a);
    CHECK(amg.num_levels() >= 3); // genuinely a hierarchy, not just two levels
    std::vector<double> x_amg(n, 0.0);
    const spume::SolveResult r_amg =
        spume::fcg(A, amg, std::span<const double>(b), std::span<double>(x_amg), opt);
    CHECK(r_amg.converged);
    CHECK(r_amg.iterations < r_cheb.iterations); // coarse correction accelerates
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(x_amg[i] == doctest::Approx(x_cheb[i]).epsilon(1e-5));
    }

    // Mixed precision (ADR-0002): FP32 smoother interior, same FP64 answer.
    const spume::AmgPrecond<float> amg32(a);
    std::vector<double> x32(n, 0.0);
    const spume::SolveResult r32 =
        spume::fcg(A, amg32, std::span<const double>(b), std::span<double>(x32), opt);
    CHECK(r32.converged);
    CHECK(r32.iterations <= r_cheb.iterations);
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(x32[i] == doctest::Approx(x_cheb[i]).epsilon(1e-5));
    }
}
