// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <span>
#include <vector>

#include "core/amg.hpp"
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

TEST_CASE("K-cycle accelerates the V-cycle and reaches the same answer") {
    const spume::Csr a = spume::coo_to_csr(spume::gen::poisson7(32, 32, 32));
    const spume::Sell<double> A = spume::sell_from_csr(a);
    const auto n = static_cast<std::size_t>(a.nrows);

    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = 0.5 - static_cast<double>(i % 13) * 0.1;
    }
    spume::SolveOptions opt;
    opt.tol = 1e-8;
    opt.max_iter = 4000;

    // Plain V-cycle (kcycle = false) vs Krylov-accelerated K-cycle. The K-cycle
    // wraps the coarse correction in a short flexible-CG, recovering a stronger
    // (mesh-independent) coarse-grid correction on the same unsmoothed-
    // aggregation hierarchy — so it must not need MORE outer iterations, and on
    // this anisotropy-free operator reaches the identical FP64 answer.
    const spume::AmgPrecond<double> vcycle(
        a, {}, 200, 20, 1e-2, 500, spume::Dispatch::reference, /*kcycle=*/false);
    std::vector<double> xv(n, 0.0);
    const spume::SolveResult rv =
        spume::fcg(A, vcycle, std::span<const double>(b), std::span<double>(xv), opt);
    REQUIRE(rv.converged);

    const spume::AmgPrecond<double> kcyc(
        a, {}, 200, 20, 1e-2, 500, spume::Dispatch::reference, /*kcycle=*/true);
    std::vector<double> xk(n, 0.0);
    const spume::SolveResult rk =
        spume::fcg(A, kcyc, std::span<const double>(b), std::span<double>(xk), opt);
    CHECK(rk.converged);
    CHECK(rk.iterations <= rv.iterations); // Krylov coarse correction never hurts
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(xk[i] == doctest::Approx(xv[i]).epsilon(1e-5));
    }

    // FP32 K-cycle: the full production path (mixed precision + K-cycle) still
    // lands on the FP64 answer (the outer-Krylov firewall, ADR-0002).
    const spume::AmgPrecond<float> kcyc32(
        a, {}, 200, 20, 1e-2, 500, spume::Dispatch::reference, /*kcycle=*/true);
    std::vector<double> xk32(n, 0.0);
    const spume::SolveResult rk32 =
        spume::fcg(A, kcyc32, std::span<const double>(b), std::span<double>(xk32), opt);
    CHECK(rk32.converged);
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(xk32[i] == doctest::Approx(xv[i]).epsilon(1e-5));
    }
}

TEST_CASE("AMG accepts an externally-supplied hierarchy (the GAMG-reuse path)") {
    const spume::Csr a = spume::coo_to_csr(spume::gen::poisson7(24, 24, 24));
    const spume::Sell<double> A = spume::sell_from_csr(a);
    const auto n = static_cast<std::size_t>(a.nrows);
    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = 0.5 - static_cast<double>(i % 13) * 0.1;
    }

    // Build a hierarchy externally (here from aggregate(), standing in for an
    // OpenFOAM GAMGAgglomeration) and feed it to the external constructor.
    std::vector<spume::Aggregation> hierarchy;
    spume::Csr cur = a;
    for (int lvl = 0; lvl < 4 && cur.nrows > 100; ++lvl) {
        spume::Aggregation agg = spume::aggregate(cur);
        if (agg.ncoarse >= cur.nrows) {
            break;
        }
        spume::Csr coarse = spume::galerkin(cur, agg);
        hierarchy.push_back(std::move(agg));
        cur = std::move(coarse);
    }
    REQUIRE(hierarchy.size() >= 2);

    const spume::AmgPrecond<float> amg(a, hierarchy);
    CHECK(amg.num_levels() == static_cast<int>(hierarchy.size()) + 1);

    spume::SolveOptions opt;
    opt.tol = 1e-8;
    opt.max_iter = 3000;
    std::vector<double> x(n, 0.0);
    const spume::SolveResult r =
        spume::fcg(A, amg, std::span<const double>(b), std::span<double>(x), opt);
    CHECK(r.converged);

    // matches the FP64 reference solve
    const auto eqop = spume::make_eq_operator<double>(a);
    const spume::ChebyshevPrecond<double> cheb(eqop);
    std::vector<double> xref(n, 0.0);
    spume::fcg(A, cheb, std::span<const double>(b), std::span<double>(xref), opt);
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(x[i] == doctest::Approx(xref[i]).epsilon(1e-5));
    }
}
