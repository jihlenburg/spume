// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

// Deterministic-reduction mode must make full solves bitwise reproducible
// across thread counts (numerics policy: the one place where bitwise
// comparison is required). SpMV and elementwise updates are row-owned and
// therefore always thread-count-invariant; deterministic mode fixes the
// remaining source of variation, the dot-product reductions.

#include <doctest/doctest.h>

#include <bit>
#include <cstdint>
#include <random>
#include <vector>

#include <omp.h>

#include "core/equilibrate.hpp"
#include "core/poisson.hpp"
#include "core/precond.hpp"
#include "core/solver.hpp"

namespace {

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(b[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("deterministic mode: identical bits across 1/4/16 threads") {
    omp_set_dynamic(0);

    const auto csr = spume::coo_to_csr(spume::gen::poisson7(12, 12, 12));
    const auto a = spume::sell_from_csr(csr);
    const auto n = static_cast<std::size_t>(a.nrows);

    std::mt19937_64 rng(31337);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> b(n);
    for (auto& v : b) {
        v = u(rng);
    }

    spume::SolveOptions opt;
    opt.tol = 1e-9;
    opt.dispatch = spume::Dispatch::openmp;
    opt.reduction = spume::Reduction::deterministic;

    SUBCASE("plain CG") {
        std::vector<double> x1(n, 0.0);
        omp_set_num_threads(1);
        const auto r1 = spume::cg(a, b, x1, opt);
        REQUIRE(r1.converged);

        for (int threads : {4, 16}) {
            omp_set_num_threads(threads);
            std::vector<double> xt(n, 0.0);
            const auto rt = spume::cg(a, b, xt, opt);
            CAPTURE(threads);
            CHECK(rt.converged);
            CHECK(rt.iterations == r1.iterations);
            CHECK(std::bit_cast<std::uint64_t>(rt.rel_residual) ==
                  std::bit_cast<std::uint64_t>(r1.rel_residual));
            CHECK(bitwise_equal(xt, x1));
        }
    }

    SUBCASE("FCG with FP32 Chebyshev preconditioner") {
        const auto op32 = spume::make_eq_operator<float>(csr);

        std::vector<double> x1(n, 0.0);
        omp_set_num_threads(1);
        const auto r1 =
            spume::fcg(a, spume::ChebyshevPrecond<float>(op32, {}, opt.dispatch), b, x1, opt);
        REQUIRE(r1.converged);

        for (int threads : {4, 16}) {
            omp_set_num_threads(threads);
            std::vector<double> xt(n, 0.0);
            const auto rt =
                spume::fcg(a, spume::ChebyshevPrecond<float>(op32, {}, opt.dispatch), b, xt, opt);
            CAPTURE(threads);
            CHECK(rt.converged);
            CHECK(rt.iterations == r1.iterations);
            CHECK(bitwise_equal(xt, x1));
        }
    }
}
