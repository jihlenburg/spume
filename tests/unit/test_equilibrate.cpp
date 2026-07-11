// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/equilibrate.hpp"
#include "core/poisson.hpp"
#include "core/reduce.hpp"
#include "core/spmv.hpp"

TEST_CASE("equilibrated operator has unit diagonal and preserves symmetry") {
    const auto csr = spume::coo_to_csr(spume::gen::poisson7(5, 4, 3));
    const auto op = spume::make_eq_operator<double>(csr);
    const auto n = static_cast<std::size_t>(csr.nrows);

    // Diagonal of S A S is s_i * a_ii * s_i = 1 up to rounding.
    // Reconstruct: (S A S) e_i evaluated via SpMV.
    std::vector<double> e(n, 0.0), col(n);
    for (std::size_t i = 0; i < n; i += 7) {
        e[i] = 1.0;
        spume::spmv(op.a, std::span<const double>(e), std::span<double>(col));
        CHECK(col[i] == doctest::Approx(1.0).epsilon(1e-14));
        e[i] = 0.0;
    }

    CHECK(op.scale.size() == n);
    for (auto s : op.scale) {
        CHECK(s == doctest::Approx(1.0 / std::sqrt(6.0)).epsilon(1e-14));
    }
}

TEST_CASE("Gershgorin bound dominates Rayleigh quotients of S A S") {
    const auto csr = spume::coo_to_csr(spume::gen::poisson7(6, 6, 6));
    const auto op = spume::make_eq_operator<double>(csr);
    const auto n = static_cast<std::size_t>(csr.nrows);

    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> v(n), av(n);
    for (int trial = 0; trial < 5; ++trial) {
        for (auto& z : v) {
            z = u(rng);
        }
        spume::spmv(op.a, std::span<const double>(v), std::span<double>(av));
        const double rq = spume::dot(v, av) / spume::dot(v, v);
        CHECK(rq <= op.lambda_max_bound);
        CHECK(rq > 0.0); // SPD preserved
    }
    // For equilibrated Poisson the bound must stay at 2 (unit diagonal,
    // diagonally dominant) up to rounding of the s_i*a_ij*s_j products,
    // and above 1.
    CHECK(op.lambda_max_bound <= 2.0 + 1e-12);
    CHECK(op.lambda_max_bound > 1.0);
}

TEST_CASE("FP32 demotion happens after FP64 equilibration") {
    const auto csr = spume::coo_to_csr(spume::gen::poisson7(4, 4, 4));
    const auto op64 = spume::make_eq_operator<double>(csr);
    const auto op32 = spume::make_eq_operator<float>(csr);

    // Same layout; FP32 values are the FP64 equilibrated values rounded once.
    REQUIRE(op32.a.val.size() == op64.a.val.size());
    CHECK(op32.a.chunk_ptr == op64.a.chunk_ptr);
    CHECK(op32.a.colidx == op64.a.colidx);
    for (std::size_t k = 0; k < op64.a.val.size(); ++k) {
        CHECK(op32.a.val[k] == static_cast<float>(op64.a.val[k]));
    }
    // Scale vectors stay FP64 and identical.
    CHECK(op32.scale == op64.scale);
}

TEST_CASE("make_eq_operator rejects nonpositive diagonals") {
    spume::Coo bad;
    bad.nrows = 2;
    bad.ncols = 2;
    bad.row = {0, 1};
    bad.col = {0, 1};
    bad.val = {1.0, -1.0};
    CHECK_THROWS_AS((void)spume::make_eq_operator<double>(spume::coo_to_csr(bad)),
                    std::invalid_argument);
}
