// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

// Milestone 0 acceptance: a flexible FP64 outer Krylov solver with an FP32
// preconditioner converges to FP64-accurate solutions.
//
// Equivalence-class check (numerics policy): two converged solutions may
// differ at rounding-reordering level only. With true residuals
// r_a = b - A x_a, r_b = b - A x_b it holds exactly that
// x_a - x_b = A^{-1}(r_b - r_a), hence
//
//     ||x_a - x_b||_2 <= (||r_a||_2 + ||r_b||_2) / lambda_min(A).
//
// We assert this theorem-backed bound (with 10% slack for the FP64
// evaluation of the residuals themselves) — never bitwise equality.
// Iteration-count parity is asserted at 20% between the FP64 and FP32
// variants of the SAME preconditioner: precision of the preconditioner is
// the only variable.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "core/equilibrate.hpp"
#include "core/poisson.hpp"
#include "core/precond.hpp"
#include "core/reduce.hpp"
#include "core/solver.hpp"
#include "core/spmv.hpp"

#include "mm_io.hpp"

namespace {

std::vector<double> random_rhs(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> b(n);
    for (auto& v : b) {
        v = u(rng);
    }
    return b;
}

double true_resnorm(const spume::Sell<double>& a, std::span<const double> b,
                    std::span<const double> x) {
    std::vector<double> r(b.size());
    spume::spmv(a, x, std::span<double>(r));
    for (std::size_t i = 0; i < r.size(); ++i) {
        r[i] = b[i] - r[i];
    }
    return spume::nrm2(r);
}

double diff_norm(std::span<const double> a, std::span<const double> b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

void check_equivalence_class(const spume::Sell<double>& a, std::span<const double> b,
                             std::span<const double> xa, std::span<const double> xb,
                             double lambda_min) {
    const double bound = 1.1 * (true_resnorm(a, b, xa) + true_resnorm(a, b, xb)) / lambda_min;
    const double diff = diff_norm(xa, xb);
    CHECK(diff <= bound);
    // The bound must also be meaningfully tight in relative terms, or the
    // check would be vacuous.
    const double xnorm = spume::nrm2(xa);
    CHECK(bound / xnorm < 1e-5);
}

void check_iteration_parity(int i64, int i32) {
    const int hi = std::max(i64, i32);
    CHECK(std::abs(i64 - i32) <= std::max(static_cast<int>(0.2 * hi), 2));
}

} // namespace

TEST_CASE("Poisson: mixed-precision FCG matches pure-FP64 within the equivalence class") {
    struct Case {
        spume::index_t nx, ny, nz;
    };
    for (const Case cse : {Case{24, 24, 24}, Case{32, 32, 32}}) {
        CAPTURE(cse.nx);
        const auto csr = spume::coo_to_csr(spume::gen::poisson7(cse.nx, cse.ny, cse.nz));
        const auto a = spume::sell_from_csr(csr);
        const auto n = static_cast<std::size_t>(a.nrows);
        const auto b = random_rhs(n, 1000 + static_cast<std::uint64_t>(cse.nx));
        const double lmin = spume::gen::poisson7_bounds(cse.nx, cse.ny, cse.nz).lambda_min;

        spume::SolveOptions opt;
        opt.tol = 1e-10;
        opt.dispatch = spume::Dispatch::openmp;

        std::vector<double> x_ref(n, 0.0);
        const auto r_ref = spume::cg(a, b, x_ref, opt);
        REQUIRE(r_ref.converged);

        const auto op64 = spume::make_eq_operator<double>(csr);
        const auto op32 = spume::make_eq_operator<float>(csr);

        SUBCASE("Jacobi: FP64 vs FP32") {
            std::vector<double> x64(n, 0.0), x32(n, 0.0);
            const auto s64 =
                spume::fcg(a, spume::JacobiPrecond<double>(op64, opt.dispatch), b, x64, opt);
            const auto s32 =
                spume::fcg(a, spume::JacobiPrecond<float>(op32, opt.dispatch), b, x32, opt);
            REQUIRE(s64.converged);
            REQUIRE(s32.converged);
            check_iteration_parity(s64.iterations, s32.iterations);
            check_equivalence_class(a, b, x64, x32, lmin);
            check_equivalence_class(a, b, x_ref, x32, lmin);
        }

        SUBCASE("Chebyshev: FP64 vs FP32") {
            const spume::ChebyshevOptions copt{5, 30.0};
            std::vector<double> x64(n, 0.0), x32(n, 0.0);
            const auto s64 = spume::fcg(
                a, spume::ChebyshevPrecond<double>(op64, copt, opt.dispatch), b, x64, opt);
            const auto s32 = spume::fcg(a, spume::ChebyshevPrecond<float>(op32, copt, opt.dispatch),
                                        b, x32, opt);
            REQUIRE(s64.converged);
            REQUIRE(s32.converged);
            check_iteration_parity(s64.iterations, s32.iterations);
            check_equivalence_class(a, b, x64, x32, lmin);
            check_equivalence_class(a, b, x_ref, x32, lmin);
            // Preconditioning must actually help vs plain CG.
            CHECK(s32.iterations < r_ref.iterations / 2);
        }
    }
}

TEST_CASE("unstructured fixture: mixed-precision FCG matches FP64 within the class") {
    const auto coo = spume::testing::read_matrix_market_file(std::string(SPUME_TEST_DATA_DIR) +
                                                             "/unstructured_rgg.mtx");
    const auto csr = spume::coo_to_csr(coo);
    const auto a = spume::sell_from_csr(csr);
    const auto n = static_cast<std::size_t>(a.nrows);
    const auto b = random_rhs(n, 4242);
    // lambda_min >= 0.1 by construction of the fixture (L + 0.1 I, L PSD) —
    // documented in the .mtx header and the generator script.
    const double lmin = 0.1;

    spume::SolveOptions opt;
    opt.tol = 1e-10;
    opt.dispatch = spume::Dispatch::openmp;

    std::vector<double> x_ref(n, 0.0);
    const auto r_ref = spume::cg(a, b, x_ref, opt);
    REQUIRE(r_ref.converged);

    const auto op64 = spume::make_eq_operator<double>(csr);
    const auto op32 = spume::make_eq_operator<float>(csr);
    const spume::ChebyshevOptions copt{4, 20.0};

    std::vector<double> x64(n, 0.0), x32(n, 0.0);
    const auto s64 =
        spume::fcg(a, spume::ChebyshevPrecond<double>(op64, copt, opt.dispatch), b, x64, opt);
    const auto s32 =
        spume::fcg(a, spume::ChebyshevPrecond<float>(op32, copt, opt.dispatch), b, x32, opt);
    REQUIRE(s64.converged);
    REQUIRE(s32.converged);

    check_iteration_parity(s64.iterations, s32.iterations);
    check_equivalence_class(a, b, x64, x32, lmin);
    check_equivalence_class(a, b, x_ref, x32, lmin);
}
