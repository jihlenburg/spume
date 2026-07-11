// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <random>
#include <vector>

#include "core/equilibrate.hpp"
#include "core/poisson.hpp"
#include "core/precond.hpp"
#include "core/reduce.hpp"
#include "core/solver.hpp"
#include "core/spmv.hpp"

namespace {

struct Problem {
    spume::Csr csr;
    spume::Sell<double> a;
    std::vector<double> b;
};

Problem poisson_problem(spume::index_t nx, spume::index_t ny, spume::index_t nz,
                        std::uint64_t seed) {
    Problem p;
    p.csr = spume::coo_to_csr(spume::gen::poisson7(nx, ny, nz));
    p.a = spume::sell_from_csr(p.csr);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    p.b.resize(static_cast<std::size_t>(p.a.nrows));
    for (auto& v : p.b) {
        v = u(rng);
    }
    return p;
}

// True residual, recomputed from scratch in FP64.
double true_relres(const spume::Sell<double>& a, std::span<const double> b,
                   std::span<const double> x) {
    std::vector<double> r(b.size());
    spume::spmv(a, x, std::span<double>(r));
    for (std::size_t i = 0; i < r.size(); ++i) {
        r[i] = b[i] - r[i];
    }
    return spume::nrm2(r) / spume::nrm2(b);
}

} // namespace

TEST_CASE("CG reference converges on Poisson and the true residual agrees") {
    auto p = poisson_problem(8, 8, 8, 21);
    std::vector<double> x(p.b.size(), 0.0);
    spume::SolveOptions opt;
    opt.tol = 1e-10;

    const auto res = spume::cg(p.a, p.b, x, opt);
    CHECK(res.converged);
    CHECK(res.iterations > 0);
    CHECK(res.iterations < 200);
    // Recursive and true residual must agree at this conditioning.
    CHECK(true_relres(p.a, p.b, x) <= 2.0 * opt.tol);
}

TEST_CASE("CG handles b = 0 and a converged initial guess") {
    auto p = poisson_problem(4, 4, 4, 22);

    std::vector<double> zero(p.b.size(), 0.0);
    std::vector<double> x(p.b.size(), 0.5);
    const auto res0 = spume::cg(p.a, zero, x, {});
    CHECK(res0.converged);
    CHECK(res0.iterations == 0);
    for (auto v : x) {
        CHECK(v == 0.0);
    }

    // Solve, then restart from the solution: 0 iterations.
    std::vector<double> y(p.b.size(), 0.0);
    spume::SolveOptions opt;
    opt.tol = 1e-8;
    (void)spume::cg(p.a, p.b, y, opt);
    const auto res1 = spume::cg(p.a, p.b, y, opt);
    CHECK(res1.converged);
    CHECK(res1.iterations == 0);
}

TEST_CASE("FCG with identity preconditioner tracks plain CG") {
    auto p = poisson_problem(8, 8, 8, 23);
    spume::SolveOptions opt;
    opt.tol = 1e-10;

    std::vector<double> x_cg(p.b.size(), 0.0), x_fcg(p.b.size(), 0.0);
    const auto rc = spume::cg(p.a, p.b, x_cg, opt);
    const auto rf = spume::fcg(p.a, spume::IdentityPrecond{}, p.b, x_fcg, opt);

    CHECK(rc.converged);
    CHECK(rf.converged);
    // Same Krylov space, same arithmetic up to the beta formula: iteration
    // counts must be very close.
    CHECK(std::abs(rc.iterations - rf.iterations) <= 2);
}

TEST_CASE("FCG with FP32 Jacobi and Chebyshev preconditioners converges to FP64 accuracy") {
    auto p = poisson_problem(10, 9, 8, 24);
    spume::SolveOptions opt;
    opt.tol = 1e-10;

    std::vector<double> x_ref(p.b.size(), 0.0);
    const auto rc = spume::cg(p.a, p.b, x_ref, opt);
    REQUIRE(rc.converged);

    const auto op32 = spume::make_eq_operator<float>(p.csr);

    SUBCASE("Jacobi FP32") {
        std::vector<double> x(p.b.size(), 0.0);
        const auto r = spume::fcg(p.a, spume::JacobiPrecond<float>(op32), p.b, x, opt);
        CHECK(r.converged);
        CHECK(true_relres(p.a, p.b, x) <= 2.0 * opt.tol);
    }
    SUBCASE("Chebyshev FP32") {
        std::vector<double> x(p.b.size(), 0.0);
        const auto r = spume::fcg(p.a, spume::ChebyshevPrecond<float>(op32), p.b, x, opt);
        CHECK(r.converged);
        CHECK(true_relres(p.a, p.b, x) <= 2.0 * opt.tol);

        // The whole point: Chebyshev cuts outer iterations vs plain CG.
        CHECK(r.iterations < rc.iterations / 2);
    }
}

TEST_CASE("solvers reject shape mismatches") {
    auto p = poisson_problem(3, 3, 3, 25);
    std::vector<double> shortb(5, 1.0), x(p.b.size(), 0.0);
    CHECK_THROWS_AS((void)spume::cg(p.a, shortb, x, {}), std::invalid_argument);
}
