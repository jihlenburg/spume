// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <omp.h>

#include "core/reduce.hpp"
#include "core/vecops.hpp"

namespace {

// Mixed-magnitude, mixed-sign data: stresses summation-order sensitivity so
// that determinism failures actually show up as differing bits.
std::vector<double> make_data(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = u(rng) * std::pow(10.0, static_cast<double>(i % 9) - 4.0);
    }
    return v;
}

std::uint64_t bits(double v) {
    return std::bit_cast<std::uint64_t>(v);
}

} // namespace

TEST_CASE("dot: standard reference agrees with high-precision sum") {
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{1000}}) {
        auto x = make_data(n, 1);
        auto y = make_data(n, 2);
        long double ref = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            ref += static_cast<long double>(x[i]) * static_cast<long double>(y[i]);
        }
        const double got = spume::dot(x, y);
        if (n == 0) {
            CHECK(got == 0.0);
        } else {
            CHECK(got == doctest::Approx(static_cast<double>(ref)).epsilon(1e-12));
        }
    }
}

TEST_CASE("dot: deterministic mode is bitwise identical across dispatch and thread counts") {
    omp_set_dynamic(0);
    for (std::size_t n : {std::size_t{1}, std::size_t{15}, std::size_t{4095}, std::size_t{4096},
                          std::size_t{4097}, std::size_t{131085}}) {
        auto x = make_data(n, 3);
        auto y = make_data(n, 4);

        const double ref =
            spume::dot(x, y, spume::Dispatch::reference, spume::Reduction::deterministic);

        for (int threads : {1, 4, 16}) {
            omp_set_num_threads(threads);
            const double par =
                spume::dot(x, y, spume::Dispatch::openmp, spume::Reduction::deterministic);
            CAPTURE(n);
            CAPTURE(threads);
            CHECK(bits(par) == bits(ref));
        }
    }
}

TEST_CASE("dot: deterministic mode is accurate, not just reproducible") {
    const std::size_t n = 50000;
    auto x = make_data(n, 5);
    auto y = make_data(n, 6);
    long double ref = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        ref += static_cast<long double>(x[i]) * static_cast<long double>(y[i]);
    }
    const double det =
        spume::dot(x, y, spume::Dispatch::reference, spume::Reduction::deterministic);
    CHECK(det == doctest::Approx(static_cast<double>(ref)).epsilon(1e-12));
}

TEST_CASE("nrm2 matches sqrt(dot)") {
    auto x = make_data(777, 7);
    CHECK(spume::nrm2(x) == std::sqrt(spume::dot(x, x)));
}

TEST_CASE("vecops: axpy/aypx/axpby match scalar loops and are dispatch-invariant") {
    omp_set_dynamic(0);
    const std::size_t n = 10007;
    const auto x = make_data(n, 8);
    const auto y0 = make_data(n, 9);

    std::vector<double> expect(n);
    for (std::size_t i = 0; i < n; ++i) {
        expect[i] = y0[i] + 0.37 * x[i];
    }

    for (auto d : {spume::Dispatch::reference, spume::Dispatch::openmp}) {
        omp_set_num_threads(3);
        auto y = y0;
        spume::axpy(0.37, std::span<const double>(x), std::span<double>(y), d);
        CHECK(y == expect);
    }

    auto ya = y0;
    auto yb = y0;
    spume::aypx(2.0, std::span<const double>(x), std::span<double>(ya));
    spume::axpby(1.0, std::span<const double>(x), 2.0, std::span<double>(yb));
    CHECK(ya == yb); // aypx(a) == axpby(1, a) elementwise (identical arithmetic)
}
