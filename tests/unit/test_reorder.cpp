// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <numeric>
#include <random>
#include <span>
#include <vector>

#include "core/formats.hpp"
#include "core/poisson.hpp"
#include "core/reorder.hpp"
#include "core/sell.hpp"
#include "core/spmv.hpp"

namespace {

std::vector<spume::index_t> random_perm(spume::index_t n, unsigned seed) {
    std::vector<spume::index_t> p(static_cast<std::size_t>(n));
    std::iota(p.begin(), p.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(p.begin(), p.end(), rng);
    return p;
}

} // namespace

TEST_CASE("reorder preserves the operator: SpMV commutes with relabelling") {
    const spume::Csr a = spume::coo_to_csr(spume::gen::poisson7(6, 6, 6));
    const std::vector<spume::index_t> p = spume::rcm_order(a);
    const spume::Csr b = spume::reorder(a, p);
    REQUIRE(b.nrows == a.nrows);

    const auto n = static_cast<std::size_t>(a.nrows);
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = 0.3 + 0.7 * static_cast<double>(i % 11);
    }

    // y = A x
    const spume::Sell<double> as = spume::sell_from_csr(a);
    std::vector<double> y(n);
    spume::spmv(as, std::span<const double>(x), std::span<double>(y));

    // B (P^T x) must equal P^T (A x): b_y[k] == y[p[k]]
    std::vector<double> xp(n);
    for (std::size_t k = 0; k < n; ++k) {
        xp[k] = x[static_cast<std::size_t>(p[k])];
    }
    const spume::Sell<double> bs = spume::sell_from_csr(b);
    std::vector<double> by(n);
    spume::spmv(bs, std::span<const double>(xp), std::span<double>(by));

    for (std::size_t k = 0; k < n; ++k) {
        CHECK(by[k] == doctest::Approx(y[static_cast<std::size_t>(p[k])]));
    }
}

TEST_CASE("RCM restores locality: bandwidth of a scrambled operator collapses") {
    const spume::Csr structured = spume::coo_to_csr(spume::gen::poisson7(12, 12, 12));
    const spume::index_t bw_struct = spume::bandwidth(structured);

    // scramble the numbering — destroys locality, inflates bandwidth
    const spume::Csr scrambled = spume::reorder(structured, random_perm(structured.nrows, 7));
    const spume::index_t bw_scr = spume::bandwidth(scrambled);

    // RCM the scrambled operator back toward locality
    const std::vector<spume::index_t> p = spume::rcm_order(scrambled);
    const spume::Csr recovered = spume::reorder(scrambled, p);
    const spume::index_t bw_rcm = spume::bandwidth(recovered);

    CHECK(bw_scr > bw_struct * 3);   // scrambling genuinely wrecked it
    CHECK(bw_rcm < bw_scr / 3);      // RCM recovered most of the locality
    CHECK(bw_rcm <= bw_struct * 2);  // and lands near the natural bandwidth
}
