// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "core/formats.hpp"
#include "core/poisson.hpp"

TEST_CASE("poisson7 structure: size, nnz, symmetry, diagonal dominance") {
    const spume::index_t nx = 5, ny = 4, nz = 3;
    const auto coo = spume::gen::poisson7(nx, ny, nz);
    const auto a = spume::coo_to_csr(coo);
    const auto n = static_cast<spume::offset_t>(nx) * ny * nz;

    CHECK(a.nrows == n);
    // nnz = 7n - 2*(faces): each dimension removes 2 * (product of others).
    const spume::offset_t expect_nnz =
        7 * n - 2 * (static_cast<spume::offset_t>(ny) * nz + static_cast<spume::offset_t>(nx) * nz +
                     static_cast<spume::offset_t>(nx) * ny);
    CHECK(a.nnz() == expect_nnz);

    // Symmetry and weak diagonal dominance, row by row.
    for (spume::index_t r = 0; r < a.nrows; ++r) {
        double offsum = 0.0;
        for (auto k = a.rowptr[static_cast<std::size_t>(r)];
             k < a.rowptr[static_cast<std::size_t>(r) + 1]; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            const auto c = a.col[kk];
            if (c == r) {
                CHECK(a.val[kk] == 6.0);
            } else {
                CHECK(a.val[kk] == -1.0);
                offsum += 1.0;
                // symmetric partner exists with the same value
                bool found = false;
                for (auto k2 = a.rowptr[static_cast<std::size_t>(c)];
                     k2 < a.rowptr[static_cast<std::size_t>(c) + 1]; ++k2) {
                    if (a.col[static_cast<std::size_t>(k2)] == r) {
                        found = a.val[static_cast<std::size_t>(k2)] == a.val[kk];
                        break;
                    }
                }
                CHECK(found);
            }
        }
        CHECK(offsum <= 6.0);
    }
}

TEST_CASE("poisson7_bounds brackets the spectrum sensibly") {
    const auto b = spume::gen::poisson7_bounds(8, 8, 8);
    CHECK(b.lambda_min > 0.0);
    CHECK(b.lambda_max < 12.0);
    CHECK(b.lambda_min < b.lambda_max);

    // 1D sanity: for n=1 the only eigenvalue per dimension is 4 sin^2(pi/4)=2.
    const auto b1 = spume::gen::poisson7_bounds(1, 1, 1);
    CHECK(b1.lambda_min == doctest::Approx(6.0));
    CHECK(b1.lambda_max == doctest::Approx(6.0));
}
