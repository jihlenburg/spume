// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <random>
#include <vector>

#include "core/formats.hpp"
#include "core/sell.hpp"
#include "core/spmv.hpp"

namespace {

// Dense reconstruction of any format for small-matrix ground truth.
std::vector<double> dense_from_csr(const spume::Csr& a) {
    std::vector<double> d(static_cast<std::size_t>(a.nrows) * static_cast<std::size_t>(a.ncols),
                          0.0);
    for (spume::index_t r = 0; r < a.nrows; ++r) {
        for (auto k = a.rowptr[static_cast<std::size_t>(r)];
             k < a.rowptr[static_cast<std::size_t>(r) + 1]; ++k) {
            d[static_cast<std::size_t>(r) * static_cast<std::size_t>(a.ncols) +
              static_cast<std::size_t>(a.col[static_cast<std::size_t>(k)])] +=
                a.val[static_cast<std::size_t>(k)];
        }
    }
    return d;
}

template<typename T>
std::vector<double> dense_from_sell(const spume::Sell<T>& a) {
    std::vector<double> d(static_cast<std::size_t>(a.nrows) * static_cast<std::size_t>(a.ncols),
                          0.0);
    for (spume::index_t c = 0; c < a.nchunks(); ++c) {
        const auto base = a.chunk_ptr[static_cast<std::size_t>(c)];
        const auto w = (a.chunk_ptr[static_cast<std::size_t>(c) + 1] - base) / spume::kSellC;
        for (spume::offset_t j = 0; j < w; ++j) {
            for (spume::index_t r = 0; r < spume::kSellC; ++r) {
                const spume::index_t row = c * spume::kSellC + r;
                if (row >= a.nrows) {
                    continue;
                }
                const auto pos = static_cast<std::size_t>(base + j * spume::kSellC + r);
                d[static_cast<std::size_t>(row) * static_cast<std::size_t>(a.ncols) +
                  static_cast<std::size_t>(a.colidx[pos])] += static_cast<double>(a.val[pos]);
            }
        }
    }
    return d;
}

// Random sparse matrix with ragged rows (including empty ones).
spume::Coo random_coo(spume::index_t nrows, spume::index_t ncols, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<spume::index_t> col(0, ncols - 1);
    std::uniform_int_distribution<int> rowlen(0, 11);
    std::uniform_real_distribution<double> v(-2.0, 2.0);
    spume::Coo a;
    a.nrows = nrows;
    a.ncols = ncols;
    for (spume::index_t r = 0; r < nrows; ++r) {
        const int len = rowlen(rng);
        for (int k = 0; k < len; ++k) {
            a.row.push_back(r);
            a.col.push_back(col(rng));
            a.val.push_back(v(rng));
        }
    }
    return a;
}

} // namespace

TEST_CASE("coo_to_csr sorts rows by column and sums duplicates") {
    spume::Coo a;
    a.nrows = 3;
    a.ncols = 4;
    // Unsorted, with an intentional duplicate at (1,2) and an empty row 2.
    a.row = {1, 0, 1, 1, 0};
    a.col = {2, 3, 0, 2, 1};
    a.val = {5.0, 1.0, 2.0, 0.5, 3.0};

    const auto c = spume::coo_to_csr(a);
    CHECK(c.nnz() == 4);
    CHECK(c.rowptr == std::vector<spume::offset_t>{0, 2, 4, 4});
    CHECK(c.col == std::vector<spume::index_t>{1, 3, 0, 2});
    CHECK(c.val == std::vector<double>{3.0, 1.0, 2.0, 5.5});
}

TEST_CASE("coo_to_csr rejects out-of-range indices") {
    spume::Coo a;
    a.nrows = 2;
    a.ncols = 2;
    a.row = {0};
    a.col = {2}; // out of range
    a.val = {1.0};
    CHECK_THROWS_AS((void)spume::coo_to_csr(a), std::invalid_argument);
}

TEST_CASE("SELL-C-8 reproduces CSR exactly, including ragged and tail chunks") {
    for (spume::index_t nrows : {1, 7, 8, 9, 63, 64, 65, 200}) {
        const auto coo = random_coo(nrows, 50, static_cast<std::uint64_t>(nrows));
        const auto csr = spume::coo_to_csr(coo);
        const auto sell = spume::sell_from_csr(csr);

        CAPTURE(nrows);
        CHECK(sell.nnz == csr.nnz());
        CHECK(sell.nchunks() == (nrows + spume::kSellC - 1) / spume::kSellC);
        CHECK(sell.stored() >= sell.nnz);
        CHECK(dense_from_sell(sell) == dense_from_csr(csr));
    }
}

TEST_CASE("sell_from_coo equals sell_from_csr(coo_to_csr)") {
    const auto coo = random_coo(100, 40, 77);
    const auto a = spume::sell_from_coo(coo);
    const auto b = spume::sell_from_csr(spume::coo_to_csr(coo));
    CHECK(a.val == b.val);
    CHECK(a.colidx == b.colidx);
    CHECK(a.chunk_ptr == b.chunk_ptr);
}

TEST_CASE("spmv_bytes model counts values, indices, and both vectors") {
    const auto coo = random_coo(64, 64, 42);
    const auto a = spume::sell_from_csr(spume::coo_to_csr(coo));
    const double expect = static_cast<double>(a.stored()) * (8.0 + 4.0) + 64.0 * 8.0 + 64.0 * 8.0;
    CHECK(a.spmv_bytes() == expect);
}

TEST_CASE("SpMV: reference matches dense product; OpenMP is bitwise identical") {
    const auto coo = random_coo(133, 133, 9);
    const auto csr = spume::coo_to_csr(coo);
    const auto a = spume::sell_from_csr(csr);
    const auto dense = dense_from_csr(csr);

    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> x(133);
    for (auto& v : x) {
        v = u(rng);
    }

    std::vector<double> y_ref(133), y_omp(133), y_dense(133, 0.0);
    for (std::size_t r = 0; r < 133; ++r) {
        for (std::size_t c = 0; c < 133; ++c) {
            y_dense[r] += dense[r * 133 + c] * x[c];
        }
    }

    spume::spmv(a, std::span<const double>(x), std::span<double>(y_ref));
    spume::spmv(a, std::span<const double>(x), std::span<double>(y_omp), spume::Dispatch::openmp);

    CHECK(y_ref == y_omp); // row-owned accumulation: bitwise equal
    for (std::size_t r = 0; r < 133; ++r) {
        CHECK(y_ref[r] == doctest::Approx(y_dense[r]).epsilon(1e-12));
    }
}
