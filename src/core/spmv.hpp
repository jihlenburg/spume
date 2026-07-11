// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <cassert>
#include <span>

#include "core/sell.hpp"
#include "core/types.hpp"

namespace spume {

namespace detail {

// One SELL chunk: 8 output rows, streamed value/index arrays, per-lane
// accumulators. Row sums accumulate in fixed j-order, so SpMV output is
// bitwise identical for any thread count and both dispatch paths.
template<typename T>
inline void spmv_chunk(const Sell<T>& a, std::span<const T> x, std::span<T> y, index_t c) {
    const offset_t base = a.chunk_ptr[static_cast<std::size_t>(c)];
    const offset_t w = (a.chunk_ptr[static_cast<std::size_t>(c) + 1] - base) / kSellC;
    T acc[kSellC] = {};
    for (offset_t j = 0; j < w; ++j) {
        const offset_t off = base + j * kSellC;
        for (index_t r = 0; r < kSellC; ++r) {
            const auto pos = static_cast<std::size_t>(off + r);
            acc[r] += a.val[pos] * x[static_cast<std::size_t>(a.colidx[pos])];
        }
    }
    const index_t row0 = c * kSellC;
    for (index_t r = 0; r < kSellC; ++r) {
        if (row0 + r < a.nrows) { // tail chunk: lanes past nrows are dead
            y[static_cast<std::size_t>(row0 + r)] = acc[r];
        }
    }
}

} // namespace detail

// y = A x (overwrite). One thread owns whole chunks and therefore whole
// output rows — no scattered writes, no atomics. Default is the portable
// reference path (architecture invariant 4).
template<typename T>
void spmv(const Sell<T>& a, std::span<const T> x, std::span<T> y,
          Dispatch dispatch = Dispatch::reference) {
    assert(x.size() == static_cast<std::size_t>(a.ncols));
    assert(y.size() == static_cast<std::size_t>(a.nrows));
    const index_t nchunks = a.nchunks();
    if (dispatch == Dispatch::openmp) {
#pragma omp parallel for schedule(static)
        for (index_t c = 0; c < nchunks; ++c) {
            detail::spmv_chunk(a, x, y, c);
        }
    } else {
        for (index_t c = 0; c < nchunks; ++c) {
            detail::spmv_chunk(a, x, y, c);
        }
    }
}

} // namespace spume
