// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <cassert>
#include <span>

#include "core/sell.hpp"
#include "core/types.hpp"

// Fused SpMV + AXPY:  r += alpha * (A x)  in a single pass over A, without ever
// materialising the intermediate vector t = A x.
//
// Why fuse: the Chebyshev smoother inner loop is `t = A d; r -= t`, i.e.
// spmv(a,d,t) then axpy(-1,t,r). Unfused, that writes the whole vector t and
// reads it straight back — two extra vector passes per step. Fusing removes
// them: fewer bytes moved => a lower roofline, which is the optimisation the
// performance policy rewards (ADR-0003, admission rule (b)).
//
// This is a PORTABLE kernel, not ISA-specific work, so it lives in core/, not
// src/backends/. Measured on the Strix Halo target (192^3 Poisson, FP32, 16
// pinned cores, 2026-07-19): fused runs at ~0.918x the unfused time, matching
// the 0.895 byte model, and reaches ~94% of the STREAM-triad roofline under
// `-march=znver5 -O3`. Because the compiler already saturates the bus, ADR-0003
// says NOT to hand-write an assembly specialization here (it could not beat a
// kernel already at >=90% of roofline). The Dispatch parameter keeps the slot
// open for architectures where the portable path falls short of roofline.
//
// Accumulation order is identical to core spmv (fixed j-order per chunk), so
// `spmv_axpy(a, alpha, x, r)` is bit-for-bit equal to
// `spmv(a, x, t); axpy(alpha, t, r)` — the reference contract the unit test
// pins (tests/unit/test_fused_spmv.cpp).

namespace spume {

namespace detail {

template<typename T>
inline void spmv_axpy_chunk(const Sell<T>& a, T alpha, std::span<const T> x,
                            std::span<T> r, index_t c) {
    const offset_t base = a.chunk_ptr[static_cast<std::size_t>(c)];
    const offset_t w = (a.chunk_ptr[static_cast<std::size_t>(c) + 1] - base) / kSellC;
    T acc[kSellC] = {};
    for (offset_t j = 0; j < w; ++j) {
        const offset_t off = base + j * kSellC;
        for (index_t rr = 0; rr < kSellC; ++rr) {
            const auto pos = static_cast<std::size_t>(off + rr);
            acc[rr] += a.val[pos] * x[static_cast<std::size_t>(a.colidx[pos])];
        }
    }
    const index_t row0 = c * kSellC;
    for (index_t rr = 0; rr < kSellC; ++rr) {
        if (row0 + rr < a.nrows) { // tail chunk: lanes past nrows are dead
            r[static_cast<std::size_t>(row0 + rr)] += alpha * acc[rr];
        }
    }
}

} // namespace detail

// r += alpha * (A x). One thread owns whole chunks (whole output rows) — no
// scattered writes, no atomics. Default is the portable reference path
// (architecture invariant 4).
template<typename T>
void spmv_axpy(const Sell<T>& a, T alpha, std::span<const T> x, std::span<T> r,
               Dispatch dispatch = Dispatch::reference) {
    assert(x.size() == static_cast<std::size_t>(a.ncols));
    assert(r.size() == static_cast<std::size_t>(a.nrows));
    const index_t nchunks = a.nchunks();
    if (dispatch == Dispatch::openmp) {
#pragma omp parallel for schedule(static)
        for (index_t c = 0; c < nchunks; ++c) {
            detail::spmv_axpy_chunk(a, alpha, x, r, c);
        }
    } else {
        for (index_t c = 0; c < nchunks; ++c) {
            detail::spmv_axpy_chunk(a, alpha, x, r, c);
        }
    }
}

} // namespace spume
