// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/formats.hpp"
#include "core/types.hpp"

namespace spume {

// Chunk height C of the SELL-C-8 format.
inline constexpr index_t kSellC = 8;

// SELL-C-8: Sliced ELLPACK with chunk height C = 8 (sigma = 1, i.e. no row
// sorting, in Milestone 0 — rows keep their natural order).
//
// Layout
// ------
// Rows are grouped into chunks of 8 consecutive rows. Chunk c stores
// w_c = max row length within the chunk, column-major inside the chunk:
// element j of the row r-within-chunk lives at
//
//     val[chunk_ptr[c] + j*8 + r],  colidx[same]
//
// so a SIMD lane per row streams the value/index arrays contiguously.
// Rows shorter than w_c are padded with (val = 0, col = 0): padding adds
// 0 * x[0] to the row sum. x must therefore contain finite values, which the
// solvers guarantee.
//
// Bytes-per-row model (documented traffic model used by the bench harness)
// ------------------------------------------------------------------------
// Per SpMV y = A x, streaming model — matrix streamed once, x assumed cached
// after one read, y written once, write-allocate ignored (the standard
// optimistic roofline model):
//
//     V(T) = stored() * (sizeof(T) + sizeof(index_t))   // values + indices
//          + ncols * sizeof(T)                          // x read once
//          + nrows * sizeof(T)                          // y written once
//
// where stored() >= nnz includes chunk padding (ratio stored()/nnz is the
// padding overhead; the bench harness reports it). Per row with
// k = nnz/nrows and pad = stored()/nnz:
//
//     V/nrows ≈ k * pad * (sizeof(T) + 4) + 2 * sizeof(T)
//
// FP32 coefficients cut the dominant value+index stream by (4+4)/(8+4) = 2/3
// and halve the vector traffic — the reason the FP32 preconditioner interior
// lowers the roofline.
template<typename T>
struct Sell {
    index_t nrows = 0;
    index_t ncols = 0;
    offset_t nnz = 0;                // logical nonzeros (padding excluded)
    std::vector<offset_t> chunk_ptr; // size nchunks()+1, element offsets
    std::vector<T> val;              // padded values, chunk-column-major
    std::vector<index_t> colidx;     // padded column indices (32-bit)

    index_t nchunks() const { return (nrows + kSellC - 1) / kSellC; }

    // Stored elements including padding.
    offset_t stored() const { return chunk_ptr.empty() ? 0 : chunk_ptr.back(); }

    // Padding overhead: stored()/nnz (1.0 = no padding).
    double padding_ratio() const {
        return nnz == 0 ? 1.0 : static_cast<double>(stored()) / static_cast<double>(nnz);
    }

    // Modelled bytes moved by one SpMV (see model above).
    double spmv_bytes() const {
        return static_cast<double>(stored()) * static_cast<double>(sizeof(T) + sizeof(index_t)) +
               static_cast<double>(ncols) * sizeof(T) + static_cast<double>(nrows) * sizeof(T);
    }
};

namespace detail {

// Build a Sell<T> from FP64 CSR, casting values to T at the final store.
//
// Numerics policy: this is the ONLY value-narrowing point in the core
// library. It is namespaced detail:: because demoting to FP32 without prior
// diagonal equilibration is forbidden — the public FP32 route is
// make_eq_operator<float> (equilibrate.hpp), which equilibrates in FP64
// first. Public FP64 conversion goes through sell_from_csr below.
template<typename T>
Sell<T> sell_from_csr_as(const Csr& a) {
    if (a.rowptr.size() != static_cast<std::size_t>(a.nrows) + 1) {
        throw std::invalid_argument("sell_from_csr: malformed rowptr");
    }
    Sell<T> out;
    out.nrows = a.nrows;
    out.ncols = a.ncols;
    out.nnz = a.nnz();

    const index_t nchunks = out.nchunks();
    out.chunk_ptr.assign(static_cast<std::size_t>(nchunks) + 1, 0);
    for (index_t c = 0; c < nchunks; ++c) {
        offset_t w = 0;
        for (index_t r = c * kSellC; r < std::min<index_t>((c + 1) * kSellC, a.nrows); ++r) {
            const offset_t len =
                a.rowptr[static_cast<std::size_t>(r) + 1] - a.rowptr[static_cast<std::size_t>(r)];
            w = std::max(w, len);
        }
        out.chunk_ptr[static_cast<std::size_t>(c) + 1] =
            out.chunk_ptr[static_cast<std::size_t>(c)] + w * kSellC;
    }

    out.val.assign(static_cast<std::size_t>(out.stored()), T{0});
    out.colidx.assign(static_cast<std::size_t>(out.stored()), 0);

    for (index_t r = 0; r < a.nrows; ++r) {
        const index_t c = r / kSellC;
        const index_t lane = r % kSellC;
        const offset_t base = out.chunk_ptr[static_cast<std::size_t>(c)];
        const offset_t lo = a.rowptr[static_cast<std::size_t>(r)];
        const offset_t hi = a.rowptr[static_cast<std::size_t>(r) + 1];
        for (offset_t k = lo; k < hi; ++k) {
            const offset_t pos = base + (k - lo) * kSellC + lane;
            out.val[static_cast<std::size_t>(pos)] =
                static_cast<T>(a.val[static_cast<std::size_t>(k)]);
            out.colidx[static_cast<std::size_t>(pos)] = a.col[static_cast<std::size_t>(k)];
        }
    }
    return out;
}

} // namespace detail

// Public FP64 converters.
Sell<double> sell_from_csr(const Csr& a);
Sell<double> sell_from_coo(const Coo& a);

} // namespace spume
