// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include "bridge/ldu_to_sell.hpp"

#include <cstddef>
#include <stdexcept>

#include "core/formats.hpp"

namespace spume {

Sell<double> assemble_sell(std::span<const int> lowerAddr,
                           std::span<const int> upperAddr,
                           std::span<const double> diag,
                           std::span<const double> upper,
                           std::span<const double> lower,
                           int nCells) {
    if (nCells < 0) {
        throw std::invalid_argument("assemble_sell: negative nCells");
    }
    if (diag.size() != static_cast<std::size_t>(nCells)) {
        throw std::invalid_argument("assemble_sell: diag size != nCells");
    }
    const std::size_t nFaces = lowerAddr.size();
    if (upperAddr.size() != nFaces || upper.size() != nFaces) {
        throw std::invalid_argument("assemble_sell: addressing/upper size mismatch");
    }
    const bool symmetric = lower.empty();
    if (!symmetric && lower.size() != nFaces) {
        throw std::invalid_argument("assemble_sell: lower size != nFaces");
    }

    // Build a COO with one diagonal entry per cell and two off-diagonal
    // entries per face, then let the core assemble/validate the SELL operator
    // (coo_to_csr range-checks the indices and sums any duplicates).
    Coo coo;
    coo.nrows = nCells;
    coo.ncols = nCells;
    const std::size_t nnz = static_cast<std::size_t>(nCells) + 2 * nFaces;
    coo.row.reserve(nnz);
    coo.col.reserve(nnz);
    coo.val.reserve(nnz);

    for (int i = 0; i < nCells; ++i) {
        coo.row.push_back(i);
        coo.col.push_back(i);
        coo.val.push_back(diag[static_cast<std::size_t>(i)]);
    }

    for (std::size_t f = 0; f < nFaces; ++f) {
        const int l = lowerAddr[f];
        const int u = upperAddr[f];
        // A[l][u] = upper[f]
        coo.row.push_back(l);
        coo.col.push_back(u);
        coo.val.push_back(upper[f]);
        // A[u][l] = lower[f]  (== upper[f] when symmetric)
        coo.row.push_back(u);
        coo.col.push_back(l);
        coo.val.push_back(symmetric ? upper[f] : lower[f]);
    }

    return sell_from_coo(coo);
}

} // namespace spume
