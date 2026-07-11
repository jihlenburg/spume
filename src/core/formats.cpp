// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include "core/formats.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace spume {

Csr coo_to_csr(const Coo& a) {
    const std::size_t nnz_in = a.val.size();
    if (a.row.size() != nnz_in || a.col.size() != nnz_in) {
        throw std::invalid_argument("coo_to_csr: row/col/val size mismatch");
    }
    for (std::size_t k = 0; k < nnz_in; ++k) {
        if (a.row[k] < 0 || a.row[k] >= a.nrows || a.col[k] < 0 || a.col[k] >= a.ncols) {
            throw std::invalid_argument("coo_to_csr: index out of range");
        }
    }

    // Sort entry ids by (row, col), then merge duplicates in one pass.
    std::vector<std::size_t> order(nnz_in);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t l, std::size_t r) {
        if (a.row[l] != a.row[r]) {
            return a.row[l] < a.row[r];
        }
        return a.col[l] < a.col[r];
    });

    Csr out;
    out.nrows = a.nrows;
    out.ncols = a.ncols;
    out.rowptr.assign(static_cast<std::size_t>(a.nrows) + 1, 0);
    out.col.reserve(nnz_in);
    out.val.reserve(nnz_in);

    for (std::size_t k = 0; k < nnz_in; ++k) {
        const std::size_t e = order[k];
        if (!out.col.empty() && k > 0 && a.row[order[k - 1]] == a.row[e] &&
            a.col[order[k - 1]] == a.col[e]) {
            out.val.back() += a.val[e]; // duplicate: accumulate
        } else {
            out.rowptr[static_cast<std::size_t>(a.row[e]) + 1] += 1;
            out.col.push_back(a.col[e]);
            out.val.push_back(a.val[e]);
        }
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(a.nrows); ++i) {
        out.rowptr[i + 1] += out.rowptr[i];
    }
    return out;
}

} // namespace spume
