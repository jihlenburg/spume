// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include "core/amg.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace spume {

Aggregation aggregate(const Csr& a, double theta) {
    const index_t n = a.nrows;
    std::vector<index_t> agg(static_cast<std::size_t>(n), -1);
    index_t nc = 0;

    for (index_t i = 0; i < n; ++i) {
        if (agg[static_cast<std::size_t>(i)] != -1) {
            continue;
        }
        const offset_t lo = a.rowptr[static_cast<std::size_t>(i)];
        const offset_t hi = a.rowptr[static_cast<std::size_t>(i) + 1];

        // strongest off-diagonal magnitude in this row sets the strength scale
        double maxod = 0.0;
        for (offset_t k = lo; k < hi; ++k) {
            if (a.col[static_cast<std::size_t>(k)] != i) {
                maxod = std::max(maxod, std::abs(a.val[static_cast<std::size_t>(k)]));
            }
        }

        // seed a new aggregate with i and its still-free strong neighbours
        agg[static_cast<std::size_t>(i)] = nc;
        for (offset_t k = lo; k < hi; ++k) {
            const index_t j = a.col[static_cast<std::size_t>(k)];
            if (j == i || agg[static_cast<std::size_t>(j)] != -1) {
                continue;
            }
            if (std::abs(a.val[static_cast<std::size_t>(k)]) >= theta * maxod) {
                agg[static_cast<std::size_t>(j)] = nc;
            }
        }
        ++nc;
    }

    return Aggregation{std::move(agg), nc};
}

Csr galerkin(const Csr& a, const Aggregation& agg) {
    Coo c;
    c.nrows = agg.ncoarse;
    c.ncols = agg.ncoarse;
    c.row.reserve(static_cast<std::size_t>(a.nnz()));
    c.col.reserve(static_cast<std::size_t>(a.nnz()));
    c.val.reserve(static_cast<std::size_t>(a.nnz()));

    for (index_t i = 0; i < a.nrows; ++i) {
        const index_t ci = agg.agg[static_cast<std::size_t>(i)];
        for (offset_t k = a.rowptr[static_cast<std::size_t>(i)];
             k < a.rowptr[static_cast<std::size_t>(i) + 1]; ++k) {
            c.row.push_back(ci);
            c.col.push_back(agg.agg[static_cast<std::size_t>(a.col[static_cast<std::size_t>(k)])]);
            c.val.push_back(a.val[static_cast<std::size_t>(k)]);
        }
    }

    // coo_to_csr sorts by (row,col) and sums the duplicate (I,J) contributions.
    return coo_to_csr(c);
}

std::vector<Aggregation> aggregate_hierarchy(const Csr& fine, index_t coarse_size,
                                             int max_levels, double theta) {
    std::vector<Aggregation> aggs;
    Csr cur = fine;
    while (static_cast<int>(aggs.size()) + 1 < max_levels &&
           cur.nrows > coarse_size) {
        Aggregation agg = aggregate(cur, theta);
        if (agg.ncoarse >= cur.nrows) {
            break; // aggregation made no progress; stop coarsening
        }
        Csr coarse = galerkin(cur, agg);
        aggs.push_back(std::move(agg));
        cur = std::move(coarse);
    }
    return aggs;
}

} // namespace spume
