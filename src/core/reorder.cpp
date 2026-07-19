// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include "core/reorder.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <queue>

namespace spume {

namespace {

// Off-diagonal degree of each row (number of stored neighbours, excluding self).
std::vector<index_t> degrees(const Csr& a) {
    std::vector<index_t> deg(static_cast<std::size_t>(a.nrows), 0);
    for (index_t i = 0; i < a.nrows; ++i) {
        index_t d = 0;
        for (offset_t k = a.rowptr[static_cast<std::size_t>(i)];
             k < a.rowptr[static_cast<std::size_t>(i) + 1]; ++k) {
            if (a.col[static_cast<std::size_t>(k)] != i) {
                ++d;
            }
        }
        deg[static_cast<std::size_t>(i)] = d;
    }
    return deg;
}

} // namespace

std::vector<index_t> rcm_order(const Csr& a) {
    const index_t n = a.nrows;
    const std::vector<index_t> deg = degrees(a);
    std::vector<char> seen(static_cast<std::size_t>(n), 0);
    std::vector<index_t> order;
    order.reserve(static_cast<std::size_t>(n));

    // Cuthill-McKee over each connected component; components are discovered by
    // scanning for the lowest-degree unseen node as the next start.
    index_t scan = 0;
    while (order.size() != static_cast<std::size_t>(n)) {
        // lowest-degree unseen node (pseudo-peripheral-ish start)
        index_t start = -1;
        for (index_t i = scan; i < n; ++i) {
            if (!seen[static_cast<std::size_t>(i)]) {
                if (start == -1 ||
                    deg[static_cast<std::size_t>(i)] < deg[static_cast<std::size_t>(start)]) {
                    start = i;
                }
            }
        }
        while (scan < n && seen[static_cast<std::size_t>(scan)]) {
            ++scan; // advance the low-water mark past already-seen nodes
        }

        std::queue<index_t> q;
        q.push(start);
        seen[static_cast<std::size_t>(start)] = 1;
        while (!q.empty()) {
            const index_t u = q.front();
            q.pop();
            order.push_back(u);

            // gather unseen neighbours of u, then enqueue them in ascending degree
            std::vector<index_t> nbr;
            for (offset_t k = a.rowptr[static_cast<std::size_t>(u)];
                 k < a.rowptr[static_cast<std::size_t>(u) + 1]; ++k) {
                const index_t j = a.col[static_cast<std::size_t>(k)];
                if (j != u && !seen[static_cast<std::size_t>(j)]) {
                    seen[static_cast<std::size_t>(j)] = 1; // mark on enqueue (no dups)
                    nbr.push_back(j);
                }
            }
            std::sort(nbr.begin(), nbr.end(), [&](index_t x, index_t y) {
                return deg[static_cast<std::size_t>(x)] < deg[static_cast<std::size_t>(y)];
            });
            for (const index_t j : nbr) {
                q.push(j);
            }
        }
    }

    std::reverse(order.begin(), order.end()); // Cuthill-McKee -> Reverse CM
    return order;
}

Csr reorder(const Csr& a, const std::vector<index_t>& new_to_old) {
    const index_t n = a.nrows;
    std::vector<index_t> old_to_new(static_cast<std::size_t>(n));
    for (index_t k = 0; k < n; ++k) {
        old_to_new[static_cast<std::size_t>(new_to_old[static_cast<std::size_t>(k)])] = k;
    }

    Coo c;
    c.nrows = n;
    c.ncols = a.ncols;
    c.row.reserve(static_cast<std::size_t>(a.nnz()));
    c.col.reserve(static_cast<std::size_t>(a.nnz()));
    c.val.reserve(static_cast<std::size_t>(a.nnz()));

    for (index_t k = 0; k < n; ++k) {
        const index_t i = new_to_old[static_cast<std::size_t>(k)]; // old row
        for (offset_t p = a.rowptr[static_cast<std::size_t>(i)];
             p < a.rowptr[static_cast<std::size_t>(i) + 1]; ++p) {
            c.row.push_back(k);
            c.col.push_back(old_to_new[static_cast<std::size_t>(a.col[static_cast<std::size_t>(p)])]);
            c.val.push_back(a.val[static_cast<std::size_t>(p)]);
        }
    }
    return coo_to_csr(c);
}

index_t bandwidth(const Csr& a) {
    index_t bw = 0;
    for (index_t i = 0; i < a.nrows; ++i) {
        for (offset_t k = a.rowptr[static_cast<std::size_t>(i)];
             k < a.rowptr[static_cast<std::size_t>(i) + 1]; ++k) {
            const index_t d = std::abs(i - a.col[static_cast<std::size_t>(k)]);
            bw = std::max(bw, d);
        }
    }
    return bw;
}

} // namespace spume
