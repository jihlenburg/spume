// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <vector>

#include "core/types.hpp"

namespace spume {

// Coordinate format: assembly-friendly, unsorted, duplicates allowed
// (summed on conversion).
struct Coo {
    index_t nrows = 0;
    index_t ncols = 0;
    std::vector<index_t> row;
    std::vector<index_t> col;
    std::vector<double> val;
};

// Compressed sparse row: the FP64 ground-truth format every other format is
// built from. Rows sorted by column, no duplicates.
struct Csr {
    index_t nrows = 0;
    index_t ncols = 0;
    std::vector<offset_t> rowptr; // size nrows + 1
    std::vector<index_t> col;
    std::vector<double> val;

    offset_t nnz() const { return rowptr.empty() ? 0 : rowptr.back(); }
};

// Sort by (row, col) and sum duplicates. Throws std::invalid_argument on
// out-of-range indices.
Csr coo_to_csr(const Coo& a);

} // namespace spume
