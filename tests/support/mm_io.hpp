// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "core/formats.hpp"

namespace spume::testing {

// Minimal MatrixMarket reader for test fixtures: coordinate, real,
// general or symmetric (symmetric entries are mirrored). 1-based indices.
inline Coo read_matrix_market(std::istream& in) {
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("mm_io: empty stream");
    }
    const bool symmetric = line.find("symmetric") != std::string::npos;
    if (line.rfind("%%MatrixMarket", 0) != 0 || line.find("coordinate") == std::string::npos ||
        line.find("real") == std::string::npos) {
        throw std::runtime_error("mm_io: unsupported header: " + line);
    }

    do {
        if (!std::getline(in, line)) {
            throw std::runtime_error("mm_io: missing size line");
        }
    } while (!line.empty() && line[0] == '%');

    std::istringstream sz(line);
    long nrows = 0, ncols = 0, nnz = 0;
    if (!(sz >> nrows >> ncols >> nnz) || nrows <= 0 || ncols <= 0 || nnz < 0) {
        throw std::runtime_error("mm_io: bad size line: " + line);
    }

    Coo a;
    a.nrows = static_cast<index_t>(nrows);
    a.ncols = static_cast<index_t>(ncols);
    for (long k = 0; k < nnz; ++k) {
        long r = 0, c = 0;
        double v = 0.0;
        if (!(in >> r >> c >> v)) {
            throw std::runtime_error("mm_io: truncated entry list");
        }
        a.row.push_back(static_cast<index_t>(r - 1));
        a.col.push_back(static_cast<index_t>(c - 1));
        a.val.push_back(v);
        if (symmetric && r != c) {
            a.row.push_back(static_cast<index_t>(c - 1));
            a.col.push_back(static_cast<index_t>(r - 1));
            a.val.push_back(v);
        }
    }
    return a;
}

inline Coo read_matrix_market_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("mm_io: cannot open " + path);
    }
    return read_matrix_market(f);
}

} // namespace spume::testing
