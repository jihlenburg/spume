// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include "core/poisson.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace spume::gen {

Coo poisson7(index_t nx, index_t ny, index_t nz) {
    if (nx < 1 || ny < 1 || nz < 1) {
        throw std::invalid_argument("poisson7: grid dimensions must be >= 1");
    }
    const auto n64 = static_cast<offset_t>(nx) * ny * nz;
    if (n64 > 2147483647) {
        throw std::invalid_argument("poisson7: grid exceeds 32-bit index range");
    }
    const auto n = static_cast<index_t>(n64);

    Coo a;
    a.nrows = n;
    a.ncols = n;
    a.row.reserve(static_cast<std::size_t>(n64) * 7);
    a.col.reserve(static_cast<std::size_t>(n64) * 7);
    a.val.reserve(static_cast<std::size_t>(n64) * 7);

    auto idx = [nx, ny](index_t i, index_t j, index_t k) { return i + nx * (j + ny * k); };
    auto push = [&a](index_t r, index_t c, double v) {
        a.row.push_back(r);
        a.col.push_back(c);
        a.val.push_back(v);
    };

    for (index_t k = 0; k < nz; ++k) {
        for (index_t j = 0; j < ny; ++j) {
            for (index_t i = 0; i < nx; ++i) {
                const index_t r = idx(i, j, k);
                push(r, r, 6.0);
                if (i > 0) {
                    push(r, idx(i - 1, j, k), -1.0);
                }
                if (i < nx - 1) {
                    push(r, idx(i + 1, j, k), -1.0);
                }
                if (j > 0) {
                    push(r, idx(i, j - 1, k), -1.0);
                }
                if (j < ny - 1) {
                    push(r, idx(i, j + 1, k), -1.0);
                }
                if (k > 0) {
                    push(r, idx(i, j, k - 1), -1.0);
                }
                if (k < nz - 1) {
                    push(r, idx(i, j, k + 1), -1.0);
                }
            }
        }
    }
    return a;
}

PoissonBounds poisson7_bounds(index_t nx, index_t ny, index_t nz) {
    auto mu = [](index_t n, bool low) {
        const double arg =
            low ? std::numbers::pi / (2.0 * (n + 1)) : n * std::numbers::pi / (2.0 * (n + 1));
        const double s = std::sin(arg);
        return 4.0 * s * s;
    };
    return PoissonBounds{
        mu(nx, true) + mu(ny, true) + mu(nz, true),
        mu(nx, false) + mu(ny, false) + mu(nz, false),
    };
}

} // namespace spume::gen
