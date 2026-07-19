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

Coo poisson7_graded(index_t nx, index_t ny, index_t nz, double cz_min, double cz_max) {
    if (nx < 1 || ny < 1 || nz < 1) {
        throw std::invalid_argument("poisson7_graded: grid dimensions must be >= 1");
    }
    if (!(cz_min > 0.0) || !(cz_max > 0.0)) {
        throw std::invalid_argument("poisson7_graded: conductivities must be > 0");
    }
    const auto n64 = static_cast<offset_t>(nx) * ny * nz;
    if (n64 > 2147483647) {
        throw std::invalid_argument("poisson7_graded: grid exceeds 32-bit index range");
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
    // Conductivity of z-face f (between cells f-1 and f), f in [0, nz]: linear
    // ramp cz_min .. cz_max. Both boundary faces (f=0, f=nz) get a ramp value.
    auto czface = [nz, cz_min, cz_max](index_t f) {
        const double t = (nz > 0) ? static_cast<double>(f) / static_cast<double>(nz) : 0.0;
        return cz_min + (cz_max - cz_min) * t;
    };

    for (index_t k = 0; k < nz; ++k) {
        const double cz_lo = czface(k);     // face below cell k
        const double cz_hi = czface(k + 1); // face above cell k
        for (index_t j = 0; j < ny; ++j) {
            for (index_t i = 0; i < nx; ++i) {
                const index_t r = idx(i, j, k);
                // diagonal = sum of ALL six incident face conductivities
                // (Dirichlet faces included): 4 unit x/y faces + two z faces.
                push(r, r, 4.0 + cz_lo + cz_hi);
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
                    push(r, idx(i, j, k - 1), -cz_lo);
                }
                if (k < nz - 1) {
                    push(r, idx(i, j, k + 1), -cz_hi);
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
