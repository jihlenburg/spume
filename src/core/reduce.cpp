// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include "core/reduce.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace spume {

namespace {

// Serial product-sum over [0, n); the leaf of the deterministic tree.
double block_dot(const double* x, const double* y, std::int64_t n) {
    double s = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}

// Fixed-shape pairwise tree over the block partials. The split point depends
// only on n, never on the thread count.
double pairwise_sum(const double* v, std::int64_t n) {
    if (n <= 4) {
        double s = 0.0;
        for (std::int64_t i = 0; i < n; ++i) {
            s += v[i];
        }
        return s;
    }
    const std::int64_t h = n / 2;
    return pairwise_sum(v, h) + pairwise_sum(v + h, n - h);
}

double det_dot(std::span<const double> x, std::span<const double> y, Dispatch dispatch) {
    const auto n = static_cast<std::int64_t>(x.size());
    if (n == 0) {
        return 0.0;
    }
    const std::int64_t nb = (n + kDetBlock - 1) / kDetBlock;
    std::vector<double> partials(static_cast<std::size_t>(nb));
    if (dispatch == Dispatch::openmp) {
#pragma omp parallel for schedule(static)
        for (std::int64_t b = 0; b < nb; ++b) {
            const std::int64_t lo = b * kDetBlock;
            partials[static_cast<std::size_t>(b)] =
                block_dot(x.data() + lo, y.data() + lo, std::min(kDetBlock, n - lo));
        }
    } else {
        for (std::int64_t b = 0; b < nb; ++b) {
            const std::int64_t lo = b * kDetBlock;
            partials[static_cast<std::size_t>(b)] =
                block_dot(x.data() + lo, y.data() + lo, std::min(kDetBlock, n - lo));
        }
    }
    return pairwise_sum(partials.data(), nb);
}

} // namespace

double dot(std::span<const double> x, std::span<const double> y, Dispatch dispatch,
           Reduction mode) {
    assert(x.size() == y.size());
    if (mode == Reduction::deterministic) {
        return det_dot(x, y, dispatch);
    }
    const auto n = static_cast<std::int64_t>(x.size());
    double s = 0.0;
    if (dispatch == Dispatch::openmp) {
#pragma omp parallel for schedule(static) reduction(+ : s)
        for (std::int64_t i = 0; i < n; ++i) {
            s += x[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)];
        }
    } else {
        s = block_dot(x.data(), y.data(), n);
    }
    return s;
}

double nrm2(std::span<const double> x, Dispatch dispatch, Reduction mode) {
    return std::sqrt(dot(x, x, dispatch, mode));
}

} // namespace spume
