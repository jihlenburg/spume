// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <cassert>
#include <cstdint>
#include <span>

#include "core/types.hpp"

namespace spume {

// Element-wise vector operations. Each output element depends only on the
// same-index inputs, so results are bitwise identical for any thread count
// and both dispatch paths — no deterministic mode is needed here.

// y += a * x
template<typename T>
void axpy(T a, std::span<const T> x, std::span<T> y, Dispatch dispatch = Dispatch::reference) {
    assert(x.size() == y.size());
    const auto n = static_cast<std::int64_t>(x.size());
    if (dispatch == Dispatch::openmp) {
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < n; ++i) {
            y[static_cast<std::size_t>(i)] += a * x[static_cast<std::size_t>(i)];
        }
    } else {
        for (std::int64_t i = 0; i < n; ++i) {
            y[static_cast<std::size_t>(i)] += a * x[static_cast<std::size_t>(i)];
        }
    }
}

// y = x + a * y
template<typename T>
void aypx(T a, std::span<const T> x, std::span<T> y, Dispatch dispatch = Dispatch::reference) {
    assert(x.size() == y.size());
    const auto n = static_cast<std::int64_t>(x.size());
    if (dispatch == Dispatch::openmp) {
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < n; ++i) {
            y[static_cast<std::size_t>(i)] =
                x[static_cast<std::size_t>(i)] + a * y[static_cast<std::size_t>(i)];
        }
    } else {
        for (std::int64_t i = 0; i < n; ++i) {
            y[static_cast<std::size_t>(i)] =
                x[static_cast<std::size_t>(i)] + a * y[static_cast<std::size_t>(i)];
        }
    }
}

// y = a * x + b * y
template<typename T>
void axpby(T a, std::span<const T> x, T b, std::span<T> y,
           Dispatch dispatch = Dispatch::reference) {
    assert(x.size() == y.size());
    const auto n = static_cast<std::int64_t>(x.size());
    if (dispatch == Dispatch::openmp) {
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < n; ++i) {
            y[static_cast<std::size_t>(i)] =
                a * x[static_cast<std::size_t>(i)] + b * y[static_cast<std::size_t>(i)];
        }
    } else {
        for (std::int64_t i = 0; i < n; ++i) {
            y[static_cast<std::size_t>(i)] =
                a * x[static_cast<std::size_t>(i)] + b * y[static_cast<std::size_t>(i)];
        }
    }
}

} // namespace spume
