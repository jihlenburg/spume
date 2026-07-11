// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <span>

#include "core/types.hpp"

namespace spume {

// Dot products and norms with runtime-dispatched execution and a
// deterministic-reduction debug mode.
//
// Reduction::standard
//   reference: plain left-to-right summation.
//   openmp:    `omp parallel for reduction(+:...)` — summation order (and
//              therefore the exact rounding) varies with the thread count.
//
// Reduction::deterministic
//   The vector is cut into fixed blocks of kDetBlock elements. Each block is
//   summed left-to-right; the block partials are then combined with a serial
//   pairwise tree whose shape depends only on the vector length. Threads may
//   compute different blocks, but every block partial and the combination
//   order are independent of the thread count, so the result is bitwise
//   identical across thread counts AND across both dispatch paths.
//
// Deterministic mode is the debug/validation mode required by the numerics
// policy; standard mode is the default everywhere.

inline constexpr std::int64_t kDetBlock = 4096;

double dot(std::span<const double> x, std::span<const double> y,
           Dispatch dispatch = Dispatch::reference, Reduction mode = Reduction::standard);

double nrm2(std::span<const double> x, Dispatch dispatch = Dispatch::reference,
            Reduction mode = Reduction::standard);

} // namespace spume
