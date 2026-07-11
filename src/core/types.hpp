// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <cstdint>

namespace spume {

// Column/row indices are 32-bit by design: index bytes are matrix traffic,
// and bytes are the budget. Cases beyond 2^31-1 rows are out of scope for
// a single rank.
using index_t = std::int32_t;

// Offsets into value/index arrays can exceed 2^31 (padded nnz), so they are
// 64-bit.
using offset_t = std::int64_t;

// Runtime kernel dispatch. Every optimized path has a portable reference
// implementation; the default is always the reference path (AGENTS.md,
// architecture invariant 4).
enum class Dispatch {
    reference, // portable serial implementation
    openmp     // OpenMP-parallel implementation
};

// Reduction mode for dot products and norms.
//
// `deterministic` uses a fixed-order pairwise tree (see reduce.hpp) whose
// result is bitwise identical for any thread count and for both dispatch
// paths. It is a debug/validation mode; `standard` is the default.
enum class Reduction {
    standard,     // fastest available reduction; order may vary with threads
    deterministic // fixed-order pairwise tree; bitwise reproducible
};

} // namespace spume
