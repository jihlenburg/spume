// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <vector>

#include "core/formats.hpp"
#include "core/types.hpp"

// Locality reordering — the access-pattern-first front end (ADR-0002/0004).
//
// Measured on the target hardware (see memory / the fused-SpMV work): an
// irregular sparse operator is LATENCY-bound, not bandwidth-bound — a scattered
// gather moves *fewer* DRAM bytes yet runs ~2.3x slower, because it thrashes the
// dTLB (~29x more page-table walks), defeats the hardware prefetcher (cores
// stall), and misses the DRAM row buffer. The fix is to renumber the unknowns so
// that coupled unknowns get nearby indices: the prefetcher predicts the stream,
// pages are reused, row-buffer locality returns.
//
// Because the mesh is static per case, this reordering is computed ONCE at setup
// and reused every timestep (amortised to nearly free). In the FP32-GAMG it is
// applied per level — coarse Galerkin operators are the most scattered.

namespace spume {

// Reverse Cuthill-McKee ordering of the matrix graph (symmetric pattern
// assumed; the strict lower/upper structure is taken from the stored entries).
// Returns `new_to_old`: new_to_old[k] is the ORIGINAL index that becomes row k
// of the reordered operator. Bandwidth-reducing, deterministic (min-degree
// start per connected component, degree-ordered BFS, reversed).
std::vector<index_t> rcm_order(const Csr& a);

// Reorder a matrix by a permutation: B[k][l] = A[p[k]][p[l]] (i.e. B = P^T A P
// with P = new_to_old = `p`). Symmetric relabelling — preserves the spectrum.
Csr reorder(const Csr& a, const std::vector<index_t>& new_to_old);

// Half-bandwidth in the given ordering: max over stored entries of |row - col|.
// A diagnostic used by the tests and the setup phase to check a reordering
// actually improved locality.
index_t bandwidth(const Csr& a);

} // namespace spume
