// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <span>

#include "core/sell.hpp"
#include "core/types.hpp"

// GPU backend (ADR-0017): HIP kernels for gfx1151, unified-memory resident.
// This header is HIP-free so the portable core (compiled with the host C++
// compiler) can include it; the implementation lives in gpu_spmv.hip.cpp and is
// compiled with hipcc. The portable reference stays the default (ADR-0004,
// invariant 4) -- this backend is opt-in, built only with SPUME_ENABLE_HIP.

namespace spume::gpu {

// True if a usable HIP device is present. Safe to call with no GPU (returns
// false rather than aborting), so callers can fall back to the CPU path.
bool available();

// A SELL-C-8 FP64 operator resident in unified (hipMallocManaged / GTT) memory
// on the default HIP device -- the M3 residency model (ADR-0017): upload the
// operator once, launch many SpMVs, never copy the matrix again. hipMalloc is
// deliberately avoided (segfaults above the small carve-out on this APU).
//
// The kernel reproduces the CPU reference's fixed per-row accumulation order, so
// its result matches the reference within the reorder-tolerance equivalence
// class (ADR-0002) -- verified against spume::spmv, never assumed.
class SellDeviceFP64 {
public:
    explicit SellDeviceFP64(const Sell<double>& host);
    ~SellDeviceFP64();
    SellDeviceFP64(const SellDeviceFP64&) = delete;
    SellDeviceFP64& operator=(const SellDeviceFP64&) = delete;

    // y = A x on the GPU (blocking). x and y are host-visible views sized to
    // ncols()/nrows(); values are staged through the resident scratch buffers.
    void spmv(std::span<const double> x, std::span<double> y) const;

    // Device-to-device operator applications for a resident solve (no host
    // staging, no sync -- the outer loop syncs at its reductions):
    //   spmv_device:     y = A x
    //   residual_device: r = b - A x  (fused, one pass -- opt #1)
    // All pointers are into resident (managed) memory sized to nrows()/ncols().
    void spmv_device(const double* x_dev, double* y_dev) const;
    void residual_device(const double* b_dev, const double* x_dev, double* r_dev) const;

    // Mean GPU kernel time over `reps` launches on the currently-resident x
    // (call spmv() at least once first to populate it). Times the kernel only --
    // no host staging -- via HIP events, for a clean achieved-bandwidth figure.
    // Returns milliseconds per launch.
    double kernel_ms(int reps) const;

    index_t nrows() const { return nrows_; }
    index_t ncols() const { return ncols_; }

private:
    index_t nrows_ = 0;
    index_t ncols_ = 0;
    offset_t stored_ = 0;
    // Managed-memory buffers: plain pointers into unified memory, valid on both
    // host and device (no HIP types leak into this header).
    offset_t* d_chunk_ptr_ = nullptr;
    double* d_val_ = nullptr;
    index_t* d_colidx_ = nullptr;
    double* d_x_ = nullptr;
    double* d_y_ = nullptr;
};

} // namespace spume::gpu
