// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include "core/types.hpp"

// GPU FP64 dot-product reduction (ADR-0017) -- the reduction primitive the
// remaining M3 work needs: the K-cycle's Krylov coarse acceleration and the
// whole-solve GPU-resident FCG both take dot products of resident vectors and
// feed the scalar back into the next kernel launch. A block reduction produces
// per-block partials; the (small, bounded) partial array is summed on the host,
// so the result is available as a plain double. The parallel reduction order
// differs from the serial reference, so the result is in-class (not bitwise) --
// acceptable in the FP32-firewalled preconditioner and the flexible outer FCG.
// This header is HIP-free.

namespace spume::gpu {

class DotDeviceFP64 {
public:
    DotDeviceFP64();
    ~DotDeviceFP64();
    DotDeviceFP64(const DotDeviceFP64&) = delete;
    DotDeviceFP64& operator=(const DotDeviceFP64&) = delete;

    // sum_i x[i] * y[i], x and y resident (device pointers), n entries. Blocks
    // (the reduction ends with a host sum of the per-block partials).
    double dot(const double* x_dev, const double* y_dev, index_t n) const;

private:
    double* d_partial_ = nullptr; // per-block partials, resident (bounded size)
};

} // namespace spume::gpu
