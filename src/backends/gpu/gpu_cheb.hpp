// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <span>

#include "core/equilibrate.hpp"
#include "core/precond.hpp"
#include "core/types.hpp"

// GPU FP32 Chebyshev smoother (ADR-0017, ADR-0002). The equilibrated operator
// S A S (unit diagonal) and the Chebyshev recurrence run entirely in FP32 on
// the device -- the mixed-precision preconditioner interior -- while the input
// residual and output correction stay FP64 (the precision firewall). The whole
// apply is GPU-resident: one host->device residual copy in, the Saad recurrence
// as a sequence of kernels on the default stream, one correction copy out.
//
// FP32 is reorder-sensitive, so the result matches the CPU ChebyshevPrecond
// only within the reorder-tolerance class (ADR-0017 measured max-rel 2.7e-7 for
// this apply), not bitwise -- verified against the reference, never assumed.

namespace spume::gpu {

class ChebyshevDeviceFP32 {
public:
    ChebyshevDeviceFP32(const EqOperator<float>& op, ChebyshevOptions opt);
    ~ChebyshevDeviceFP32();
    ChebyshevDeviceFP32(const ChebyshevDeviceFP32&) = delete;
    ChebyshevDeviceFP32& operator=(const ChebyshevDeviceFP32&) = delete;

    // z ~= M^{-1} r (one smoother apply). r, z are host-visible FP64 views sized
    // to nrows(); staged through resident buffers. Blocking.
    void apply(std::span<const double> r, std::span<double> z) const;

    // Mean kernel time over `reps` full applies on the currently-resident
    // residual (call apply() once first). Kernel sequence only, no host staging.
    double kernel_ms(int reps) const;

    index_t nrows() const { return nrows_; }

private:
    // Launch the recurrence kernels (scale-in -> Saad steps -> scale-out) on the
    // resident buffers; no host copies, no sync. Defined in the .hip.cpp.
    void launch_apply(const double* r_dev, double* z_dev) const;

    index_t nrows_ = 0;
    offset_t stored_ = 0;
    double lmax_ = 0.0;
    int steps_ = 0;
    double eta_ = 0.0;

    // Resident operator (unified memory).
    offset_t* d_chunk_ptr_ = nullptr;
    float* d_val_ = nullptr;
    index_t* d_colidx_ = nullptr;
    double* d_scale_ = nullptr;
    // Resident workspace.
    mutable float* d_wr_ = nullptr;
    mutable float* d_wx_ = nullptr;
    mutable float* d_wd_ = nullptr;
    mutable double* d_r_ = nullptr;
    mutable double* d_z_ = nullptr;
};

} // namespace spume::gpu
