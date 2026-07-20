// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// HIP implementation of the GPU FP32 Chebyshev smoother for gfx1151 (ADR-0017,
// ADR-0002). Mirrors the CPU spume::ChebyshevPrecond<float> recurrence
// (core/precond.hpp) kernel-for-kernel so the two agree in-class.

#include "backends/gpu/gpu_cheb.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace spume::gpu {

namespace {

#define SPUME_HIP_CHECK(call)                                                  \
    do {                                                                       \
        hipError_t err_ = (call);                                              \
        if (err_ != hipSuccess) {                                             \
            std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #call,          \
                         __FILE__, __LINE__, hipGetErrorString(err_));         \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

constexpr int kBlock = 256;

int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

// wr = float(scale * r): equilibrate the FP64 residual and narrow once (the
// operand demoted is S r, policy-clean).
__global__ void scale_in_k(const double* __restrict__ r, const double* __restrict__ scale,
                           float* __restrict__ wr, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        wr[i] = static_cast<float>(scale[i] * r[i]);
    }
}

// d_0 = wr / theta, x_0 = 0.
__global__ void init_k(const float* __restrict__ wr, float inv_theta, float* __restrict__ wd,
                       float* __restrict__ wx, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        wd[i] = wr[i] * inv_theta;
        wx[i] = 0.0F;
    }
}

// y += a * x  (FP32).
__global__ void axpy_k(float a, const float* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += a * x[i];
    }
}

// y = a * x + b * y  (FP32).
__global__ void axpby_k(float a, const float* __restrict__ x, float b, float* __restrict__ y,
                        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + b * y[i];
    }
}

// wr -= A d  (fused FP32 SELL SpMV subtract; A d is never materialised), matching
// the CPU spmv_axpy(op.a, -1, wd, wr). Fixed j-order per row.
__global__ void spmv_sub_k(const std::int64_t* __restrict__ chunk_ptr,
                           const float* __restrict__ val, const std::int32_t* __restrict__ colidx,
                           const float* __restrict__ d, float* __restrict__ wr, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const int c = row >> 3;
    const int lane = row & 7;
    const std::int64_t base = chunk_ptr[c];
    const std::int64_t w = (chunk_ptr[c + 1] - base) >> 3;
    float acc = 0.0F;
    std::int64_t off = base + lane;
    for (std::int64_t j = 0; j < w; ++j) {
        acc += val[off] * d[colidx[off]];
        off += 8;
    }
    wr[row] -= acc;
}

// z = scale * float(wx), widened back to FP64.
__global__ void scale_out_k(const float* __restrict__ wx, const double* __restrict__ scale,
                            double* __restrict__ z, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = scale[i] * static_cast<double>(wx[i]);
    }
}

} // namespace

ChebyshevDeviceFP32::ChebyshevDeviceFP32(const EqOperator<float>& op, ChebyshevOptions opt)
    : nrows_(op.a.nrows), stored_(op.a.stored()), lmax_(op.lambda_max_bound),
      steps_(opt.steps), eta_(opt.eta) {
    const std::size_t ncp = op.a.chunk_ptr.size();
    const auto stored = static_cast<std::size_t>(stored_);
    const auto n = static_cast<std::size_t>(nrows_);
    SPUME_HIP_CHECK(hipMallocManaged(&d_chunk_ptr_, ncp * sizeof(offset_t)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_val_, stored * sizeof(float)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_colidx_, stored * sizeof(index_t)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_scale_, n * sizeof(double)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_wr_, n * sizeof(float)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_wx_, n * sizeof(float)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_wd_, n * sizeof(float)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_r_, n * sizeof(double)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_z_, n * sizeof(double)));

    std::copy(op.a.chunk_ptr.begin(), op.a.chunk_ptr.end(), d_chunk_ptr_);
    std::copy(op.a.val.begin(), op.a.val.end(), d_val_);
    std::copy(op.a.colidx.begin(), op.a.colidx.end(), d_colidx_);
    std::copy(op.scale.begin(), op.scale.end(), d_scale_);

    int dev = 0;
    SPUME_HIP_CHECK(hipGetDevice(&dev));
    static_cast<void>(hipMemPrefetchAsync(d_chunk_ptr_, ncp * sizeof(offset_t), dev));
    static_cast<void>(hipMemPrefetchAsync(d_val_, stored * sizeof(float), dev));
    static_cast<void>(hipMemPrefetchAsync(d_colidx_, stored * sizeof(index_t), dev));
    static_cast<void>(hipMemPrefetchAsync(d_scale_, n * sizeof(double), dev));
    SPUME_HIP_CHECK(hipDeviceSynchronize());
}

ChebyshevDeviceFP32::~ChebyshevDeviceFP32() {
    static_cast<void>(hipFree(d_chunk_ptr_));
    static_cast<void>(hipFree(d_val_));
    static_cast<void>(hipFree(d_colidx_));
    static_cast<void>(hipFree(d_scale_));
    static_cast<void>(hipFree(d_wr_));
    static_cast<void>(hipFree(d_wx_));
    static_cast<void>(hipFree(d_wd_));
    static_cast<void>(hipFree(d_r_));
    static_cast<void>(hipFree(d_z_));
}

void ChebyshevDeviceFP32::launch_apply(const double* r_dev, double* z_dev) const {
    const int n = nrows_;
    const int g = grid_for(n);

    // Spectrum bounds and the Saad recurrence scalars (host, FP64), matching
    // core/precond.hpp exactly.
    const double lmin = lmax_ / eta_;
    const double theta = 0.5 * (lmax_ + lmin);
    const double delta = 0.5 * (lmax_ - lmin);
    const double sigma1 = theta / delta;

    scale_in_k<<<g, kBlock>>>(r_dev, d_scale_, d_wr_, n);
    init_k<<<g, kBlock>>>(d_wr_, static_cast<float>(1.0 / theta), d_wd_, d_wx_, n);

    double rho = 1.0 / sigma1;
    for (int k = 0; k < steps_; ++k) {
        axpy_k<<<g, kBlock>>>(1.0F, d_wd_, d_wx_, n); // wx += wd
        if (k + 1 == steps_) {
            break; // last correction applied; the r/d updates would be dead
        }
        spmv_sub_k<<<g, kBlock>>>(d_chunk_ptr_, d_val_, d_colidx_, d_wd_, d_wr_, n); // wr -= A wd
        const double rho_new = 1.0 / (2.0 * sigma1 - rho);
        axpby_k<<<g, kBlock>>>(static_cast<float>(2.0 * rho_new / delta), d_wr_,
                               static_cast<float>(rho_new * rho), d_wd_, n); // wd = c1 wr + c2 wd
        rho = rho_new;
    }

    scale_out_k<<<g, kBlock>>>(d_wx_, d_scale_, z_dev, n);
}

void ChebyshevDeviceFP32::apply(std::span<const double> r, std::span<double> z) const {
    std::copy(r.begin(), r.end(), d_r_);
    launch_apply(d_r_, d_z_);
    SPUME_HIP_CHECK(hipGetLastError());
    SPUME_HIP_CHECK(hipDeviceSynchronize());
    std::copy(d_z_, d_z_ + nrows_, z.begin());
}

double ChebyshevDeviceFP32::kernel_ms(int reps) const {
    for (int i = 0; i < 3; ++i) { // warm-up
        launch_apply(d_r_, d_z_);
    }
    SPUME_HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    SPUME_HIP_CHECK(hipEventCreate(&start));
    SPUME_HIP_CHECK(hipEventCreate(&stop));
    SPUME_HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < reps; ++i) {
        launch_apply(d_r_, d_z_);
    }
    SPUME_HIP_CHECK(hipEventRecord(stop, 0));
    SPUME_HIP_CHECK(hipEventSynchronize(stop));
    float ms_total = 0.0F;
    SPUME_HIP_CHECK(hipEventElapsedTime(&ms_total, start, stop));
    SPUME_HIP_CHECK(hipEventDestroy(start));
    SPUME_HIP_CHECK(hipEventDestroy(stop));
    return static_cast<double>(ms_total) / reps;
}

} // namespace spume::gpu
