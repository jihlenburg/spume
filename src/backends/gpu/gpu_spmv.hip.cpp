// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// HIP implementation of the GPU SELL-C-8 FP64 SpMV for gfx1151 (Radeon 8060S,
// RDNA3.5), unified-memory resident (ADR-0017). Compiled with hipcc.

#include "backends/gpu/gpu_spmv.hpp"

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

// One thread per output row. Each thread owns a private accumulator and loops
// the row's stored columns in fixed j-order -- reproducing the reference
// spmv_chunk accumulation order, so the result is in the reorder-equivalence
// class. val[base + j*8 + lane] is 8-wide coalesced across a chunk; x[colidx]
// is the scattered gather; writes over consecutive rows are coalesced.
__global__ void sell_spmv_kernel(const std::int64_t* __restrict__ chunk_ptr,
                                 const double* __restrict__ val,
                                 const std::int32_t* __restrict__ colidx,
                                 const double* __restrict__ x,
                                 double* __restrict__ y, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const int c = row >> 3;   // chunk = row / 8
    const int lane = row & 7; // row within chunk
    const std::int64_t base = chunk_ptr[c];
    const std::int64_t w = (chunk_ptr[c + 1] - base) >> 3; // width / 8
    double acc = 0.0;
    std::int64_t off = base + lane;
    for (std::int64_t j = 0; j < w; ++j) {
        acc += val[off] * x[colidx[off]];
        off += 8;
    }
    y[row] = acc;
}

constexpr int kBlock = 256;

} // namespace

bool available() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) {
        return false;
    }
    return count > 0;
}

SellDeviceFP64::SellDeviceFP64(const Sell<double>& host)
    : nrows_(host.nrows), ncols_(host.ncols), stored_(host.stored()) {
    const std::size_t ncp = host.chunk_ptr.size();
    const auto stored = static_cast<std::size_t>(stored_);
    SPUME_HIP_CHECK(hipMallocManaged(&d_chunk_ptr_, ncp * sizeof(offset_t)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_val_, stored * sizeof(double)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_colidx_, stored * sizeof(index_t)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_x_, static_cast<std::size_t>(ncols_) * sizeof(double)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_y_, static_cast<std::size_t>(nrows_) * sizeof(double)));

    std::copy(host.chunk_ptr.begin(), host.chunk_ptr.end(), d_chunk_ptr_);
    std::copy(host.val.begin(), host.val.end(), d_val_);
    std::copy(host.colidx.begin(), host.colidx.end(), d_colidx_);

    // Prefetch the resident operator to the device before any timed launch.
    int dev = 0;
    SPUME_HIP_CHECK(hipGetDevice(&dev));
    // Best-effort prefetch (not all runtimes honour it); errors are non-fatal.
    static_cast<void>(hipMemPrefetchAsync(d_chunk_ptr_, ncp * sizeof(offset_t), dev));
    static_cast<void>(hipMemPrefetchAsync(d_val_, stored * sizeof(double), dev));
    static_cast<void>(hipMemPrefetchAsync(d_colidx_, stored * sizeof(index_t), dev));
    SPUME_HIP_CHECK(hipDeviceSynchronize());
}

SellDeviceFP64::~SellDeviceFP64() {
    static_cast<void>(hipFree(d_chunk_ptr_));
    static_cast<void>(hipFree(d_val_));
    static_cast<void>(hipFree(d_colidx_));
    static_cast<void>(hipFree(d_x_));
    static_cast<void>(hipFree(d_y_));
}

void SellDeviceFP64::spmv(std::span<const double> x, std::span<double> y) const {
    std::copy(x.begin(), x.end(), d_x_);
    const int grid = (nrows_ + kBlock - 1) / kBlock;
    sell_spmv_kernel<<<grid, kBlock>>>(d_chunk_ptr_, d_val_, d_colidx_, d_x_, d_y_, nrows_);
    SPUME_HIP_CHECK(hipGetLastError());
    SPUME_HIP_CHECK(hipDeviceSynchronize());
    std::copy(d_y_, d_y_ + nrows_, y.begin());
}

double SellDeviceFP64::kernel_ms(int reps) const {
    const int grid = (nrows_ + kBlock - 1) / kBlock;
    for (int i = 0; i < 3; ++i) { // warm-up
        sell_spmv_kernel<<<grid, kBlock>>>(d_chunk_ptr_, d_val_, d_colidx_, d_x_, d_y_, nrows_);
    }
    SPUME_HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start;
    hipEvent_t stop;
    SPUME_HIP_CHECK(hipEventCreate(&start));
    SPUME_HIP_CHECK(hipEventCreate(&stop));
    SPUME_HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < reps; ++i) {
        sell_spmv_kernel<<<grid, kBlock>>>(d_chunk_ptr_, d_val_, d_colidx_, d_x_, d_y_, nrows_);
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
