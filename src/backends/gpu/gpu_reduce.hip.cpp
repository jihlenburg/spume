// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// HIP implementation of the GPU FP64 dot-product reduction (ADR-0017).

#include "backends/gpu/gpu_reduce.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
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
constexpr int kMaxBlocks = 256; // bounds the partial array; grid-stride covers n

// Per-block reduction of x . y into partial[blockIdx]. Grid-stride so a fixed,
// small grid covers any n (keeping the host-side partial sum cheap).
__global__ void dot_k(const double* __restrict__ x, const double* __restrict__ y,
                      double* __restrict__ partial, int n) {
    __shared__ double sdata[kBlock];
    const int tid = static_cast<int>(threadIdx.x);
    double sum = 0.0;
    for (int i = static_cast<int>(blockIdx.x * blockDim.x) + tid; i < n;
         i += static_cast<int>(blockDim.x * gridDim.x)) {
        sum += x[i] * y[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

} // namespace

DotDeviceFP64::DotDeviceFP64() {
    SPUME_HIP_CHECK(hipMallocManaged(&d_partial_, kMaxBlocks * sizeof(double)));
}

DotDeviceFP64::~DotDeviceFP64() {
    static_cast<void>(hipFree(d_partial_));
}

double DotDeviceFP64::dot(const double* x_dev, const double* y_dev, index_t n) const {
    const int ni = static_cast<int>(n);
    int grid = (ni + kBlock - 1) / kBlock;
    grid = std::clamp(grid, 1, kMaxBlocks);
    dot_k<<<grid, kBlock>>>(x_dev, y_dev, d_partial_, ni);
    SPUME_HIP_CHECK(hipGetLastError());
    SPUME_HIP_CHECK(hipDeviceSynchronize());
    // Sum the (few) per-block partials on the host, left-to-right.
    double s = 0.0;
    for (int b = 0; b < grid; ++b) {
        s += d_partial_[b];
    }
    return s;
}

} // namespace spume::gpu
