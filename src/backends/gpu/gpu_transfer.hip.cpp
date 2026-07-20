// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// HIP implementation of the aggregation transfer operators (ADR-0017).

#include "backends/gpu/gpu_transfer.hpp"

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
int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

// rc[agg[i]] += res[i]. Several fine rows map to one aggregate, so the scatter
// needs an atomic; FP64 add is order-dependent, hence in-class (not bitwise) --
// acceptable in the FP32-firewalled preconditioner interior (ADR-0002).
__global__ void restrict_k(const std::int32_t* __restrict__ agg,
                           const double* __restrict__ res, double* __restrict__ rc, int nfine) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nfine) {
        atomicAdd(&rc[agg[i]], res[i]);
    }
}

// z[i] += ec[agg[i]]: a gather-add; each fine row writes its own z[i], so this
// is conflict-free and bitwise-reproducible.
__global__ void prolong_add_k(const std::int32_t* __restrict__ agg,
                              const double* __restrict__ ec, double* __restrict__ z, int nfine) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nfine) {
        z[i] += ec[agg[i]];
    }
}

} // namespace

AggTransferResident::AggTransferResident(const Aggregation& agg)
    : nfine_(static_cast<index_t>(agg.agg.size())), ncoarse_(agg.ncoarse) {
    SPUME_HIP_CHECK(hipMallocManaged(&d_agg_, static_cast<std::size_t>(nfine_) * sizeof(index_t)));
    std::copy(agg.agg.begin(), agg.agg.end(), d_agg_);
    int dev = 0;
    SPUME_HIP_CHECK(hipGetDevice(&dev));
    static_cast<void>(
        hipMemPrefetchAsync(d_agg_, static_cast<std::size_t>(nfine_) * sizeof(index_t), dev));
    SPUME_HIP_CHECK(hipDeviceSynchronize());
}

AggTransferResident::~AggTransferResident() {
    static_cast<void>(hipFree(d_agg_));
}

void AggTransferResident::restrict_device(const double* res_dev, double* rc_dev) const {
    SPUME_HIP_CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(rc_dev), 0,
                                static_cast<std::size_t>(ncoarse_) * sizeof(double)));
    restrict_k<<<grid_for(nfine_), kBlock>>>(d_agg_, res_dev, rc_dev, nfine_);
    SPUME_HIP_CHECK(hipGetLastError());
}

void AggTransferResident::prolong_add_device(const double* ec_dev, double* z_dev) const {
    prolong_add_k<<<grid_for(nfine_), kBlock>>>(d_agg_, ec_dev, z_dev, nfine_);
    SPUME_HIP_CHECK(hipGetLastError());
}

} // namespace spume::gpu
