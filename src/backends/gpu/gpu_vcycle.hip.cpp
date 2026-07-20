// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// HIP implementation of the GPU-resident FP32 V-cycle (ADR-0017). Assembles the
// resident primitives (smoother, transfers) with a fused FP64 residual and a CPU
// coarse solve; mirrors the CPU AmgPrecond<float> V-cycle.

#include "backends/gpu/gpu_vcycle.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/equilibrate.hpp"
#include "core/sell.hpp"
#include "core/solver.hpp"

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

// Fused FP64 residual res = r - A z in a single pass (SELL SpMV with a subtract
// epilogue), so the intermediate A z is never written to or read from memory --
// the top byte-saving opt for the cycle.
__global__ void residual_k(const std::int64_t* __restrict__ chunk_ptr,
                           const double* __restrict__ val, const std::int32_t* __restrict__ colidx,
                           const double* __restrict__ r, const double* __restrict__ z,
                           double* __restrict__ res, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const int c = row >> 3;
    const int lane = row & 7;
    const std::int64_t base = chunk_ptr[c];
    const std::int64_t w = (chunk_ptr[c + 1] - base) >> 3;
    double acc = 0.0;
    std::int64_t off = base + lane;
    for (std::int64_t j = 0; j < w; ++j) {
        acc += val[off] * z[colidx[off]];
        off += 8;
    }
    res[row] = r[row] - acc;
}

// y += a * x  (FP64).
__global__ void axpy_k(double a, const double* __restrict__ x, double* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += a * x[i];
    }
}

} // namespace

VcycleDeviceFP32::VcycleDeviceFP32(const Csr& fine, const std::vector<Aggregation>& aggs,
                                   ChebyshevOptions smoother_opt, double coarse_tol,
                                   int coarse_max_iter)
    : coarse_tol_(coarse_tol), coarse_max_iter_(coarse_max_iter) {
    int dev = 0;
    SPUME_HIP_CHECK(hipGetDevice(&dev));

    Csr cur = fine;
    for (const Aggregation& agg : aggs) {
        if (static_cast<std::size_t>(cur.nrows) != agg.agg.size()) {
            break; // malformed hierarchy guard
        }
        Level lev;
        lev.nrows = cur.nrows;
        lev.ncoarse = agg.ncoarse;

        // FP64 operator for the residual.
        const Sell<double> a64 = sell_from_csr(cur);
        const std::size_t ncp = a64.chunk_ptr.size();
        const auto stored = static_cast<std::size_t>(a64.stored());
        SPUME_HIP_CHECK(hipMallocManaged(&lev.d_chunk_ptr, ncp * sizeof(offset_t)));
        SPUME_HIP_CHECK(hipMallocManaged(&lev.d_val, stored * sizeof(double)));
        SPUME_HIP_CHECK(hipMallocManaged(&lev.d_colidx, stored * sizeof(index_t)));
        std::copy(a64.chunk_ptr.begin(), a64.chunk_ptr.end(), lev.d_chunk_ptr);
        std::copy(a64.val.begin(), a64.val.end(), lev.d_val);
        std::copy(a64.colidx.begin(), a64.colidx.end(), lev.d_colidx);
        static_cast<void>(hipMemPrefetchAsync(lev.d_chunk_ptr, ncp * sizeof(offset_t), dev));
        static_cast<void>(hipMemPrefetchAsync(lev.d_val, stored * sizeof(double), dev));
        static_cast<void>(hipMemPrefetchAsync(lev.d_colidx, stored * sizeof(index_t), dev));

        // FP32 smoother (equilibrated) and aggregation transfers.
        lev.smoother = std::make_unique<ChebyshevDeviceFP32>(make_eq_operator<float>(cur),
                                                             smoother_opt);
        lev.transfer = std::make_unique<AggTransferResident>(agg);

        // Workspace.
        const auto n = static_cast<std::size_t>(lev.nrows);
        const auto nc = static_cast<std::size_t>(lev.ncoarse);
        SPUME_HIP_CHECK(hipMallocManaged(&lev.d_res, n * sizeof(double)));
        SPUME_HIP_CHECK(hipMallocManaged(&lev.d_sm, n * sizeof(double)));
        SPUME_HIP_CHECK(hipMallocManaged(&lev.d_rc, nc * sizeof(double)));
        SPUME_HIP_CHECK(hipMallocManaged(&lev.d_ec, nc * sizeof(double)));

        levels_.push_back(std::move(lev));
        cur = galerkin(cur, agg);
    }
    coarsest_ = sell_from_csr(cur); // host FP64 operator for the CPU coarse solve

    const auto n0 = static_cast<std::size_t>(levels_.empty() ? 0 : levels_.front().nrows);
    SPUME_HIP_CHECK(hipMallocManaged(&d_r0_, n0 * sizeof(double)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_z0_, n0 * sizeof(double)));
    SPUME_HIP_CHECK(hipDeviceSynchronize());
}

VcycleDeviceFP32::~VcycleDeviceFP32() {
    for (Level& lev : levels_) {
        static_cast<void>(hipFree(lev.d_chunk_ptr));
        static_cast<void>(hipFree(lev.d_val));
        static_cast<void>(hipFree(lev.d_colidx));
        static_cast<void>(hipFree(lev.d_res));
        static_cast<void>(hipFree(lev.d_sm));
        static_cast<void>(hipFree(lev.d_rc));
        static_cast<void>(hipFree(lev.d_ec));
    }
    static_cast<void>(hipFree(d_r0_));
    static_cast<void>(hipFree(d_z0_));
}

void VcycleDeviceFP32::coarse_solve_host(const double* rc_dev, double* ec_dev,
                                         index_t ncoarse) const {
    // The coarse level is small and launch-latency-bound -- solve it on the CPU
    // (the heterogeneous M3 design), reusing the exact reference FP64 CG so the
    // coarse correction matches the CPU AmgPrecond bitwise. Sync so the restrict
    // kernel's writes to rc_dev are visible on the host (unified memory).
    SPUME_HIP_CHECK(hipDeviceSynchronize());
    const auto nc = static_cast<std::size_t>(ncoarse);
    std::vector<double> ec_host(nc, 0.0);
    SolveOptions opt;
    opt.tol = coarse_tol_;
    opt.max_iter = coarse_max_iter_;
    cg(coarsest_, std::span<const double>(rc_dev, nc), std::span<double>(ec_host), opt);
    std::copy(ec_host.begin(), ec_host.end(), ec_dev); // visible to the next kernel
}

void VcycleDeviceFP32::cycle(std::size_t lvl, const double* r_dev, double* z_dev) const {
    const Level& L = levels_[lvl];
    const int n = L.nrows;
    const int g = grid_for(n);

    // pre-smooth: z <- S(r)
    L.smoother->smooth_device(r_dev, z_dev);
    // res = r - A z (fused)
    residual_k<<<g, kBlock>>>(L.d_chunk_ptr, L.d_val, L.d_colidx, r_dev, z_dev, L.d_res, n);
    // rc = P^T res
    L.transfer->restrict_device(L.d_res, L.d_rc);

    // coarse correction ec = A_c^{-1} rc
    if (lvl + 1 == levels_.size()) {
        coarse_solve_host(L.d_rc, L.d_ec, L.ncoarse);
    } else {
        cycle(lvl + 1, L.d_rc, L.d_ec);
    }

    // z += P ec
    L.transfer->prolong_add_device(L.d_ec, z_dev);
    // post-smooth: z += S(r - A z)
    residual_k<<<g, kBlock>>>(L.d_chunk_ptr, L.d_val, L.d_colidx, r_dev, z_dev, L.d_res, n);
    L.smoother->smooth_device(L.d_res, L.d_sm);
    axpy_k<<<g, kBlock>>>(1.0, L.d_sm, z_dev, n);
}

void VcycleDeviceFP32::apply(std::span<const double> r, std::span<double> z) const {
    const auto n = static_cast<std::size_t>(levels_.front().nrows);
    std::copy(r.begin(), r.begin() + static_cast<std::ptrdiff_t>(n), d_r0_);
    cycle(0, d_r0_, d_z0_);
    SPUME_HIP_CHECK(hipGetLastError());
    SPUME_HIP_CHECK(hipDeviceSynchronize());
    std::copy(d_z0_, d_z0_ + n, z.begin());
}

double VcycleDeviceFP32::kernel_ms(int reps) const {
    for (int i = 0; i < 3; ++i) { // warm-up
        cycle(0, d_r0_, d_z0_);
    }
    SPUME_HIP_CHECK(hipDeviceSynchronize());

    // The cycle mixes GPU kernels with a CPU coarse solve, so wall-clock (not HIP
    // events) is the honest apply cost.
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        cycle(0, d_r0_, d_z0_);
    }
    SPUME_HIP_CHECK(hipDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

} // namespace spume::gpu
