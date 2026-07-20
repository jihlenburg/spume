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
#include <cmath>
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

// y = A x  (FP64 SELL SpMV; for the K-cycle's operator applications A c, A d).
__global__ void spmv_k(const std::int64_t* __restrict__ chunk_ptr,
                       const double* __restrict__ val, const std::int32_t* __restrict__ colidx,
                       const double* __restrict__ x, double* __restrict__ y, int nrows) {
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
        acc += val[off] * x[colidx[off]];
        off += 8;
    }
    y[row] = acc;
}

// r = b - a * v  (FP64).
__global__ void bmav_k(const double* __restrict__ b, double a, const double* __restrict__ v,
                       double* __restrict__ r, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = b[i] - a * v[i];
    }
}

// x = a * c  (FP64 scale-into).
__global__ void scale_k(double a, const double* __restrict__ c, double* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = a * c[i];
    }
}

// x = a1 * c + a2 * d  (FP64 two-vector combination).
__global__ void lincomb2_k(double a1, const double* __restrict__ c, double a2,
                           const double* __restrict__ d, double* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = a1 * c[i] + a2 * d[i];
    }
}

} // namespace

VcycleDeviceFP32::VcycleDeviceFP32(const Csr& fine, const std::vector<Aggregation>& aggs,
                                   ChebyshevOptions smoother_opt, double coarse_tol,
                                   int coarse_max_iter, bool kcycle, int kcycle_max_levels)
    : coarse_tol_(coarse_tol), coarse_max_iter_(coarse_max_iter), kcycle_(kcycle),
      kcycle_max_levels_(kcycle_max_levels) {
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
        if (kcycle) { // Krylov-direction workspace, only when the K-cycle is on
            SPUME_HIP_CHECK(hipMallocManaged(&lev.d_kc, n * sizeof(double)));
            SPUME_HIP_CHECK(hipMallocManaged(&lev.d_kv, n * sizeof(double)));
            SPUME_HIP_CHECK(hipMallocManaged(&lev.d_kd, n * sizeof(double)));
            SPUME_HIP_CHECK(hipMallocManaged(&lev.d_kw, n * sizeof(double)));
            SPUME_HIP_CHECK(hipMallocManaged(&lev.d_kr, n * sizeof(double)));
        }

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
        static_cast<void>(hipFree(lev.d_kc)); // nullptr (no-op) when kcycle is off
        static_cast<void>(hipFree(lev.d_kv));
        static_cast<void>(hipFree(lev.d_kd));
        static_cast<void>(hipFree(lev.d_kw));
        static_cast<void>(hipFree(lev.d_kr));
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

void VcycleDeviceFP32::coarse_solve(std::size_t lvl, const double* b, double* x) const {
    if (lvl == levels_.size()) {
        coarse_solve_host(b, x, static_cast<index_t>(coarsest_.nrows));
        return;
    }
    const Level& L = levels_[lvl];
    const int n = L.nrows;
    const int g = grid_for(n);
    if (!kcycle_ || static_cast<int>(lvl) > kcycle_max_levels_) {
        cycle(lvl, b, x); // plain V-cycle
        return;
    }
    // Notay's aggregation-AMG K-cycle: project the coarse correction onto
    // span{c, d} with c = M b and d = M(b - alpha A c) (Galerkin/Ritz over the
    // two directions), a residual test skipping the second when the first
    // already cuts the residual >=4x -- keeping the fan-out near-linear.
    cycle(lvl, b, L.d_kc); // c = M b
    spmv_k<<<g, kBlock>>>(L.d_chunk_ptr, L.d_val, L.d_colidx, L.d_kc, L.d_kv, n); // v = A c
    const double rho1 = dot_.dot(L.d_kc, b, n);
    const double sig1 = dot_.dot(L.d_kc, L.d_kv, n);
    if (!(sig1 > 0.0)) {
        scale_k<<<g, kBlock>>>(1.0, L.d_kc, x, n); // not SPD-usable: x = c
        return;
    }
    const double alpha = rho1 / sig1;
    bmav_k<<<g, kBlock>>>(b, alpha, L.d_kv, L.d_kr, n); // r1 = b - alpha v
    const double nb = dot_.dot(b, b, n);
    const double nr = dot_.dot(L.d_kr, L.d_kr, n);
    if (nr <= 0.0625 * nb) { // ||r1|| <= 0.25 ||b|| : one direction is enough
        scale_k<<<g, kBlock>>>(alpha, L.d_kc, x, n); // x = alpha c
        return;
    }
    cycle(lvl, L.d_kr, L.d_kd); // d = M r1
    spmv_k<<<g, kBlock>>>(L.d_chunk_ptr, L.d_val, L.d_colidx, L.d_kd, L.d_kw, n); // w = A d
    const double gam = dot_.dot(L.d_kc, L.d_kw, n);
    const double bet = dot_.dot(L.d_kd, L.d_kw, n);
    const double rho2 = dot_.dot(L.d_kd, b, n);
    const double det = sig1 * bet - gam * gam;
    if (!(std::abs(det) > 0.0)) { // directions dependent: fall back to one
        scale_k<<<g, kBlock>>>(alpha, L.d_kc, x, n);
        return;
    }
    const double a1 = (rho1 * bet - gam * rho2) / det;
    const double a2 = (sig1 * rho2 - gam * rho1) / det;
    lincomb2_k<<<g, kBlock>>>(a1, L.d_kc, a2, L.d_kd, x, n); // x = a1 c + a2 d
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

    // coarse correction ec = A_c^{-1} rc: host CG at the coarsest, else a plain
    // V-cycle or the Krylov-accelerated K-cycle at level lvl+1.
    coarse_solve(lvl + 1, L.d_rc, L.d_ec);

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
