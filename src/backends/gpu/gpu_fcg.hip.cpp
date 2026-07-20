// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// HIP implementation of the whole-solve GPU-resident flexible CG (ADR-0017).

#include "backends/gpu/gpu_fcg.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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

// y += a * x  (FP64).
__global__ void axpy_k(double a, const double* __restrict__ x, double* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += a * x[i];
    }
}

// y = x + a * y  (FP64).
__global__ void aypx_k(double a, const double* __restrict__ x, double* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] + a * y[i];
    }
}

} // namespace

FcgSolverGPU::FcgSolverGPU(const Csr& fine, const std::vector<Aggregation>& aggs,
                           ChebyshevOptions smoother_opt, double coarse_tol, int coarse_max_iter)
    : n_(fine.nrows), op_(sell_from_csr(fine)),
      precond_(fine, aggs, smoother_opt, coarse_tol, coarse_max_iter) {
    const auto n = static_cast<std::size_t>(n_);
    SPUME_HIP_CHECK(hipMallocManaged(&d_b_, n * sizeof(double)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_x_, n * sizeof(double)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_r_, n * sizeof(double)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_z_, n * sizeof(double)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_p_, n * sizeof(double)));
    SPUME_HIP_CHECK(hipMallocManaged(&d_q_, n * sizeof(double)));
}

FcgSolverGPU::~FcgSolverGPU() {
    static_cast<void>(hipFree(d_b_));
    static_cast<void>(hipFree(d_x_));
    static_cast<void>(hipFree(d_r_));
    static_cast<void>(hipFree(d_z_));
    static_cast<void>(hipFree(d_p_));
    static_cast<void>(hipFree(d_q_));
}

FcgResult FcgSolverGPU::solve(std::span<const double> b, std::span<double> x, double tol,
                              int max_iter) const {
    const int n = n_;
    const int g = grid_for(n);
    std::copy(b.begin(), b.end(), d_b_);
    std::copy(x.begin(), x.end(), d_x_);

    const double bnorm = std::sqrt(dot_.dot(d_b_, d_b_, n_));
    if (bnorm == 0.0) {
        std::fill(x.begin(), x.end(), 0.0);
        return FcgResult{0, 0.0, true};
    }

    // r = b - A x
    op_.residual_device(d_b_, d_x_, d_r_);
    double relres = std::sqrt(dot_.dot(d_r_, d_r_, n_)) / bnorm;
    if (relres <= tol) {
        std::copy(d_x_, d_x_ + n, x.begin());
        return FcgResult{0, relres, true};
    }

    // z = M^{-1} r ; p = z
    precond_.apply_device(d_r_, d_z_);
    SPUME_HIP_CHECK(hipMemcpy(d_p_, d_z_, static_cast<std::size_t>(n) * sizeof(double),
                              hipMemcpyDeviceToDevice));
    double rho = dot_.dot(d_r_, d_z_, n_); // (z, r)
    if (!(rho > 0.0) || !std::isfinite(rho)) {
        std::copy(d_x_, d_x_ + n, x.begin());
        return FcgResult{0, relres, false};
    }

    int it = 0;
    bool converged = false;
    for (it = 1; it <= max_iter; ++it) {
        op_.spmv_device(d_p_, d_q_); // q = A p
        const double pq = dot_.dot(d_p_, d_q_, n_);
        if (!(pq > 0.0) || !std::isfinite(pq)) {
            break; // not SPD / breakdown
        }
        const double alpha = rho / pq;
        axpy_k<<<g, kBlock>>>(alpha, d_p_, d_x_, n);  // x += alpha p
        axpy_k<<<g, kBlock>>>(-alpha, d_q_, d_r_, n); // r -= alpha q

        relres = std::sqrt(dot_.dot(d_r_, d_r_, n_)) / bnorm;
        if (relres <= tol) {
            converged = true;
            break;
        }

        precond_.apply_device(d_r_, d_z_); // z = M^{-1} r
        const double rho_new = dot_.dot(d_r_, d_z_, n_);
        if (!(rho_new > 0.0) || !std::isfinite(rho_new)) {
            break;
        }
        // Flexible (Polak-Ribiere) beta = -alpha (z, q) / rho.
        const double beta = -alpha * dot_.dot(d_z_, d_q_, n_) / rho;
        aypx_k<<<g, kBlock>>>(beta, d_z_, d_p_, n); // p = z + beta p
        rho = rho_new;
    }

    SPUME_HIP_CHECK(hipDeviceSynchronize());
    std::copy(d_x_, d_x_ + n, x.begin());
    return FcgResult{std::min(it, max_iter), relres, converged};
}

} // namespace spume::gpu
