// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "backends/gpu/gpu_cheb.hpp"
#include "backends/gpu/gpu_reduce.hpp"
#include "backends/gpu/gpu_transfer.hpp"
#include "core/amg.hpp"
#include "core/precond.hpp"
#include "core/sell.hpp"
#include "core/types.hpp"

// GPU-resident FP32 multigrid V-cycle (ADR-0017, ADR-0002) -- the M3 flagship
// preconditioner assembled from the resident primitives: the FP32 Chebyshev
// smoother (gpu_cheb), the aggregation transfers (gpu_transfer), and a fused
// FP64 residual (res = r - A z in one pass). Mirrors the CPU AmgPrecond<float>
// V-cycle (core/amg_precond.hpp) precision-for-precision: FP64 operators,
// residual, and transfers; FP32 smoother interior. Verified in-class against it.
//
// Residency (ADR-0017): every level's operator, smoother, and aggregation live
// in unified memory; a cycle is a sequence of kernels on the default stream with
// a SINGLE host sync. The coarsest level is solved on the CPU (FP64 CG) -- the
// small, launch-latency-bound level belongs on the CPU (the heterogeneous M3
// design), and it reuses the exact reference solver so the coarse correction
// matches bitwise. The only in-class divergence is the FP32 smoother interior.

namespace spume::gpu {

class VcycleDeviceFP32 {
public:
    // kcycle enables Notay's Krylov-accelerated coarse correction (GAMG-parity
    // convergence on graded meshes) at coarse levels 1..kcycle_max_levels; the
    // finest level stays a plain V-cycle (the outer FCG already accelerates it).
    VcycleDeviceFP32(const Csr& fine, const std::vector<Aggregation>& aggs,
                     ChebyshevOptions smoother_opt = {}, double coarse_tol = 1e-2,
                     int coarse_max_iter = 500, bool kcycle = false,
                     int kcycle_max_levels = 5);
    ~VcycleDeviceFP32();
    VcycleDeviceFP32(const VcycleDeviceFP32&) = delete;
    VcycleDeviceFP32& operator=(const VcycleDeviceFP32&) = delete;

    // z ~= M^{-1} r (one V-cycle). Host-visible FP64 views sized to the finest
    // nrows; staged through resident buffers. Blocking.
    void apply(std::span<const double> r, std::span<double> z) const;

    // Device-to-device apply for the resident FCG: one V-cycle on device buffers
    // (no host staging). Runs the internal coarse-solve sync; the caller syncs
    // again at its next reduction. r_dev, z_dev have the finest nrows.
    void apply_device(const double* r_dev, double* z_dev) const { cycle(0, r_dev, z_dev); }

    // Mean cycle time over `reps` applies on the resident input (call apply()
    // once first). Includes the CPU coarse solve, so it is the true apply cost.
    double kernel_ms(int reps) const;

    int num_levels() const { return static_cast<int>(levels_.size()) + 1; }

private:
    struct Level {
        index_t nrows = 0;
        index_t ncoarse = 0;
        // FP64 operator (for the fused residual), resident.
        offset_t* d_chunk_ptr = nullptr;
        double* d_val = nullptr;
        index_t* d_colidx = nullptr;
        // FP32 smoother + FP64 aggregation transfers.
        std::unique_ptr<ChebyshevDeviceFP32> smoother;
        std::unique_ptr<AggTransferResident> transfer;
        // Workspace (resident): residual + post-smooth (nrows), coarse rhs/corr.
        double* d_res = nullptr;
        double* d_sm = nullptr;
        double* d_rc = nullptr;
        double* d_ec = nullptr;
        // K-cycle workspace (nrows each): the two Krylov directions c,d, their
        // operator images v=Ac,w=Ad, and the intermediate residual r1.
        double* d_kc = nullptr;
        double* d_kv = nullptr;
        double* d_kd = nullptr;
        double* d_kw = nullptr;
        double* d_kr = nullptr;
    };

    void cycle(std::size_t lvl, const double* r_dev, double* z_dev) const;
    // Coarse correction at level lvl: host CG (coarsest), a plain V-cycle, or
    // (kcycle) the Krylov-accelerated K-cycle.
    void coarse_solve(std::size_t lvl, const double* b_dev, double* x_dev) const;
    void coarse_solve_host(const double* rc_dev, double* ec_dev, index_t ncoarse) const;

    std::vector<Level> levels_;
    Sell<double> coarsest_; // host operator for the CPU coarse solve
    double coarse_tol_;
    int coarse_max_iter_;
    bool kcycle_ = false;
    int kcycle_max_levels_ = 5;
    DotDeviceFP64 dot_; // resident FP64 reductions for the K-cycle projection
    mutable double* d_r0_ = nullptr; // finest input scratch
    mutable double* d_z0_ = nullptr; // finest output scratch
};

} // namespace spume::gpu
