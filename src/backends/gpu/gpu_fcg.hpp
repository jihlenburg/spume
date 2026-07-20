// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <span>
#include <vector>

#include "backends/gpu/gpu_reduce.hpp"
#include "backends/gpu/gpu_spmv.hpp"
#include "backends/gpu/gpu_vcycle.hpp"
#include "core/amg.hpp"
#include "core/precond.hpp"
#include "core/sell.hpp"
#include "core/types.hpp"

// Whole-solve GPU-resident flexible CG (ADR-0017) -- the M3 goal: A x = b solved
// entirely on the GPU, preconditioned by the resident FP32 V-cycle. The FP64
// outer Krylov keeps the answer at FP64 accuracy (ADR-0002); reduced precision
// lives only inside the V-cycle. All vectors stay device-resident across
// iterations; the only host traffic is b/x in and x out, plus the scalar dot
// results that drive alpha/beta (one sync per reduction). Flexible (Polak-
// Ribiere) beta tolerates the mildly-varying FP32 preconditioner, mirroring the
// CPU spume::fcg. This header is HIP-free.

namespace spume::gpu {

struct FcgResult {
    int iterations = 0;
    double rel_residual = 0.0;
    bool converged = false;
};

class FcgSolverGPU {
public:
    // coarse_max_iter defaults to 100 -- see VcycleDeviceFP32: the coarsest CG is
    // a preconditioner component, so capping it avoids the aggregation-stall
    // pathology (400+ coarse iters, GPU idle) with no effect on a healthy
    // coarsest. ~3x GPU speedup on a real stalled-coarsening matrix.
    FcgSolverGPU(const Csr& fine, const std::vector<Aggregation>& aggs,
                 ChebyshevOptions smoother_opt = {}, double coarse_tol = 1e-2,
                 int coarse_max_iter = 100, bool kcycle = false, int kcycle_max_levels = 5);
    ~FcgSolverGPU();
    FcgSolverGPU(const FcgSolverGPU&) = delete;
    FcgSolverGPU& operator=(const FcgSolverGPU&) = delete;

    // Solve A x = b to ||r||/||b|| <= tol (or max_iter). x is the initial guess
    // in and the solution out (host views sized to the fine nrows).
    FcgResult solve(std::span<const double> b, std::span<double> x, double tol,
                    int max_iter) const;

    index_t nrows() const { return n_; }

private:
    index_t n_ = 0;
    SellDeviceFP64 op_;        // fine FP64 operator (A p, b - A x)
    VcycleDeviceFP32 precond_; // resident FP32 V-cycle preconditioner
    DotDeviceFP64 dot_;        // resident FP64 reductions
    // Resident Krylov vectors.
    double* d_b_ = nullptr;
    double* d_x_ = nullptr;
    double* d_r_ = nullptr;
    double* d_z_ = nullptr;
    double* d_p_ = nullptr;
    double* d_q_ = nullptr;
};

} // namespace spume::gpu
