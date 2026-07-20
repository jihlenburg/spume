// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// Verify-then-bench gate for the whole-solve GPU-resident FCG (ADR-0016/0017):
// solve A x = b on the GPU, then (1) recompute the TRUE residual on the host and
// require it below tol -- the honest "it actually solved the system" check -- and
// (2) compare against the CPU spume::fcg with the same FP32 AMG preconditioner:
// the iteration counts should be close (same-class preconditioner) and the
// solutions agree in-class. Skips (exit 0) with no GPU.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include "backends/gpu/gpu_fcg.hpp"
#include "backends/gpu/gpu_spmv.hpp" // available()
#include "core/amg.hpp"
#include "core/amg_precond.hpp"
#include "core/formats.hpp"
#include "core/poisson.hpp"
#include "core/reduce.hpp"
#include "core/sell.hpp"
#include "core/solver.hpp"
#include "core/spmv.hpp"
#include "core/types.hpp"

int main(int argc, char** argv) {
    if (!spume::gpu::available()) {
        std::printf("gpu-fcg-check: no HIP device present -- skipping (exit 0)\n");
        return 0;
    }

    const spume::index_t n = (argc > 1) ? static_cast<spume::index_t>(std::atoi(argv[1])) : 96;
    const double tol = 1e-8;
    const int max_iter = 2000;

    const spume::Csr csr = spume::coo_to_csr(spume::gen::poisson7(n, n, n));
    const spume::Sell<double> A = spume::sell_from_csr(csr);
    const std::vector<spume::Aggregation> aggs = spume::aggregate_hierarchy(csr);
    const auto nr = static_cast<std::size_t>(csr.nrows);

    std::vector<double> b(nr);
    for (std::size_t i = 0; i < nr; ++i) {
        b[i] = 0.5 - static_cast<double>(i % 13) * 0.1;
    }
    const double bnorm = spume::nrm2(std::span<const double>(b));
    std::printf("gpu-fcg-check: poisson7 %d^3  nrows=%zu  levels=%zu  tol=%.0e\n", n, nr,
                aggs.size(), tol);

    // GPU solve.
    spume::gpu::FcgSolverGPU solver(csr, aggs);
    std::vector<double> x_gpu(nr, 0.0);
    const auto g0 = std::chrono::steady_clock::now();
    const spume::gpu::FcgResult rg = solver.solve(std::span<const double>(b),
                                                  std::span<double>(x_gpu), tol, max_iter);
    const auto g1 = std::chrono::steady_clock::now();
    const double gpu_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

    // Honest check: TRUE residual of the GPU solution, recomputed on the host.
    std::vector<double> ax(nr);
    spume::spmv(A, std::span<const double>(x_gpu), std::span<double>(ax));
    std::vector<double> res(nr);
    for (std::size_t i = 0; i < nr; ++i) {
        res[i] = b[i] - ax[i];
    }
    const double true_relres = spume::nrm2(std::span<const double>(res)) / bnorm;

    // CPU reference solve (same FP32 AMG preconditioner).
    const spume::AmgPrecond<float> cpu_pc(csr, aggs);
    std::vector<double> x_cpu(nr, 0.0);
    spume::SolveOptions opt;
    opt.tol = tol;
    opt.max_iter = max_iter;
    const auto c0 = std::chrono::steady_clock::now();
    const spume::SolveResult rc =
        spume::fcg(A, cpu_pc, std::span<const double>(b), std::span<double>(x_cpu), opt);
    const auto c1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

    // Solution agreement (both converge to the same FP64 answer).
    double dnum = 0.0;
    double dden = 0.0;
    for (std::size_t i = 0; i < nr; ++i) {
        dnum += (x_gpu[i] - x_cpu[i]) * (x_gpu[i] - x_cpu[i]);
        dden += x_cpu[i] * x_cpu[i];
    }
    const double sol_rel = std::sqrt(dnum) / (std::sqrt(dden) + 1e-300);

    const bool solved = rg.converged && (true_relres <= 5.0 * tol);
    const bool iters_ok = rg.iterations <= 2 * rc.iterations + 10; // same-class convergence
    const bool sol_ok = sol_rel <= 1e-5;
    const bool pass = solved && iters_ok && sol_ok;

    std::printf("  VERIFY: gpu converged=%d  true_relres=%.2e  iters gpu=%d cpu=%d  sol_rel=%.2e"
                " -> %s\n",
                static_cast<int>(rg.converged), true_relres, rg.iterations, rc.iterations, sol_rel,
                pass ? "PASS" : "FAIL");
    std::printf("  MEASURE: GPU solve %.1f ms  CPU solve %.1f ms  (%.2fx)\n", gpu_ms, cpu_ms,
                cpu_ms / gpu_ms);
    std::printf("  RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
