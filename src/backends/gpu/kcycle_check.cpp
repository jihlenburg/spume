// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// Verify gate for the GPU K-cycle (ADR-0016/0017). On a graded operator (where a
// plain V-cycle stalls and the K-cycle recovers a mesh-independent rate), solve
// with the GPU FCG using (a) the V-cycle and (b) the K-cycle preconditioner, and
// the CPU AmgPrecond<float> K-cycle. Checks: the GPU K-cycle solves to FP64
// accuracy, converges in NO MORE iterations than the V-cycle (the acceleration),
// matches the CPU K-cycle iteration count (same-class), and the solutions agree
// in-class. Skips (exit 0) with no GPU.

#include <algorithm>
#include <cmath>
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
        std::printf("gpu-kcycle-check: no HIP device present -- skipping (exit 0)\n");
        return 0;
    }

    const spume::index_t n = (argc > 1) ? static_cast<spume::index_t>(std::atoi(argv[1])) : 64;
    const double tol = 1e-8;
    const int max_iter = 4000;

    // Graded z-conductivity 1..1000: the textbook-weak case for a plain V-cycle.
    const spume::Csr csr = spume::coo_to_csr(spume::gen::poisson7_graded(n, n, n, 1.0, 1000.0));
    const spume::Sell<double> A = spume::sell_from_csr(csr);
    const std::vector<spume::Aggregation> aggs = spume::aggregate_hierarchy(csr, 2000, 20);
    const auto nr = static_cast<std::size_t>(csr.nrows);
    std::vector<double> b(nr);
    for (std::size_t i = 0; i < nr; ++i) b[i] = 0.5 - static_cast<double>(i % 13) * 0.1;
    const double bnorm = spume::nrm2(std::span<const double>(b));
    std::printf("gpu-kcycle-check: poisson7_graded %d^3 (cz 1..1000)  nrows=%zu  levels=%zu\n", n,
                nr, aggs.size() + 1);

    // GPU V-cycle FCG.
    spume::gpu::FcgSolverGPU vsolver(csr, aggs, {}, 1e-2, 500, /*kcycle=*/false);
    std::vector<double> x_v(nr, 0.0);
    const auto rv = vsolver.solve(std::span<const double>(b), std::span<double>(x_v), tol, max_iter);

    // GPU K-cycle FCG.
    spume::gpu::FcgSolverGPU ksolver(csr, aggs, {}, 1e-2, 500, /*kcycle=*/true, 5);
    std::vector<double> x_k(nr, 0.0);
    const auto rk = ksolver.solve(std::span<const double>(b), std::span<double>(x_k), tol, max_iter);

    // CPU K-cycle FCG (reference).
    const spume::AmgPrecond<float> cpu_k(csr, aggs, {}, 1e-2, 500, spume::Dispatch::reference,
                                         /*kcycle=*/true, 5);
    std::vector<double> x_cpu(nr, 0.0);
    spume::SolveOptions opt;
    opt.tol = tol;
    opt.max_iter = max_iter;
    const auto rc = spume::fcg(A, cpu_k, std::span<const double>(b), std::span<double>(x_cpu), opt);

    // True residual of the GPU K-cycle solution.
    std::vector<double> ax(nr), res(nr);
    spume::spmv(A, std::span<const double>(x_k), std::span<double>(ax));
    for (std::size_t i = 0; i < nr; ++i) res[i] = b[i] - ax[i];
    const double true_relres = spume::nrm2(std::span<const double>(res)) / bnorm;

    // GPU-K vs CPU-K solution agreement.
    double dnum = 0.0, dden = 0.0;
    for (std::size_t i = 0; i < nr; ++i) {
        dnum += (x_k[i] - x_cpu[i]) * (x_k[i] - x_cpu[i]);
        dden += x_cpu[i] * x_cpu[i];
    }
    const double sol_rel = std::sqrt(dnum) / (std::sqrt(dden) + 1e-300);

    const bool solved = rk.converged && (true_relres <= 5.0 * tol);
    const bool accelerates = rk.iterations <= rv.iterations;          // K never worse than V
    const bool matches_cpu = rk.iterations <= 2 * rc.iterations + 10; // same-class convergence
    const bool sol_ok = sol_rel <= 1e-5;
    const bool pass = solved && accelerates && matches_cpu && sol_ok;

    std::printf("  FCG iters: V-cycle=%d  K-cycle(gpu)=%d  K-cycle(cpu)=%d\n", rv.iterations,
                rk.iterations, rc.iterations);
    std::printf("  VERIFY: gpu-K converged=%d  true_relres=%.2e  sol_rel(vs cpu-K)=%.2e -> %s\n",
                static_cast<int>(rk.converged), true_relres, sol_rel, pass ? "PASS" : "FAIL");
    std::printf("  RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
