// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// Verify-then-bench gate for the GPU-resident FP32 V-cycle (ADR-0016/0017):
// build one aggregation hierarchy, construct the CPU AmgPrecond<float> V-cycle
// and the GPU VcycleDeviceFP32 from it, apply both to the same residual, and
// check they agree in-class (only the FP32 smoother interior differs; operators,
// residual, transfers, and the coarse CG all match). Then measure the cycle
// cost. Skips (exit 0) with no GPU.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include "backends/gpu/gpu_spmv.hpp" // available()
#include "backends/gpu/gpu_vcycle.hpp"
#include "core/amg.hpp"
#include "core/amg_precond.hpp"
#include "core/formats.hpp"
#include "core/poisson.hpp"
#include "core/sell.hpp"
#include "core/types.hpp"

int main(int argc, char** argv) {
    if (!spume::gpu::available()) {
        std::printf("gpu-vcycle-check: no HIP device present -- skipping (exit 0)\n");
        return 0;
    }

    const spume::index_t n = (argc > 1) ? static_cast<spume::index_t>(std::atoi(argv[1])) : 96;
    const spume::Csr csr = spume::coo_to_csr(spume::gen::poisson7(n, n, n));
    const std::vector<spume::Aggregation> aggs = spume::aggregate_hierarchy(csr);
    const auto nr = static_cast<std::size_t>(csr.nrows);
    std::printf("gpu-vcycle-check: poisson7 %d^3  nrows=%zu  hierarchy levels=%zu (+coarsest)\n",
                n, nr, aggs.size());

    // Non-smooth deterministic residual.
    std::vector<double> r(nr);
    for (std::size_t i = 0; i < nr; ++i) {
        const std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U + 12345U;
        r[i] = static_cast<double>(h >> 16) / 32768.0 - 1.0;
    }
    std::vector<double> z_ref(nr, 0.0);
    std::vector<double> z_gpu(nr, 0.0);

    // CPU reference: AmgPrecond<float> V-cycle (defaults -> kcycle=false, reference
    // dispatch), the exact structure the GPU cycle mirrors.
    const spume::AmgPrecond<float> cpu(csr, aggs);
    spume::gpu::VcycleDeviceFP32 gpu(csr, aggs);
    if (gpu.num_levels() != cpu.num_levels()) {
        std::printf("  level-count mismatch: gpu=%d cpu=%d -> FAIL\n", gpu.num_levels(),
                    cpu.num_levels());
        return 1;
    }

    const auto c0 = std::chrono::steady_clock::now();
    cpu.apply(std::span<const double>(r), std::span<double>(z_ref));
    const auto c1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

    gpu.apply(std::span<const double>(r), std::span<double>(z_gpu));

    // In-class verify, scaled by ||z||_inf (robust to near-zero cycle outputs).
    double max_abs = 0.0;
    double zinf = 0.0;
    double ndiff2 = 0.0;
    double nref2 = 0.0;
    for (std::size_t i = 0; i < nr; ++i) {
        const double d = z_gpu[i] - z_ref[i];
        max_abs = std::max(max_abs, std::fabs(d));
        zinf = std::max(zinf, std::fabs(z_ref[i]));
        ndiff2 += d * d;
        nref2 += z_ref[i] * z_ref[i];
    }
    const double max_scaled = max_abs / (zinf + 1e-300);
    const double l2_rel = std::sqrt(ndiff2) / (std::sqrt(nref2) + 1e-300);
    // Only the FP32 smoother interior differs (operators/residual/transfers FP64,
    // coarse CG identical), compounded over 2*levels smooths; 1e-3 is a safe bar.
    const bool pass = (l2_rel < 1e-3) && (max_scaled < 1e-3);
    std::printf("  VERIFY: max_abs/||z||inf=%.3e  L2_rel=%.3e -> %s\n", max_scaled, l2_rel,
                pass ? "PASS" : "FAIL");

    const double gpu_ms = gpu.kernel_ms(50);
    std::printf("  MEASURE: GPU %.3f ms/V-cycle   CPU AmgPrecond<float> %.3f ms/apply  (%.2fx)\n",
                gpu_ms, cpu_ms, cpu_ms / gpu_ms);
    std::printf("  RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
