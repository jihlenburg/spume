// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// Verify-then-bench gate for the GPU SELL SpMV (ADR-0016 discipline on the GPU,
// ADR-0017 in-class verification): build a poisson7 operator, run it on the GPU,
// VERIFY the result against the CPU reference (spume::spmv) within the
// reorder-tolerance class, then MEASURE achieved bandwidth. Exits non-zero only
// on a correctness failure; skips cleanly (exit 0) when no HIP device is present
// so it is a no-op on CPU-only machines.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#include "backends/gpu/gpu_spmv.hpp"
#include "core/formats.hpp"
#include "core/poisson.hpp"
#include "core/sell.hpp"
#include "core/spmv.hpp"
#include "core/types.hpp"

int main(int argc, char** argv) {
    if (!spume::gpu::available()) {
        std::printf("gpu-spmv-check: no HIP device present -- skipping (exit 0)\n");
        return 0;
    }

    const spume::index_t n = (argc > 1) ? static_cast<spume::index_t>(std::atoi(argv[1])) : 128;
    const spume::Sell<double> a =
        spume::sell_from_csr(spume::coo_to_csr(spume::gen::poisson7(n, n, n)));
    const auto nr = static_cast<std::size_t>(a.nrows);
    const auto nc = static_cast<std::size_t>(a.ncols);
    const double model_bytes = a.spmv_bytes();
    std::printf("gpu-spmv-check: poisson7 %d^3  nrows=%d  nnz=%lld  model=%.1f MB/SpMV\n",
                n, a.nrows, static_cast<long long>(a.nnz), model_bytes / 1e6);

    // Non-smooth, deterministic x in [1, 2): a SMOOTH x would make the discrete
    // Laplacian y = 6 x_i - neighbours nearly cancel to zero, inflating the
    // RELATIVE error (tiny y) even though the absolute error stays at FMA level.
    // An uncorrelated x keeps y ~ O(1), so max_rel measures true agreement.
    std::vector<double> x(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        const std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U + 12345U;
        x[i] = 1.0 + static_cast<double>(h >> 16) / 65536.0;
    }
    std::vector<double> y_ref(nr, 0.0);
    std::vector<double> y_gpu(nr, 0.0);

    // CPU reference (the source of truth) + a serial baseline time.
    const int cpu_reps = 5;
    const auto c0 = std::chrono::steady_clock::now();
    for (int i = 0; i < cpu_reps; ++i) {
        spume::spmv<double>(a, std::span<const double>(x), std::span<double>(y_ref),
                            spume::Dispatch::reference);
    }
    const auto c1 = std::chrono::steady_clock::now();
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(c1 - c0).count() / cpu_reps;

    // GPU: build the resident operator, run once, verify.
    spume::gpu::SellDeviceFP64 dev(a);
    dev.spmv(x, y_gpu);

    double max_abs = 0.0;
    double max_rel = 0.0;
    std::int64_t bit_mismatch = 0;
    for (std::size_t i = 0; i < nr; ++i) {
        const double g = y_gpu[i];
        const double r = y_ref[i];
        const double ad = std::fabs(g - r);
        const double rd = ad / (std::fabs(r) + 1e-300);
        max_abs = std::max(max_abs, ad);
        max_rel = std::max(max_rel, rd);
        std::uint64_t gb = 0;
        std::uint64_t rb = 0;
        std::memcpy(&gb, &g, 8);
        std::memcpy(&rb, &r, 8);
        if (gb != rb) {
            ++bit_mismatch;
        }
    }
    // FP64 in-class bar: host/device may differ only at FMA-contraction /
    // reordering level (ADR-0017) -- a few ULP over ~7 terms. Far tighter than
    // the FP32 class (ADR-0017's Chebyshev apply verified at max-rel 2.7e-7).
    const bool pass = max_rel < 1e-12;
    std::printf("  VERIFY: max_abs=%.3e  max_rel=%.3e  bitwise_mismatch=%lld/%zu -> %s%s\n",
                max_abs, max_rel, static_cast<long long>(bit_mismatch), nr,
                pass ? "PASS" : "FAIL", bit_mismatch == 0 ? " (bitwise-exact)" : "");

    // Measure GPU kernel bandwidth (kernel only, HIP events).
    const double gpu_ms = dev.kernel_ms(50);
    const double gpu_gbs = model_bytes / (gpu_ms * 1e-3) / 1e9;
    const double cpu_gbs = model_bytes / (cpu_ms * 1e-3) / 1e9;
    constexpr double kLpddr5xPeak = 256.0; // GB/s theoretical, Strix Halo

    std::printf("  MEASURE: GPU %.3f ms/SpMV  %.1f GB/s (%.0f%% of %.0f GB/s LPDDR5X peak)\n",
                gpu_ms, gpu_gbs, 100.0 * gpu_gbs / kLpddr5xPeak, kLpddr5xPeak);
    std::printf("           CPU ref (serial) %.3f ms  %.1f GB/s  -> GPU %.2fx vs CPU-ref\n",
                cpu_ms, cpu_gbs, cpu_ms / gpu_ms);
    std::printf("  RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
