// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// Verify-then-bench gate for the GPU FP32 Chebyshev smoother (ADR-0016/0017):
// build an equilibrated FP32 operator, apply the smoother on the GPU, VERIFY
// against the CPU spume::ChebyshevPrecond<float> within the reorder-tolerance
// class (FP32 is reorder-sensitive -- not bitwise), then MEASURE bandwidth.
// Exits non-zero only on a correctness failure; skips (exit 0) with no GPU.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include "backends/gpu/gpu_cheb.hpp"
#include "backends/gpu/gpu_spmv.hpp" // spume::gpu::available()
#include "core/equilibrate.hpp"
#include "core/formats.hpp"
#include "core/poisson.hpp"
#include "core/precond.hpp"
#include "core/sell.hpp"
#include "core/types.hpp"

int main(int argc, char** argv) {
    if (!spume::gpu::available()) {
        std::printf("gpu-cheb-check: no HIP device present -- skipping (exit 0)\n");
        return 0;
    }

    const spume::index_t n = (argc > 1) ? static_cast<spume::index_t>(std::atoi(argv[1])) : 128;
    const spume::ChebyshevOptions copt{5, 30.0};

    const spume::Csr csr = spume::coo_to_csr(spume::gen::poisson7(n, n, n));
    const spume::EqOperator<float> op = spume::make_eq_operator<float>(csr);
    const auto nr = static_cast<std::size_t>(op.a.nrows);
    std::printf("gpu-cheb-check: poisson7 %d^3  nrows=%zu  steps=%d eta=%.0f\n", n, nr,
                copt.steps, copt.eta);

    // Deterministic non-smooth residual in [-1, 1].
    std::vector<double> r(nr);
    for (std::size_t i = 0; i < nr; ++i) {
        const std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U + 12345U;
        r[i] = static_cast<double>(h >> 16) / 32768.0 - 1.0;
    }
    std::vector<double> z_ref(nr, 0.0);
    std::vector<double> z_gpu(nr, 0.0);

    // CPU reference (op is copied into the preconditioner, left intact for the GPU).
    spume::ChebyshevPrecond<float> cpu(op, copt);
    cpu.apply(std::span<const double>(r), std::span<double>(z_ref));

    // GPU smoother.
    spume::gpu::ChebyshevDeviceFP32 gpu(op, copt);
    gpu.apply(std::span<const double>(r), std::span<double>(z_gpu));

    // In-class verify. Scale the error by the SOLUTION magnitude ||z||_inf, not
    // per-element |z_i|: z = M^{-1} r has near-zero entries where the smoothing
    // polynomial cancels, which would inflate an elementwise relative error even
    // though the absolute error there is still at FP32 level. Both the L2
    // relative difference and max_abs/||z||_inf are robust to that.
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
    // FP32 Chebyshev in-class bar (ADR-0017 measured max-rel 2.7e-7 for this
    // apply); 1e-4 is a comfortable ceiling that still catches a real defect.
    const bool pass = (l2_rel < 1e-4) && (max_scaled < 1e-4);
    std::printf("  VERIFY: max_abs/||z||inf=%.3e  L2_rel=%.3e -> %s\n", max_scaled, l2_rel,
                pass ? "PASS" : "FAIL");

    // Measure: FP32 Chebyshev apply traffic model (matches bench/main.cpp).
    const double nn = static_cast<double>(nr);
    const double spmv_bytes_t = op.a.spmv_bytes();
    const double w = 4.0 * nn;
    const double model = (copt.steps - 1) * spmv_bytes_t + 16.0 * nn +
                         w * (4.0 + 3.0 * copt.steps + 6.0 * (copt.steps - 1));
    const double ms = gpu.kernel_ms(50);
    const double gbs = model / (ms * 1e-3) / 1e9;
    constexpr double kLpddr5xPeak = 256.0;
    std::printf("  MEASURE: GPU %.3f ms/apply  %.1f GB/s (%.0f%% of %.0f GB/s LPDDR5X peak)\n",
                ms, gbs, 100.0 * gbs / kLpddr5xPeak, kLpddr5xPeak);
    std::printf("  RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
