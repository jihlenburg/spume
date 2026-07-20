// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// Verify-then-bench gate for the GPU FP64 dot reduction (ADR-0016/0017): dot two
// resident vectors on the GPU, check against spume::dot (reference) in-class
// (the parallel reduction order differs), then measure bandwidth (2n reads).
// Skips (exit 0) with no GPU. HIP TU (manages managed buffers directly).

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include "backends/gpu/gpu_reduce.hpp"
#include "backends/gpu/gpu_spmv.hpp" // available()
#include "core/reduce.hpp"
#include "core/types.hpp"

int main(int argc, char** argv) {
    if (!spume::gpu::available()) {
        std::printf("gpu-reduce-check: no HIP device present -- skipping (exit 0)\n");
        return 0;
    }

    const auto n = (argc > 1) ? static_cast<std::size_t>(std::atoll(argv[1])) : std::size_t{4000000};
    double* x = nullptr;
    double* y = nullptr;
    static_cast<void>(hipMallocManaged(&x, n * sizeof(double)));
    static_cast<void>(hipMallocManaged(&y, n * sizeof(double)));
    // Mixed-magnitude, mixed-sign so the reduction order actually matters.
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::sin(0.001 * static_cast<double>(i)) * std::pow(10.0, (i % 7) - 3.0);
        y[i] = std::cos(0.002 * static_cast<double>(i)) + 0.5;
    }
    std::printf("gpu-reduce-check: n=%zu\n", n);

    const double ref = spume::dot(std::span<const double>(x, n), std::span<const double>(y, n));

    spume::gpu::DotDeviceFP64 r;
    const double got = r.dot(x, y, static_cast<spume::index_t>(n));
    const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1e-300);
    const bool pass = rel < 1e-12;
    std::printf("  VERIFY: gpu=%.10e  ref=%.10e  rel=%.3e -> %s\n", got, ref, rel,
                pass ? "PASS" : "FAIL");

    // Bench: dot reads 2n FP64 = 16n bytes.
    const int reps = 100;
    for (int i = 0; i < 5; ++i) {
        static_cast<void>(r.dot(x, y, static_cast<spume::index_t>(n)));
    }
    const auto t0 = std::chrono::steady_clock::now();
    volatile double sink = 0.0;
    for (int i = 0; i < reps; ++i) {
        sink += r.dot(x, y, static_cast<spume::index_t>(n));
    }
    const auto t1 = std::chrono::steady_clock::now();
    static_cast<void>(sink);
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    const double gbs = 16.0 * static_cast<double>(n) / (ms * 1e-3) / 1e9;
    constexpr double kLpddr5xPeak = 256.0;
    std::printf("  MEASURE: %.3f ms/dot  %.1f GB/s (%.0f%% of %.0f GB/s peak)\n", ms, gbs,
                100.0 * gbs / kLpddr5xPeak, kLpddr5xPeak);

    static_cast<void>(hipFree(x));
    static_cast<void>(hipFree(y));
    std::printf("  RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
