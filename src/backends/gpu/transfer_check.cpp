// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// Verify gate for the GPU aggregation transfer operators (ADR-0016/0017): build
// a real aggregation, restrict/prolong on the GPU, and check against a CPU
// reference -- prolong bitwise (conflict-free gather), restrict in-class (the
// scatter uses FP64 atomicAdd, so the sum order varies). Skips (exit 0) with no
// GPU. Uses managed buffers directly (it is a HIP TU) since the transfer API is
// device-pointer based.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "backends/gpu/gpu_spmv.hpp" // available()
#include "backends/gpu/gpu_transfer.hpp"
#include "core/amg.hpp"
#include "core/formats.hpp"
#include "core/poisson.hpp"
#include "core/types.hpp"

int main(int argc, char** argv) {
    if (!spume::gpu::available()) {
        std::printf("gpu-transfer-check: no HIP device present -- skipping (exit 0)\n");
        return 0;
    }

    const spume::index_t n = (argc > 1) ? static_cast<spume::index_t>(std::atoi(argv[1])) : 64;
    const spume::Csr csr = spume::coo_to_csr(spume::gen::poisson7(n, n, n));
    const spume::Aggregation agg = spume::aggregate(csr);
    const auto nf = static_cast<std::size_t>(agg.agg.size());
    const auto ncrs = static_cast<std::size_t>(agg.ncoarse);
    std::printf("gpu-transfer-check: poisson7 %d^3  nfine=%zu  ncoarse=%zu\n", n, nf, ncrs);

    // Managed buffers (host+device visible).
    double* res = nullptr;
    double* rc = nullptr;
    double* ec = nullptr;
    double* z = nullptr;
    static_cast<void>(hipMallocManaged(&res, nf * sizeof(double)));
    static_cast<void>(hipMallocManaged(&rc, ncrs * sizeof(double)));
    static_cast<void>(hipMallocManaged(&ec, ncrs * sizeof(double)));
    static_cast<void>(hipMallocManaged(&z, nf * sizeof(double)));

    std::vector<double> z_init(nf);
    for (std::size_t i = 0; i < nf; ++i) {
        res[i] = std::sin(0.001 * static_cast<double>(i)) - 0.3;
        z_init[i] = 0.05 * static_cast<double>((i % 7)) - 0.1;
        z[i] = z_init[i];
    }
    for (std::size_t j = 0; j < ncrs; ++j) {
        ec[j] = 0.5 - 0.01 * static_cast<double>(j % 11);
    }

    // CPU reference.
    std::vector<double> rc_ref(ncrs, 0.0);
    for (std::size_t i = 0; i < nf; ++i) {
        rc_ref[static_cast<std::size_t>(agg.agg[i])] += res[i];
    }
    std::vector<double> z_ref(z_init);
    for (std::size_t i = 0; i < nf; ++i) {
        z_ref[i] += ec[static_cast<std::size_t>(agg.agg[i])];
    }

    // GPU.
    spume::gpu::AggTransferResident t(agg);
    t.restrict_device(res, rc);
    t.prolong_add_device(ec, z);
    static_cast<void>(hipDeviceSynchronize());

    // Compare: restrict in-class, prolong bitwise.
    double r_maxrel = 0.0;
    for (std::size_t j = 0; j < ncrs; ++j) {
        r_maxrel = std::max(r_maxrel, std::fabs(rc[j] - rc_ref[j]) / (std::fabs(rc_ref[j]) + 1e-300));
    }
    std::int64_t z_mismatch = 0;
    for (std::size_t i = 0; i < nf; ++i) {
        std::uint64_t a = 0;
        std::uint64_t b = 0;
        std::memcpy(&a, &z[i], 8);
        std::memcpy(&b, &z_ref[i], 8);
        if (a != b) {
            ++z_mismatch;
        }
    }
    const bool pass = (r_maxrel < 1e-12) && (z_mismatch == 0);
    std::printf("  VERIFY: restrict max_rel=%.3e (in-class)  prolong bit_mismatch=%lld/%zu -> %s\n",
                r_maxrel, static_cast<long long>(z_mismatch), nf, pass ? "PASS" : "FAIL");

    static_cast<void>(hipFree(res));
    static_cast<void>(hipFree(rc));
    static_cast<void>(hipFree(ec));
    static_cast<void>(hipFree(z));
    std::printf("  RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
