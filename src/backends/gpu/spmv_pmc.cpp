// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// Minimal controlled-dispatch driver for profiling the GPU SELL SpMV: it issues
// only a few dispatches (no 50-rep bench loop), so a profiler can attribute
// cleanly. Intended for `rocprofv3 --kernel-trace` (per-dispatch durations),
// which works on gfx1151.
//
// NOTE: `rocprofv3 --pmc` hardware-counter collection (FETCH_SIZE/WRITE_SIZE ->
// true DRAM bytes) HANGS on this gfx1151 APU even for a single dispatch/counter
// -- a known limitation of PMC on consumer RDNA3.5 (vs CDNA/Instinct). Until
// that is resolved, achieved bandwidth is the documented traffic model over
// kernel-trace time (ADR-0013), which structurally cannot overstate. Not a
// ctest; run under a profiler by hand (see README).

#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

#include "backends/gpu/gpu_spmv.hpp"
#include "core/formats.hpp"
#include "core/poisson.hpp"
#include "core/sell.hpp"

int main(int argc, char** argv) {
    if (!spume::gpu::available()) {
        std::printf("spmv-pmc: no HIP device -- nothing to profile\n");
        return 0;
    }
    const spume::index_t n = (argc > 1) ? static_cast<spume::index_t>(std::atoi(argv[1])) : 128;
    const int launches = (argc > 2) ? std::atoi(argv[2]) : 8;

    const spume::Sell<double> a =
        spume::sell_from_csr(spume::coo_to_csr(spume::gen::poisson7(n, n, n)));
    const auto nr = static_cast<std::size_t>(a.nrows);
    const auto nc = static_cast<std::size_t>(a.ncols);
    std::printf("spmv-pmc: poisson7 %d^3  nrows=%d  model bytes/SpMV = %.0f (%.1f MB)\n",
                n, a.nrows, a.spmv_bytes(), a.spmv_bytes() / 1e6);

    std::vector<double> x(nc, 1.0);
    std::vector<double> y(nr, 0.0);
    spume::gpu::SellDeviceFP64 dev(a);
    for (int i = 0; i < launches; ++i) {
        dev.spmv(x, y); // one kernel dispatch each; resident operator, prefetched
    }
    std::printf("spmv-pmc: %d dispatches done (compare rocprofv3 FETCH_SIZE+WRITE_SIZE"
                " per dispatch to model bytes above)\n",
                launches);
    return 0;
}
