<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (C) 2026 Joern Ihlenburg -->

# src/backends/gpu — HIP backend (M3, ADR-0017)

GPU kernels for the M3 flagship: HIP on **gfx1151** (Radeon 8060S / Strix Halo,
RDNA3.5), **unified-memory resident** (`hipMallocManaged` / GTT zero-copy — never
`hipMalloc`, which segfaults above the small carve-out on this APU). Reduced
precision is confined to preconditioners under the FP64 outer Krylov (ADR-0002);
the portable CPU reference stays the default (ADR-0004) — this backend is opt-in.

## Contents

- `gpu_spmv.hpp` — HIP-free host API (`spume::gpu::available()`,
  `SellDeviceFP64`), so the host-compiled core can include it.
- `gpu_spmv.hip.cpp` — the SELL-C-8 FP64 SpMV kernel + resident-operator
  implementation (compiled with hipcc). One thread per row, fixed j-order
  accumulation matching the CPU reference.
- `spmv_check.cpp` — verify-then-bench gate (ADR-0016 discipline): builds a
  poisson7 operator, runs it on the GPU, VERIFIES against `spume::spmv`
  (reference) within the reorder-tolerance class (ADR-0017), then MEASURES
  achieved GB/s. Registered as the `gpu-spmv-check` ctest; skips (exit 0) with no
  GPU, so it is a no-op on CPU-only machines.

## Build & run

```
cmake --preset gpu && cmake --build --preset gpu
ctest --preset gpu                       # runs gpu-spmv-check + the CPU suite
./build/gpu/src/backends/gpu/spume-gpu-spmv-check 128
```

Requires ROCm/HIP (tested with HIP 7.2). The default build (`SPUME_ENABLE_HIP`
OFF) never touches this directory.

## Measured (gfx1151, poisson7 128³, 2.1M rows)

FP64 SELL SpMV: **~207 GB/s (81% of the 256 GB/s LPDDR5X peak), bitwise-exact vs
the CPU reference, ~8.3× a single-core CPU reference SpMV.** rocprof is not
installed on the dev box; bandwidth is the documented traffic model over
HIP-event kernel time (ADR-0013 honest-degradation methodology), which
structurally cannot overstate achieved bandwidth (the model undercounts gather
traffic).

## Profiling / instrumentation

Install the ROCm profiling layer (rocprofv3, rocprof-compute, rocm-bandwidth-test,
rocprof-sys, rocgdb, amd-smi) reproducibly:

```
scripts/install_gpu_tooling.sh            # install + verify
scripts/install_gpu_tooling.sh --verify   # check an existing install
```

Kernel trace of the SpMV check (durations per launch):

```
rocprofv3 --kernel-trace --output-format csv -d out -- \
    ./build/gpu/src/backends/gpu/spume-gpu-spmv-check 128
```

`rocprofv3 --pmc FETCH_SIZE WRITE_SIZE` gives hardware memory-traffic counters
(true achieved bytes vs the traffic model) but serialises every dispatch, so
point it at a single-launch driver, not the 50-rep bench loop. `rocm-bandwidth-test`
(v2.6+ uses a `run`/`plugin` subcommand CLI) gives the measured DRAM roofline to
replace the 256 GB/s theoretical denominator. rocprof counter analysis is the
next profiling pass (see roadmap M3 Phase 3).

## Next (M3 Phase 3)

Port the FP32 Chebyshev smoother and the K-cycle (prototypes in
`~/spume-m3-gpu-prototypes/`), keep the whole solve GPU-resident, wire a
cell-count fallback to the CPU path, and add rocprof roofline evidence once the
profiler is available.
