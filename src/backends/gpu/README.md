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
- `gpu_cheb.{hpp,hip.cpp}` — the FP32 Chebyshev smoother (`ChebyshevDeviceFP32`):
  the equilibrated operator + Saad recurrence run in FP32 on-device (the ADR-0002
  preconditioner interior), residual in / correction out stay FP64 (the firewall).
  Six kernels driving the same recurrence as the CPU `ChebyshevPrecond<float>`.
- `*_check.cpp` — verify-then-bench gates (ADR-0016 discipline): build a poisson7
  operator, run the kernel on the GPU, VERIFY against the CPU reference within the
  reorder-tolerance class (ADR-0017), then MEASURE achieved GB/s. Registered as
  the `gpu-spmv-check` / `gpu-cheb-check` ctests; skip (exit 0) with no GPU, so
  they are no-ops on CPU-only machines.

## Build & run

```
cmake --preset gpu && cmake --build --preset gpu
ctest --preset gpu                       # runs gpu-spmv-check + the CPU suite
./build/gpu/src/backends/gpu/spume-gpu-spmv-check 128
```

Requires ROCm/HIP (tested with HIP 7.2). The default build (`SPUME_ENABLE_HIP`
OFF) never touches this directory.

## Measured (gfx1151, poisson7 128³, 2.1M rows)

- **FP64 SELL SpMV: ~207 GB/s (81% of the 256 GB/s LPDDR5X peak), bitwise-exact
  vs the CPU reference, ~8.3× a single-core CPU reference SpMV.**
- **FP32 Chebyshev smoother (5 steps): ~219 GB/s (85% of peak), in-class vs the
  CPU `ChebyshevPrecond<float>` (max_abs/‖z‖∞ 4.6e-7, L2-rel 1.7e-7 — at the
  ADR-0017 FP32 bar of 2.7e-7).**

Bandwidth is the documented traffic model over HIP-event kernel time (ADR-0013
honest-degradation methodology), which structurally cannot overstate achieved
bandwidth (the model undercounts gather traffic). rocprof `--pmc` hardware
counters would give measured bytes directly but hang on gfx1151 (see above).

## Profiling / instrumentation

Install the ROCm profiling layer (rocprofv3, rocprof-compute, rocm-bandwidth-test,
rocprof-sys, rocgdb, amd-smi) reproducibly:

```
scripts/install_gpu_tooling.sh            # install + verify
scripts/install_gpu_tooling.sh --verify   # check an existing install
```

Kernel trace (per-dispatch durations) of the controlled-dispatch driver:

```
rocprofv3 --kernel-trace --output-format csv -d out -- \
    ./build/gpu/src/backends/gpu/spume-gpu-spmv-pmc 128 8
```

**Hardware counters (`rocprofv3 --pmc`) hang on this gfx1151 APU** even for a
single dispatch/single counter -- a known PMC limitation on consumer RDNA3.5
(vs CDNA/Instinct). So achieved bandwidth is the documented traffic model over
kernel time (ADR-0013), which structurally cannot overstate (the model
undercounts gather traffic); kernel-trace supplies the time. `rocm-bandwidth-test`
(v2.6+ uses a `run`/`plugin` subcommand CLI) gives a measured DRAM roofline to
replace the 256 GB/s theoretical denominator. Revisit `--pmc` when ROCm/driver
support for gfx1151 counters improves.

## Next (M3 Phase 3)

SpMV and the FP32 Chebyshev smoother are landed and verified. Next: the FP32
V-cycle / K-cycle over a resident hierarchy (reusing these kernels), then the
whole-solve GPU-resident FCG (fuse the CG reductions, drop per-iteration sync),
and a measured cell-count fallback to the CPU path. Prototypes for the full
solve are in `~/spume-m3-gpu-prototypes/`.
