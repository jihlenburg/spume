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
- `gpu_transfer.{hpp,hip.cpp}` — the aggregation grid-transfer operators
  (`AggTransferResident`): restrict (`P^T`, FP64 atomic scatter-add) and prolong
  (`P`, conflict-free gather-add), device-to-device so the V-cycle chains without
  host round-trips.
- `gpu_vcycle.{hpp,hip.cpp}` — the assembled GPU-resident FP32 V-cycle
  (`VcycleDeviceFP32`): the resident hierarchy (per level: FP64 operator, FP32
  Chebyshev smoother, aggregation transfers) driven through smooth → fused
  residual → restrict → recurse → prolong → post-smooth, with the coarsest level
  solved on the CPU (opt #4). Mirrors the CPU `AmgPrecond<float>` V-cycle; one
  host sync per apply. Folds in the fused residual (opt #1).
- `gpu_reduce.{hpp,hip.cpp}` — the FP64 dot reduction (`DotDeviceFP64`): block
  tree + host partial-sum, for the Krylov scalars.
- `gpu_fcg.{hpp,hip.cpp}` — the **whole-solve GPU-resident flexible CG**
  (`FcgSolverGPU`), the M3 goal: `A x = b` solved entirely on the GPU, FP64 outer
  Krylov preconditioned by the resident FP32 V-cycle. Vectors stay device-
  resident across iterations; the only host traffic is b/x and the scalar dot
  results driving alpha/beta.
- `*_check.cpp` — verify-(then-bench) gates (ADR-0016 discipline): build a
  poisson7 operator, run the kernel on the GPU, VERIFY against the CPU reference
  within the reorder-tolerance class (ADR-0017), then MEASURE achieved GB/s.
  Registered as the `gpu-spmv-check` / `gpu-cheb-check` / `gpu-transfer-check`
  ctests; skip (exit 0) with no GPU, so they are no-ops on CPU-only machines.

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
- **FP32 V-cycle (15-level hierarchy, poisson7 96³): in-class vs the CPU
  `AmgPrecond<float>` (max_abs/‖z‖∞ 1.2e-7), 17 ms/cycle = 3.1× the CPU apply.**
  The 3.1× (vs 8× for raw SpMV) is the coarse-level tax: with 15 levels the small
  coarse levels are launch-latency-bound and the CPU coarse solve serialises the
  cycle — see opt #4 below.
- **FP64 dot reduction: 224 GB/s (87% of peak).**
- **Whole-solve GPU FCG (poisson7 96³, tol 1e-8) — the M3 goal:** solves to FP64
  accuracy on the GPU (**true residual 8e-9**), **identical convergence to the
  CPU (24 == 24 iterations)**, and the GPU/CPU solutions agree to **1.3e-14** —
  the empirical proof of the ADR-0002 firewall: the FP32 preconditioner yields a
  bit-class-identical FP64 answer. **5.6× the CPU solve** after the coarse-tax fix
  (opt #4): coarsening only to ~2500 rows instead of 200 (the CPU-default depth)
  cut the solve 1.7× — 15→5 levels, same iterations.

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

## Optimization opportunities (scouted)

The individual kernels already run at 81–85% of the memory roofline, so per the
performance policy the remaining wins are **moving fewer bytes**, not
instruction-level tuning of a near-roofline kernel. Prioritised for the V-cycle
assembly:

1. **Fused residual `res = r − A z`** (one kernel, one pass) instead of SpMV then
   a separate subtract — saves a full read+write of the intermediate `A z` vector
   per level. Highest-value byte reduction; do it as the V-cycle's residual step.
2. **Fully-FP32 V-cycle scaffolding.** The transfers/residual are FP64 here to
   match the CPU reference tightly; an all-FP32 cycle halves the operator,
   residual, and transfer traffic (the ADR-0002 firewall makes it convergence-
   safe). Measure the convergence-vs-bandwidth trade before committing.
3. **Resident chaining kills host copies.** The standalone `apply()`s stage
   x/y through the host each call; the V-cycle/FCG must keep vectors device-
   resident and use the device-to-device entry points (already provided). A
   whole-solve loop should sync once per outer iteration, not per kernel.
4. **Coarse-level tax — measured and (partly) fixed.** With the CPU-default
   coarsening (down to 200 rows, 15 levels) the small coarse levels are launch-
   latency-bound and the whole FCG solve ran at 3.4× the CPU. A `coarse_size`
   sweep (poisson7 96³) showed the deep tail adds **zero** convergence benefit:

   | coarse_size | levels | iters | solve ms |
   |---|---|---|---|
   | 200 | 16 | 24 | 432 |
   | 2500 | 5 | 24 | **257** |
   | 200000 | 3 | 18 | 775 |

   Stopping at ~2500 rows (CPU CG takes the coarsest) is **1.7× faster** with the
   same iterations → the whole solve went 3.4× → **5.6×**. The FCG check uses the
   tuned depth. Further levers not yet taken: overlap the CPU coarse solve with
   fine GPU work (the heterogeneous design), batch the remaining coarse kernels,
   and make the optimal `coarse_size` a mesh-size-aware heuristic.
5. **Chebyshev micro-fusions** (scale-in+init; final axpy+scale-out) shave a
   couple of vector round-trips — low priority (chases the last ~15% on an
   already-near-roofline kernel).
6. **SELL-C-σ row sorting (σ>1)** cuts padding/improves coalescing on irregular
   meshes; no benefit on the regular poisson7 test but relevant for real cases
   (`padding_ratio()` reports the overhead).

## Next (M3 Phase 3)

The whole-solve GPU-resident FCG is landed and verified — the M3 engine works.
Remaining, in leverage order:
1. **Attack the coarse-level tax** (opt #4) — the 3.4× is bounded by the coarse
   V-cycle overhead, so this is the highest-leverage win for the whole solve.
2. **Port the K-cycle** (the reduction now exists for its Krylov acceleration) so
   the GPU reaches the CPU's GAMG-parity iteration counts on hard graded meshes.
3. **Share the fine operator** — `FcgSolverGPU` currently builds it twice (once
   for `A p`, once as V-cycle level 0); reuse one resident copy (~200 MB saved on
   the 96³ case).
4. **Fuse the per-iteration reductions** (fewer syncs) and the **cell-count
   fallback** to the CPU path.
Prototypes for the full solve are in `~/spume-m3-gpu-prototypes/`.
