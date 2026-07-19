<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (C) 2026 Joern Ihlenburg -->

# M2 progress + side-quest tracker

A living map of the CPU-performance-path work so the tangents don't bury the
direction. The end goal never moves: bandwidth-first CFD, motorBike/DrivAer on
Strix Halo at 2.5-4x (M3). M2 is the CPU FP32-multigrid lever that gets us
there.

## Direction

M1 (done) -> **M2 (here): FP32-GAMG-under-FP64-Krylov** -> M3 (GPU flagship).
The single question M2 must answer: **does SPUME's mixed-precision multigrid
beat a tuned stock GAMG on a large case?**

## Landed (main)

- **Solver seam** — `spumePCG` (runtime-selectable `lduMatrix::solver`, zero
  core diff), LDU->SELL bridge, FP64 CG, coupled-interface fallback.
- **Mixed precision** — FP32 Chebyshev preconditioner (equilibrated), in-class.
- **Strix Halo baseline** — `docs/perf-strix-halo.md`: triad 114 GB/s, SpMV
  ~88% roofline, FP32/FP64 SpMV 0.626.
- **ADR-0015** local Strix Halo = perf reference (vs tuned znver5); **ADR-0016**
  inline asm permitted (FFmpeg/checkasm discipline).
- **Comparison harness** — `run_solver_comparison.sh` (SPUME vs stock GAMG, znver5).
- **Fused SpMV+AXPY** — ~8%, in the Chebyshev smoother.
- **RCM renumbering** — recovers 120% of scattered-access loss.
- **AMG engine** — aggregation + Galerkin, multi-level V-cycle, FP32 = FP64
  convergence.
- **spumeGAMG hybrid** — reuse OpenFOAM's cached GAMGAgglomeration + SPUME FP32
  V-cycle => **PARITY (1.007x) with stock GAMG** on pitzDaily, in-class.
- **OpenMP dispatch through the V-cycle** — `spumePCG` now defaults to
  `Dispatch::openmp` (was single-threaded `reference`; the SELL kernels ran on
  one core). Threaded through `AmgPrecond` (smoother, coarse CG, residual).
  `bench checkasm` machine-verifies openmp SpMV is **bitwise-identical** to the
  reference (invariant #4) before it is timed. Opt-out: `spumeThreaded false`.
- **Instrumentation baked in** — per-phase setup/solve/iters timers in
  `spumePCG` (`log 1`); `bench/perf_counters.hpp` (perf_event_open: cycles,
  instructions, LLC misses, dTLB misses, raw AMD DRAM-fill codes; degrades
  honestly to the timing model when paranoid > 2); `bench/checkasm.hpp`
  (FFmpeg-style verify-then-bench, exact + within-tolerance classes), both
  wired into `spume-bench checkasm`.

## Side quests (done, with outcomes)

| tangent | outcome |
|---|---|
| perf counters (perf_event_paranoid=4, no uProf) | use timing + triad prediction-error; owner can unlock paranoid via passwordless sudo |
| access-pattern investigation | **KEY:** scattered SpMV is *latency-bound* (TLB/prefetch), not bandwidth-bound -> motivated RCM. See memory. |
| GAMG source analysis | the "reuse the trunk, own the kernels" pivot -> the parity result |
| NT-store AVX-512 experiment | **validated ~7%** on FP64 write-once SpMV; **not yet productionized** (open thread 2) |
| ryzenadj | out of scope — ratios are drift-immune |

## The honest M2 answer (measured 2026-07-19, pitzBig 1.48M cells, 16 threads)

Pure precision isolation — `spumePCG` gamgFP32 vs gamgFP64, same solver, same
threaded dispatch, same reused GAMGAgglomeration, same FP64 outer flexible-CG,
ONLY float vs double inside the V-cycle (`scratchpad/precision_isolation.sh`,
per-phase `log 1` timers):

| metric | gamgFP32 | gamgFP64 |
|---|---|---|
| pressure iterations | 118/119/122/125 | **118/119/122/125 (identical)** |
| solve phase / solve | ~10.3–11.1 s | ~13.8–14.7 s (**1.34x**) |
| setup / solve | ~1.0 s | ~1.0 s (equal — FP64 assembly both) |
| per-iteration | ~88 ms | ~118 ms |
| whole-app wall time | **77.8 s** | 92.2 s (**1.19x**) |
| final p field | — | FP32 == FP64, **in reorder-tolerance class** |

**Two findings, both honest:**
1. **The mixed-precision firewall works (ADR-0002 validated).** FP32 V-cycle
   gives *identical* convergence to FP64 — the FP32 preconditioner under the
   FP64 Krylov does not cost a single extra iteration. And FP32 solve is
   **~1.34x faster** at identical iterations: the pure bandwidth win, no
   thread-count or algorithm confound. Consistent with the 0.63 FP32/FP64 SpMV
   byte ratio diluted by the FP64 outer work.
2. **The V-cycle is a weak preconditioner (the bigger lever).** 118 iterations
   where a tuned GAMG converges in ~10-20. This is **precision-independent**
   (FP64 needs the same 118): a point-Chebyshev smoother on the high-aspect
   anisotropic pitzBig mesh, exactly where GAMG's DIC-GaussSeidel wins.
   Cutting iterations (line/ILU smoother, W-cycle) multiplies *on top of* the
   1.34x and dwarfs any per-iteration byte tuning.

**Confound retired:** never compare spumePCG(16 threads) wall-time vs stock
pimpleFoam pressure (single-thread — OpenFOAM scales by MPI, not threads); that
measures core count, not precision. A vs-OpenFOAM headline needs matched
parallelism (stock N MPI ranks vs spume N threads). The earlier large-case
"3.56x" was the GAMG-as-solver maxIter stall, not a bandwidth win — retired.

## Open threads (don't lose these)

1. **[MAIN] Strengthen the V-cycle smoother** — the named bottleneck: 118 iters
   is precision-independent weakness on anisotropic meshes. A line-implicit or
   ILU(0) smoother (GAMG-parity) to reach ~15 iters is the largest remaining M2
   lever, and it compounds with the 1.36x FP32 win.
2. **vs-OpenFOAM headline at matched parallelism** — stock N-rank MPI GAMG vs
   spume N-thread gamgFP32, same core budget, so the number is honest.
3. **Productionize NT stores** as a `src/backends/` SpMV kernel (checkasm +
   dispatch, ~7% measured). First real hand-AVX-512; wire into the AMG residual.
4. **Amortize per-solve V-cycle setup** — ~1 s/solve rebuilding Galerkin
   operators + smoothers each solve (the agglomeration is already cached).
5. **RCM per AMG level** — coarse levels are the most scattered.
6. **M3:** GPU-resident V-cycle — the Chebyshev smoother maps to the GPU;
   GAMG's Gauss-Seidel structurally cannot. (The FP32-firewall result de-risks
   this: reduced precision in the smoother is now proven convergence-neutral.)
