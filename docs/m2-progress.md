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

## Side quests (done, with outcomes)

| tangent | outcome |
|---|---|
| perf counters (perf_event_paranoid=4, no uProf) | use timing + triad prediction-error; owner can unlock paranoid via passwordless sudo |
| access-pattern investigation | **KEY:** scattered SpMV is *latency-bound* (TLB/prefetch), not bandwidth-bound -> motivated RCM. See memory. |
| GAMG source analysis | the "reuse the trunk, own the kernels" pivot -> the parity result |
| NT-store AVX-512 experiment | **validated ~7%** on FP64 write-once SpMV; **not yet productionized** (open thread 2) |
| ryzenadj | out of scope — ratios are drift-immune |

## Open threads (don't lose these)

1. **[MAIN] Measure `gamgFP32` on a large (>5M-cell) case.** The M2 DoD, and
   the only place the FP32 *bandwidth* win can show — pitzDaily (12k, partly
   cache-resident) is at parity but can't show it. This is the "did we beat
   GAMG" answer.
2. **Productionize NT stores** as a `src/backends/` SpMV kernel (checkasm +
   dispatch, ~7% measured). First real hand-AVX-512; wire into the AMG residual.
3. **Amortize per-solve V-cycle setup** — the agglomeration is cached, but the
   Galerkin operators + smoothers are still rebuilt each solve.
4. **RCM per AMG level** — coarse levels are the most scattered.
5. **M3:** GPU-resident V-cycle — the Chebyshev smoother maps to the GPU;
   GAMG's Gauss-Seidel structurally cannot.
