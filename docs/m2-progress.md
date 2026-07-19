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
2. **SPUME's multigrid is not yet competitive with GAMG (THE M2 blocker).**
   Iteration counts on pitzBig at matched tol (1e-8, relTol 0.01), each
   preconditioning an outer Krylov:

   | preconditioner | pressure iters |
   |---|---|
   | **stock GAMG (DIC-GaussSeidel)** | **11 / 27 / 11** |
   | SPUME gamgFP32 (reuse OF hierarchy + Chebyshev) | 118 / 119 / 122 |
   | SPUME amgFP32 (self-coarsening + Chebyshev) | 172 / 174 / 187 |

   pitzBig is NOT just hard — stock GAMG solves it in ~15 iters. SPUME's AMG is
   **~5-10x weaker** on this real graded mesh, and that iteration deficit dwarfs
   the 1.34x FP32 win (net, SPUME loses to GAMG on wall time even at 16 threads
   vs stock's 1). The gap is **precision-independent** (FP64 needs the same
   counts). Diagnostic: on a *synthetic uniform-anisotropy* operator
   (cz/cx=100) SPUME's self-coarsening Chebyshev AMG converges in ~16 iters —
   competitive. It is the *real graded* mesh both SPUME paths fail on and GAMG
   handles: the smoother is sound in principle, but the coarsening + Chebyshev
   pairing does not match GAMG's faceAreaPair + DIC-GaussSeidel on real CFD
   operators. This sits in tension with M3: Chebyshev was chosen *because*
   Gauss-Seidel does not map to the GPU, yet DIC-GaussSeidel is what wins here.

**Confound retired:** never compare spumePCG(16 threads) wall-time vs stock
pimpleFoam pressure (single-thread — OpenFOAM scales by MPI, not threads); that
measures core count, not precision. A vs-OpenFOAM headline needs matched
parallelism (stock N MPI ranks vs spume N threads). The earlier large-case
"3.56x" was the GAMG-as-solver maxIter stall, not a bandwidth win — retired.

## Open threads (don't lose these)

1. **[MAIN] Close the preconditioner-quality gap** — SPUME's AMG takes ~118-190
   iters where GAMG takes ~15 on real graded meshes (measured above). This
   precision-independent 5-10x deficit is THE M2 blocker; the 1.34x FP32 win is
   real but second-order until it closes.

   **Cheap cycle tweaks MEASURED and RULED OUT** (pitzBig gamgFP32, 1st solve):

   | config | iters | solve time |
   |---|---|---|
   | baseline (steps 5) | 117 | **10.5 s** |
   | steps 10 | 95 | 14.6 s |
   | steps 20 | 76 | 20.8 s |
   | steps 10, eta 100, coarse 1e-3 | 83 | 13.3 s |

   More smoothing cuts iterations ~35% but every config is *slower* on wall time
   — the extra per-iteration Chebyshev SpMVs (89 -> 274 ms/iter at steps 20)
   outweigh the iteration savings. Tighter coarse-solve had zero effect on a
   synthetic graded operator. So the gap is **structural, not tunable**: it needs
   better coarsening or a different smoother, both a design call in tension with
   the M3 GPU goal (needs an ADR):
   - a DIC/ILU(0)-style smoother to match GAMG — strongest, but Gauss-Seidel-like
     and hard to map to the GPU (the reason Chebyshev was chosen);
   - better coarsening (smoothed aggregation / match faceAreaPair) with Chebyshev
     kept — GPU-friendly, unproven on graded meshes.
   (Knobs are now dictionary-tunable: chebyshevSteps/chebyshevEta/amgCoarseTol.)
2. **vs-OpenFOAM headline at matched parallelism** — stock N-rank MPI GAMG vs
   spume N-thread gamgFP32, same core budget (only meaningful once #1 closes).
3. **Productionize NT stores** as a `src/backends/` SpMV kernel (checkasm +
   dispatch, ~7% measured). First real hand-AVX-512; wire into the AMG residual.
4. **Amortize per-solve V-cycle setup** — ~1 s/solve rebuilding Galerkin
   operators + smoothers each solve (the agglomeration is already cached).
5. **RCM per AMG level** — coarse levels are the most scattered.
6. **M3:** GPU-resident V-cycle — the Chebyshev smoother maps to the GPU;
   GAMG's Gauss-Seidel structurally cannot. (The FP32-firewall result de-risks
   this: reduced precision in the smoother is now proven convergence-neutral.)
