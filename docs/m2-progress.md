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

## M2 ANSWER (arc closed 2026-07-19, pitzBig 1.48M cells, isolated benchmark)

**Convergence: SPUME wins.** amgFP32 + K-cycle does ~½ the pressure iterations
of stock GAMG (25 vs 50 avg) and is far more stable (stock swings 17->79; SPUME
holds 22-29). The K-cycle turned a start-of-M2 5-10x deficit into a 2x lead.

**Wall-time: parity.** Isolated per-timestep solve cost (marginal method, fixed
startup removed): stock GAMG (serial) 13.64s vs SPUME amgFP32+K (16 threads)
12.89s = **0.95x, SPUME 5% faster**. Caveats: (1) 16-thread vs 1-core — at
matched parallelism (stock N-rank MPI) stock may lead; (2) SPUME's 2x iteration
lead is eaten by per-iteration K-cycle fan-out cost + a ~985ms/solve setup
rebuild. So SPUME has *caught* GAMG (better convergence, wall-time parity) but
not decisively won on wall-clock.

**Why not a wall-time win, and the CPU ceiling:** measured DRAM ceiling is
CPU-fabric-limited to ~150 GB/s (59% of the 256 GB/s LPDDR5X theoretical; NT
stores recover a 1.33x write-allocate factor; SMT halves it — pin 16 cores).
The iGPU reaches far more of the 256 GB/s. So the CPU is bandwidth-capped and
kernel micro-opts (NT-store measured 1.04x on read-dominated SpMV; block-SpMM
needs multiple RHS; s-step is complex) are marginal. The decisive bandwidth win
is M3/GPU.

**Amortization landed:** aggregation structure cached, operators rebuilt fresh
each solve (setup 1575->985ms), convergence stable across timesteps. Remaining
CPU lever, MEASURED and DEFERRED: the Galerkin coarse-operator build is 83% of
the ~985ms setup (coo_to_csr sort); caching its sparsity pattern (scatter-add
values, skip sort) -> ~350ms setup -> ~16% wall-time lead. Ready to implement,
but deferred as low-value on a fabric-walled CPU vs the M3 bandwidth opportunity.

**-> Moving to M3 (GPU/NPU).** Everything M2 proved feeds it: the FP32 firewall
(convergence-neutral), the K-cycle (Chebyshev-smoothed = GPU-mappable, no
Gauss-Seidel), coefficient-only updates (matches M3's zero-copy design), and the
2x bandwidth headroom on the iGPU.

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

1. **[LANDED] K-cycle — SPUME's AMG now reaches GAMG-parity convergence.**
   Implemented the bounded Notay K-cycle (`spumeKcycle`, default on). Measured on
   pitzBig amgFP32: **172 -> 28 pressure iters, solve 14.5s -> 5.4s (2.7x)** —
   GAMG's iteration range, with the FP32 firewall intact. Two bounds make each
   apply cheap: skip the finest level (the outer flexible-CG already accelerates
   it) and cap the Krylov depth to the top ~5 coarse levels (`spumeKcycleLevels`),
   so the fan-out is a constant, not 2^numLevels. Sweep (deterministic iters):
   V=172, K-L3=52, **K-L5=28**, K-L8=26 — L5 is the knee. Note: use amgFP32
   (self-coarsening, ~13 levels), NOT gamgFP32 (OpenFOAM's 20-level single-pairwise
   hierarchy makes the K-cycle fan-out too deep). Historical context below.

   SPUME's AMG took ~118-190 iters where GAMG takes ~15 on real graded meshes.
   This
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
   synthetic graded operator. So the gap is **structural, not tunable**.

   **ROOT CAUSE FOUND — it is the CYCLE, not the smoother.** The V-cycle uses
   unsmoothed (piecewise-constant) aggregation transfers; a plain V-cycle over
   that is the textbook-weak case, which is why no smoother rescued it (GS ruled
   out too: 115-169 iters, never near 15). Prototyped V vs W vs K cycle over the
   same hierarchy (graded operator, scratchpad/kcycle_probe.cpp):

   | grading | V-cycle | W-cycle | K-cycle |
   |---|---|---|---|
   | cz 1..100  | 40 | 14 | **12** |
   | cz 1..1000 | 72 | 19 | **12** |

   The **K-cycle collapses iterations to ~12 and is grading-independent** —
   GAMG-parity (15). It KEEPS the bandwidth-friendly Chebyshev smoother, so it
   is GPU/SIMD-neutral (no ADR tension). This is the win path: **K-cycle +
   Chebyshev + FP32 firewall = GAMG-parity iters x 1.34x bandwidth**. Caveat: a
   naive K-cycle recurses ~2-3x per level (exponential in levels); the
   production form is Notay's AGMG K-cycle with a residual-reduction test (2nd
   inner iteration only when ||r1|| > ~0.25 ||r0||), keeping it near-linear.
   (Cheap tuning knobs also landed: chebyshevSteps/chebyshevEta/amgCoarseTol.)
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
