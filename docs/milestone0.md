# Milestone 0 — mixed-precision core: results

**Claim under test.** A flexible FP64 outer Krylov solver with an FP32
preconditioner converges to FP64-accurate solutions at materially lower
memory traffic.

**Verdict: demonstrated.** With everything else held fixed, switching the
Chebyshev preconditioner interior from FP64 to FP32 leaves the outer
iteration count and the converged residual *identical* (178 iterations,
relres 8.7e-11) while cutting modeled traffic by 32% and wall time by 31%
— and the wall-time ratio (0.693) matches the modeled byte ratio (0.680),
i.e. the win is bandwidth, not compute.

## Machine and toolchain

| | |
|---|---|
| CPU | Intel Xeon @ 2.10 GHz (virtualized; `-march=native` resolves to `sapphirerapids`), 4 cores / 4 threads |
| Caches (as exposed at measurement time) | L1d 4×32 KiB, L2 4×1 MiB, L3 33 MiB shared |
| Memory | 15 GiB |
| OS / kernel | Ubuntu 24.04, Linux 6.18.5 |
| Compiler | GCC 13.3.0, `-O3 -march=native`, OpenMP, 4 threads, `OMP_PROC_BIND=close OMP_PLACES=cores` |
| Build | `cmake --preset cpu-bench` @ commit of this doc |
| Date | 2026-07-12 |

> **Note:** this container is *not* the primary dev target. The rooflines
> that govern SPUME's design live in `docs/hardware.md` (Strix Halo:
> ~124 GB/s CPU-aggregate, ~212 GB/s GPU-reachable); the only roofline
> valid for the numbers below is the STREAM triad measured in the same
> invocation. Lab-grade numbers (fixed power mode, hardware counters)
> arrive with the self-hosted Strix Halo bench runner — see ADR-0013.

## Methodology (ADR-0013)

Warm-ups precede every timed region; no I/O inside timers. All "bytes
moved" figures are the documented traffic model (`sell.hpp` for SpMV,
per-iteration counts in `bench/main.cpp` for solvers). Because this is a
shared VM without counters or power control, the harness quantifies its
own noise: per-section steal-time bracketing, median + CV over reps
(flagged above 5%), an LLC working-set guard, FP64/FP32 measured with
interleaved reps (drift-immune ratios), and a triad-calibrated prediction
error as the counter substitute.

**This machine moves under you — measured, not suspected.** Between two
sessions one day apart, the exposed L3 changed from 260 MiB to 33 MiB and
STREAM triad moved from 42.7 to ~38-41 GB/s; steal time stayed ~0% while
rep-to-rep CV reached 6-9% in some sections. Consequently: only
same-invocation comparisons are quoted below, ratios are the primary
evidence, and absolute GB/s carry their CV.

Reading "% of triad" honestly: STREAM counting omits write-allocate, so
triad's true bus traffic is 4/3 of what its GB/s figure counts. A ~97%-read
kernel like SpMV can therefore legitimately report slightly *above* 100% of
"triad" — the roofline for read-dominated streams is higher than triad's
counted number.

## Measured roofline (STREAM-style probe, 3 × 256 MiB, 10 reps, steal 0.00%)

| kernel | GB/s (best) | CV |
|---|---|---|
| copy | 37.5 | 0.7% |
| scale | 35.3 | 1.1% |
| add | 41.0 | 2.6% |
| **triad (roofline reference)** | **39.9** | 2.7% |

## SpMV — SELL-C-8, Poisson 192³ (n = 7,077,888, nnz = 49,324,032, padding 1.001)

20 interleaved reps; same-invocation triad 40.8 GB/s; steal 0.00%.
Working sets: fp64 706 MB = 20× LLC, fp32 452 MB = 13× LLC.

| precision | model bytes/SpMV | median time | achieved GB/s (best) | CV |
|---|---|---|---|---|
| FP64 | 706.0 MB | 17.51 ms | 40.7 (≈100% of triad) | 0.5% |
| FP32 (equilibrated) | 451.8 MB | 12.52 ms | 36.5 (89% of triad) | 6.7% |

Median time ratio 0.715 against a model byte ratio 0.640: FP32 SpMV
realizes most but not all of the byte saving on this host (0.70-0.81
across today's invocations; a previous session on a different backing
host measured 0.639, i.e. the full byte ratio). The shortfall pattern —
FP64 pinned at the read roofline with 0.5% CV, FP32 jittery — is
consistent with the FP32 kernel being slightly gather-limited rather than
purely bandwidth-limited at 4 threads, and with marginal LLC residency of
its 28 MB x-vector (vs 33 MiB LLC). The solver-level result below, which
is the claim that matters, is unaffected.

## Solvers — Poisson 192³, tol 1e-10, random RHS (same invocation, triad 39.9 GB/s)

`cg64` = plain FP64 CG (reference). `fcg-cheb64/32` = flexible PCG, FP64
outer, Chebyshev(steps 5, eta 30) on the equilibrated system in FP64/FP32.
True residuals recomputed outside the timer. Single-shot runs; steal
≤ 0.01% on all three.

| solver | iterations | wall time | modeled traffic | effective GB/s | pred err vs triad | final relres |
|---|---|---|---|---|---|---|
| cg64 | 790 | 26.75 s | 1094 GB | 40.9 | −2.4% | 9.8e-11 |
| fcg-cheb64 | 178 | 37.78 s | 1243 GB | 32.9 | +21.4% | 8.7e-11 |
| **fcg-cheb32** | **178** | **26.18 s** | **845 GB** | 32.3 | +23.7% | **8.7e-11** |

The controlled comparison is cheb64 → cheb32 (only the preconditioner
precision changes):

- outer iterations: **178 → 178** (identical), final residual identical —
  FP64 accuracy untouched, as the flexible outer iteration guarantees;
- modeled traffic: 1243 → 845 GB (**−32%**);
- wall time: 37.78 → 26.18 s (**−31%**); time ratio 0.693 vs model byte
  ratio 0.680.

The triad-calibrated prediction errors are the model check: cg64 lands
within 2.4% of pure-streaming prediction; both FCG variants run ~21-24%
slower than pure streaming (gather-heavier inner work), but by the *same*
factor — so the precision ratio holds.

Notes: at this conditioning (κ ≈ 1.5e4) and eta = 30, Chebyshev-preconditioned
FCG roughly ties plain CG in wall time — polynomial preconditioning
mostly trades outer iterations for inner SpMVs. That is expected and not
the claim; the claim is the *precision* axis, which delivers within 2% of
its full modeled saving end-to-end. Stronger preconditioners (and
FP16/block-scaled interiors) are later milestones.

Correctness backing: `tests/regression/` asserts, on every CI run, that
mixed-precision solutions match pure FP64 within the theorem-backed
rounding-reordering bound and iteration counts within 20% — plus bitwise
reproducibility across 1/4/16 threads in deterministic-reduction mode.

## Reproduce

```sh
cmake --preset cpu-bench && cmake --build --preset cpu-bench
export OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores
./build/cpu-bench/bench/spume-bench all  --nx 192 --ny 192 --nz 192
./build/cpu-bench/bench/spume-bench spmv --nx 192 --ny 192 --nz 192 --reps 20
```

Trust a run only when its own output shows: steal ~0%, CV within flags,
and working sets safely above LLC (watch individual vectors, not just the
total). Pick sizes so even single vectors exceed your last-level cache.
