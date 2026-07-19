<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (C) 2026 Joern Ihlenburg -->

# Performance baseline — Strix Halo (target hardware)

The first SPUME bench numbers measured on the **primary dev target** rather than
a cloud VM. Per ADR-0013, self-hosted Strix Halo results supersede the
container numbers in `docs/milestone0.md`; the ratios below are the authoritative
mixed-precision evidence, absolute GB/s carry the caveats noted.

## Provenance

- **Host:** `halobox` — AMD Ryzen AI Max+ 395 (Strix Halo), 16 Zen 5 cores /
  32 threads, 2 CCDs, LPDDR5X-8000 256-bit.
- **Kernel:** 7.0.0-28-generic. **Compiler:** GCC 15, `cpu-bench` preset
  (`-O3`, portable — no `-march=native`). **Date:** 2026-07-19.
- **Thread config:** `OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores`
  — one thread per physical core.
- **Method (ADR-0013):** interleaved A/B reps for the FP32/FP64 ratio
  (drift-immune), same-invocation STREAM triad as the only valid roofline,
  per-section steal-time bracketing, median + CV over reps (flag >5%).

### Caveats (why these are "non-lab", not "lab-grade")

- **Package power mode not fixed** (Strix Halo is 45–120 W configurable). This
  moves absolute GB/s; the **FP32/FP64 ratios are drift-immune** and are the
  primary claim. No `ryzenadj`/BIOS power pinning was applied.
- No hardware counters (no uProf/rocprof; `perf` uncore is locked at
  `perf_event_paranoid=4`). The triad-calibrated prediction error is the
  counter substitute.
- A desktop session (`gnome-shell`) was resident. Core pinning kept it off the
  bench cores: steal stayed 0.00% and CV < 1% on the pinned runs.

### Methodology lesson (recorded so it is not re-learned)

The **first** run used the default 32 OpenMP threads (SMT) and small working
sets: triad 79 GB/s, SpMV FP32 showed *no* win (time ratio 0.958), and the
solve section was contaminated (+1517% prediction error). Pinning to the **16
physical cores** and using a bandwidth-bound working set (≥13× LLC) fixed it
entirely — SMT oversubscription and cache-marginal sizes were the artefacts,
not the hardware. Bandwidth benchmarks on this box must pin to physical cores.

## Measured roofline — STREAM (3 × 256 MiB, pinned 16 cores)

| kernel | GB/s | CV |
|---|---:|---:|
| copy | 116.7 | 1.3% |
| scale | 114.7 | 1.5% |
| add | 111.6 | 1.4% |
| **triad (roofline reference)** | **114.1** | **0.5%** |

Steal 0.00%. Consistent with `docs/hardware.md` (~124 GB/s CPU-aggregate;
triad's 3-access count omits write-allocate, so true bus traffic is ~4/3 of the
figure). ~2.85× the cloud-VM triad (39.9 GB/s).

## SpMV — SELL-C-8, Poisson 192³ (n = 7,077,888, nnz = 49.3 M, padding 1.001)

10 interleaved reps; same-invocation triad 114 GB/s; steal 0.00%.

| precision | model bytes/SpMV | median time | achieved GB/s | % of triad | CV |
|---|---:|---:|---:|---:|---:|
| FP64 | 706.0 MB | 7.157 ms | 99.1 | 86.9% | 0.2% |
| FP32 (equilibrated) | 451.8 MB | 4.481 ms | 101.2 | 88.7% | 0.6% |

**FP32/FP64 time ratio 0.626 vs model byte ratio 0.640** — the mixed-precision
bandwidth win realised to the model, both kernels at ~88% of roofline (i.e. the
SpMV is bandwidth-saturated: the performance policy's "reach the roofline" is
met, and FP32 "lowers the roofline" by moving 0.64× the bytes).

## Solve — flexible CG, Poisson 128³ (n = 2,097,152, tol 1e-10)

Same-invocation triad 111 GB/s; single-shot (no CV), ratio reproduced across
two invocations (0.660, 0.661).

| solver | iters | time | model bytes | relres |
|---|---:|---:|---:|---:|
| CG (FP64, no precond) | 536 | 1.17 s | 219.9 GB | 9.9e-11 |
| FCG + Chebyshev **FP64** | 121 | 2.00 s | 250.2 GB | 7.8e-11 |
| FCG + Chebyshev **FP32** | 121 | 1.33 s | 170.2 GB | 7.8e-11 |

**FP32 vs FP64 preconditioner: identical iterations (121) and converged
residual (7.8e-11), at 0.66× the time** (model byte ratio 0.68). This is
ADR-0002 end-to-end on the target hardware: demoting only the preconditioner
interior to FP32 leaves the answer unchanged and cuts wall time by the byte
ratio. (Matches the cloud-VM claim in milestone0.md — 0.693 ratio, identical
convergence — now confirmed on Strix Halo.)

## Scope — what this does and does not show

- **Shows:** the STREAM roofline on the real target; the SpMV kernel at the
  roofline; the FP32 mixed-precision lever (isolated kernel *and* standalone
  solver) delivering the modeled byte/time reduction at unchanged accuracy.
- **Does NOT show:** the M2 DoD target (1.5–2.0× on a >5M-cell OpenFOAM case
  through `spumePCG`). That is an *integrated* wall-time comparison against a
  strong reference (GAMG) and needs the FP32-GAMG-inside-FP64-Krylov
  preconditioner — a later M2 slice. The current `spumePCG` FP32 Chebyshev is a
  correctness/bandwidth demonstrator, not yet a GAMG-beating integrated solver.
  No integrated speedup is claimed here.
