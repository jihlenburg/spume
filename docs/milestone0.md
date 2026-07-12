# Milestone 0 — mixed-precision core: results

**Claim under test.** A flexible FP64 outer Krylov solver with an FP32
preconditioner converges to FP64-accurate solutions at materially lower
memory traffic.

**Verdict: demonstrated.** With everything else held fixed, switching the
Chebyshev preconditioner interior from FP64 to FP32 leaves the outer
iteration count and the converged residual *identical* (178 iterations,
relres 8.7e-11) while cutting modeled traffic by 32% and wall time by 33% —
and the wall-time ratio (0.673) matches the modeled byte ratio (0.680),
i.e. the win is bandwidth, not compute.

## Machine and toolchain

| | |
|---|---|
| CPU | Intel Xeon @ 2.10 GHz (virtualized; `-march=native` resolves to `sapphirerapids`), 4 cores / 4 threads |
| Caches (as exposed by the VM) | L1d 4×48 KiB, L2 4×2 MiB, L3 260 MiB (single instance) |
| Memory | 15 GiB |
| OS / kernel | Ubuntu 24.04, Linux 6.18.5 |
| Compiler | GCC 13.3.0, `-O3 -march=native`, OpenMP, 4 threads |
| Build | `cmake --preset cpu-bench` @ commit of this doc |
| Date | 2026-07-12 |

## Methodology and honest caveats

- Warm-up runs precede every timed region; best-of-reps is reported; no
  I/O inside timed regions (per AGENTS.md benchmark rules).
- **Deviations from the AGENTS.md evidence bar, disclosed:** this is a
  shared cloud VM — fixed power mode cannot be set or verified, and no
  hardware counters (uProf/perf) are available in the container. All
  "bytes moved" figures are therefore the *documented traffic model*
  (`sell.hpp` for SpMV; per-iteration counts in `bench/main.cpp` for
  solvers), and the roofline reference is the *measured* STREAM triad.
  Numbers are indicative for this container, not lab-grade.
- The VM exposes a 260 MiB L3. Problem sizes for the headline numbers were
  chosen so the matrix stream (≥ 450 MB) cannot be cache-resident. Smaller
  cases (e.g. 128³, 209 MB) run largely out of L3 here and report
  above-roofline GB/s; they are excluded from conclusions.
- Solver effective GB/s can slightly exceed triad because the model counts
  each solver vector once per operation while several 56 MB vectors stay
  partially L3-resident. The FP64-vs-FP32 *comparison* is unaffected: both
  variants share the same model and the same residency behavior.

## Measured roofline (STREAM-style probe, 3 × 256 MiB, best of 10)

| kernel | GB/s |
|---|---|
| copy | 38.0 |
| scale | 35.3 |
| add | 41.5 |
| **triad (machine peak reference)** | **42.7** |

## SpMV — SELL-C-8, Poisson 192³ (n = 7,077,888, nnz = 49,324,032, padding 1.001, best of 5)

| precision | model bytes/SpMV | time | achieved GB/s | % of triad |
|---|---|---|---|---|
| FP64 | 706.0 MB | 19.49 ms | 36.2 | 85% |
| FP32 (equilibrated) | 451.8 MB | 12.46 ms | 36.3 | 85% |

Both precisions run at the same achieved bandwidth — the kernel is at the
memory roofline — and the FP32 time ratio (0.639) equals the model byte
ratio (0.640). Bytes are the budget; FP32 coefficients lower it.

## Solvers — Poisson 192³, tol 1e-10, random RHS

`cg64` = plain FP64 CG (reference). `fcg-cheb64/32` = flexible PCG, FP64
outer, Chebyshev(steps 5, eta 30) preconditioner on the equilibrated system
in FP64 / FP32. True residuals recomputed outside the timer.

| solver | iterations | wall time | modeled traffic | effective GB/s | final relres |
|---|---|---|---|---|---|
| cg64 | 790 | 24.44 s | 1094 GB | 44.8 | 9.8e-11 |
| fcg-cheb64 | 178 | 35.19 s | 1243 GB | 35.3 | 8.7e-11 |
| **fcg-cheb32** | **178** | **23.69 s** | **845 GB** | 35.7 | **8.7e-11** |

The controlled comparison is cheb64 → cheb32 (only the preconditioner
precision changes):

- outer iterations: **178 → 178** (identical), final residual identical —
  FP64 accuracy is untouched, as the flexible outer iteration guarantees;
- modeled traffic: 1243 → 845 GB (**−32%**);
- wall time: 35.19 → 23.69 s (**−33%**); time ratio 0.673 vs model byte
  ratio 0.680.

Notes: at this conditioning (κ ≈ 1.5e4) and eta = 30, Chebyshev-preconditioned
FCG only just edges out plain CG in wall time (×1.03) — polynomial
preconditioning mostly trades outer iterations for inner SpMVs. That is
expected and not the claim; the claim is the *precision* axis, which
delivers its full modeled saving. Stronger preconditioners (and the
FP16/block-scaled interiors they enable) are later milestones.

Correctness backing: `tests/regression/` asserts, on every CI run, that the
mixed-precision solutions match pure FP64 within the theorem-backed
rounding-reordering bound and that iteration counts stay within 20% —
plus bitwise reproducibility across 1/4/16 threads in
deterministic-reduction mode.

## Reproduce

```sh
cmake --preset cpu-bench && cmake --build --preset cpu-bench
OMP_NUM_THREADS=4 ./build/cpu-bench/bench/spume-bench stream
OMP_NUM_THREADS=4 ./build/cpu-bench/bench/spume-bench spmv  --nx 192 --ny 192 --nz 192 --reps 5
OMP_NUM_THREADS=4 ./build/cpu-bench/bench/spume-bench solve --nx 192 --ny 192 --nz 192
```

Pick sizes so the FP64 matrix stream comfortably exceeds your last-level
cache, or the GB/s figures will flatter you.
