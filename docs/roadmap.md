# SPUME roadmap

Milestones are sequential; each has a definition of done (DoD). Expected
speedups are bands vs. a tuned znver5 FP64 CPU baseline unless stated.
Update this file at every milestone close; record new decisions as ADRs.

## M0 — Standalone mixed-precision core (no OpenFOAM, no GPU)

Prove ADR-0002 in a dependency-free library: SELL-C-8 (FP64/FP32),
reference + OpenMP cell-row SpMV, FP64 CG reference, flexible PCG with
FP32 Jacobi/Chebyshev preconditioning and diagonal equilibration,
deterministic-reduction mode, STREAM-style roofline probe and SpMV GB/s
reporter.

DoD: all CMake presets build warnings-as-errors; unit + determinism +
mixed-vs-FP64 equivalence tests green in CI; docs/milestone0.md contains
a measured GB/s table on the dev machine.

## M1 — Vendoring, shim, contract tests — **complete (2026-07-19)**

Vendor an OpenFOAM release under vendor/, patch-stack tooling (budget
2,000 lines), src/compat shim, contract suite (LDU addressing, GAMG
behavior, tutorial iteration counts) wired to nightly CI against upstream
dev, first leaf application (spumePimpleFoam) running reference solvers.

DoD: motorBike tutorial runs through the leaf app bit-class-identical to
stock; contract suite green; one dry-run upstream rebase documented.

Outcome (see docs/milestone1.md):
- Leaf app + `src/compat` shim build against v2606 (warnings-as-errors, clean).
- `spumePimpleFoam` is **bitwise-identical** to stock `pimpleFoam` — stronger
  than the required bit-class bar — on motorBike (353,830 cells, transient,
  the M3 flagship class) and on the `pitzDaily`/`TJunction` proxy fixtures now
  gating in nightly CI via a reusable comparator (`bitwise` now,
  `reorder-tolerance` stub for M2).
- Contract suite green (unchanged).
- Rebase dry-run: patch stack empty (0/2000 budget), so replay is trivially
  clean; the develop build+contract step could not run here because
  develop.openfoam.com now requires auth (HTTP 401 for anonymous clone). The
  nightly canary is repointed at the public `gitlab.com/openfoam/core/openfoam`
  mirror (same v2606 lineage, anonymous clone, no token), so it runs on the
  next scheduled nightly.

## M2 — CPU performance path

M0 engine integrated behind runtime-selectable solver classes; fused
pipelined PCG; FP32-GAMG-inside-FP64-Krylov preconditioner; NT stores and
gather prefetch via the (initial) CPU generator backend; renumbering and
CCD-aware rank mapping documented.

Target: 1.5-2.0x on a >5M-cell pressure-dominated case.
DoD: counter evidence (uProf GB/s vs measured roofline) per kernel in the
PRs; regression suite (iteration counts, force coefficients) green.

## M3 — HIP backend and the APU demo  <- flagship

IR v0 with hipRTC emission for gfx1151; GPU-resident pressure solve,
zero-copy GTT, coefficient-only updates; staggered CPU/GPU scheduling;
flat-Krylov-vs-GAMG study on GPU; cell-count fallback threshold measured.

Target: 2.0-2.6x cumulative. Demo: motorBike (OpenFOAM HPC benchmark) or
DrivAer running interactively on a Framework Desktop (Strix Halo),
reproducible container, honest methodology, stock-OpenFOAM comparison on
identical hardware.
DoD: the demo container runs from a clean pull; results write-up in
docs/. This milestone gates everything strategic.

**Progress (2026-07-19) + direction sharpened by measurement** (see ADR-0017,
memory m3-gpu-bandwidth-validated):
- Premise validated: iGPU gfx1151 = 240 GB/s (94% of the 256 GB/s LPDDR5X),
  ~2x the fabric-walled CPU (~150). Phase 1 (SELL SpMV 208 GB/s = 1.67x CPU,
  verified) and Phase 2 (full 15-level GPU-resident V-cycle + FCG, iteration
  parity 25==25 with CPU, independently verified) DONE. Prototypes preserved in
  ~/spume-m3-gpu-prototypes/ (build with -fopenmp at link; hipMallocManaged only).
- **Go all-in GPU-resident, do NOT split bandwidth-bound work across engines.**
  Measured: concurrent CPU+GPU aggregate = 189 GB/s < GPU-alone 240 — the shared
  controller is a ceiling, not a sum; concurrency contends and both slow. The GPU
  alone already extracts 94%. So "staggered CPU/GPU scheduling" (above) means
  partition by MEMORY-RESIDENCY: coarse levels (cache-resident, launch-latency-
  bound on GPU) on the CPU concurrent with the fine bandwidth-bound levels on the
  GPU — never two engines on DRAM at once.
- SSD (PCIe Gen4 x4 ~7 GB/s) maps into GPU address space for free (GPU global
  pool IS the 32GB system RAM, fine-grained coherent — no discrete-GPU GDS
  bounce), but at 34x below memory bandwidth it is a WARM-START / out-of-core
  lever (precomputed hierarchy + JIT-kernel load), not a solve-bandwidth lever.
- NPU (XDNA2, /dev/accel/accel0, amdxdna loaded) is a dense bf16/INT8 matmul
  engine — parked for a dense-in-SRAM low-precision branch, not sparse SpMV.
- Phase 3 (in progress): the GPU SELL-C-8 FP64 SpMV is productionized into
  `src/backends/gpu/` (ADR-0017) behind the `SPUME_ENABLE_HIP` build guard, with
  a HIP-free host API and a verify-then-bench ctest (`gpu-spmv-check`, skips
  cleanly with no GPU). Measured on gfx1151 (poisson7 128³): **207 GB/s = 81% of
  the 256 GB/s LPDDR5X peak, bitwise-exact vs the CPU reference, 8.3x a
  single-core reference SpMV.** The FP32 Chebyshev smoother is also landed and
  verified (`ChebyshevDeviceFP32`, gpu-cheb-check): **~219 GB/s = 85% of peak,
  in-class vs the CPU `ChebyshevPrecond<float>` at max_abs/‖z‖∞ 4.6e-7.**
  The aggregation transfers and the assembled **FP32 V-cycle** are also landed
  and verified (`VcycleDeviceFP32`, gpu-vcycle-check): in-class vs the CPU
  `AmgPrecond<float>` (max_abs/‖z‖∞ 1.2e-7 on a 15-level poisson7 96³ hierarchy),
  **17 ms/cycle = 3.1× the CPU apply** — coarsest solved on the CPU, one host
  sync per cycle. The 3.1× (vs 8× for SpMV) is the coarse-level launch tax.
- **The whole-solve GPU-resident FCG is landed and verified (`FcgSolverGPU`,
  gpu-fcg-check) — the M3 engine.** On poisson7 96³ (tol 1e-8) it solves to FP64
  accuracy entirely on the GPU (true residual 3.3e-9), with **iteration counts
  identical to the CPU (24 == 24)** and GPU/CPU solutions agreeing to **1.2e-14**
  — the empirical proof of the ADR-0002 firewall (the FP32 preconditioner yields
  a bit-class-identical FP64 answer). **5.6× the CPU solve** after the coarse-tax
  fix: a coarse_size sweep showed the CPU-default coarsening (to 200 rows, 15
  levels) adds ~10 launch-bound GPU levels with zero convergence benefit, so
  coarsening only to ~2500 rows (CPU CG takes the coarsest) is 1.7× faster with
  identical iterations (3.4× → 5.6×). The full path — SpMV, FP32 Chebyshev
  smoother, aggregation transfers, FP32 V-cycle, FP64 reduction, FP64 FCG — is
  GPU-resident, each stage at 81–87% of the memory roofline where it is
  bandwidth-bound.
- **The K-cycle is ported to the GPU (`VcycleDeviceFP32` kcycle path,
  gpu-kcycle-check).** On poisson7_graded 64³ (cz 1..1000) it cuts the GPU FCG
  from 20 (V-cycle) to 12 iterations and matches the CPU `AmgPrecond<float>`
  K-cycle exactly (12 == 12), solutions agreeing to 3.8e-15 — GAMG-parity
  convergence on graded meshes, GPU-resident.
- **Reality check on a real OpenFOAM matrix (2026-07-20).** Dumped a real SPD
  pressure system from a refined pitzDaily (782k cells, 59 MB — DRAM-bound, past
  the MALL cache) via a new `spumeDumpMatrix` diagnostic in spumePCG, and solved
  it standalone. **Correctness holds** (GPU FCG agrees with a 1e-11 golden
  reference to ~4e-12). **But performance does NOT transfer from poisson:** the
  GPU FCG is only ~1.2-1.5x the CPU here, vs 5.6x on structured poisson 96³.
  It is NOT bandwidth-bound on the real matrix. **Root cause (profiled):** the
  CPU coarse solve, blown up by an aggregation STALL — greedy strength
  aggregation coarsens 782k->2185 then stalls (2185->685, barely shrinking), so
  the coarsest is ill-conditioned and the unpreconditioned coarse CG needs ~456
  iters (vs 40 on poisson); the K-cycle fan-out (default kcycle_max_levels=5)
  hits that coarsest 32x/FCG-iter, and the GPU sits **79% idle** waiting on the
  serial host CG (68% of the solve). `renumberMesh` (half-bandwidth 697k->456)
  did not help; the numbering was never the issue.
  Matched-kml GPU-vs-CPU on the real matrix: kml5 1.20x, kml2 1.35x, **kml1
  1.42x** — lowering the fan-out speeds up both engines (fewer coarse solves) and
  slightly widens the GPU edge (less idle). But the coarse solve is a SHARED
  Amdahl ceiling (~68%): the fine bandwidth work the GPU wins is only ~32%, so
  the GPU can't pull far ahead until the coarse bottleneck is fixed.
  **First fix landed: cap the coarsest CG at 100 iters** (GPU FCG/V-cycle default
  coarse_max_iter 500->100). The coarse correction is only a preconditioner
  component, so capping it avoids the stall pathology; on a healthy coarsest the
  CG converges well under the cap (no-op). Measured on the real matrix: GPU FCG
  3967 ms -> 1345 ms at kml5 (**2.95x**), with the fair GPU-vs-CPU rising to
  ~2-2.4x (kml2/3); correct throughout (agrees with the 1e-11 reference to
  ~5e-12). GPU-only change; the CPU AmgPrecond/M2 path is untouched.
  Still open (core AMG, benefits both engines): a GPU-resident coarse solve to
  kill the remaining sync/idle; stall-aware coarsening; retest on a 3D case (this
  refined 2D pitz is a hard case for the aggregation). Also: share the fine
  operator (built twice); cell-count fallback; the demo container (motorBike).
- **The GPU FCG runs a large OpenFOAM simulation END TO END (2026-07-20).**
  `spumePreconditioner gpuFCG` in spumePCG solves the pressure equation entirely
  on the GPU inside a live run; simpleFoam on the 782k-cell refined pitz runs to
  a clean End (pressure converging 27->17 iters/step, residuals ~5e-8, correct).
  The HIP/OpenFOAM FP-trapping conflict is handled (fedisableexcept). BUT it is
  NOT yet faster live: the GPU hierarchy is rebuilt+re-uploaded every solve
  (~2-3 s "setup"), erasing the 1.35 s standalone-solve advantage. **The last
  optimization to be competitive live is the coefficient-only GPU update** (build
  the resident hierarchy once, re-upload only the changed values per solve) --
  the ADR-0017 residency goal. Then the direct stock-GAMG wall-time comparison
  on the same live case is honest and favorable.
- **Direct live stock comparison done (2026-07-20), and it is humbling.** 782k
  refined pitz, 10 steps, tol 1e-8, same box: **stock GAMG 25.3 s (43 p-iters) |
  SPUME gpuFCG 34.4 s (19) | SPUME amgFP32-CPU 188 s (19).** SPUME converges in
  ~half the iterations but is SLOWER in wall time -- it is **not yet superior to
  stock GAMG live.** Causes, both known: (1) the CPU path is crippled by the
  coarse-solve pathology at the M2 defaults (kml5/coarse_max_iter500) -> 18.8
  s/step; the coarse cap (100) + kml2 that the GPU already uses must be brought to
  the CPU/AmgPrecond path too. (2) the GPU is 3.4 s/step -- close to GAMG's 2.5 --
  but pays ~2 s per-solve hierarchy REBUILD on top of its 1.35 s solve. **The
  amortization math is decisive: a coefficient-only GPU update removes the ~2 s
  rebuild -> ~1.35 s/step -> ~1.85x FASTER than stock GAMG.** That single
  optimization is what turns "runs end-to-end" into "superior", so it is THE
  priority. Also bring the coarse cap to the CPU AmgPrecond (fixes the 188 s).
- **Deeper finding (2026-07-20): amortization ALONE will not make SPUME superior
  on this 2D case.** Instrumented the gpuFCG build-vs-solve split: build ~560 ms
  + solve ~1.1 s = ~1.66 s/p-solve (19 iters). Even with build->0 the GPU solve
  (~1.1 s) is ~1.5x SLOWER than GAMG's ~0.75 s p-solve. Root: per-ITERATION cost
  -- SPUME's K-cycle + 5-step Chebyshev + CPU coarse ~58 ms/iter vs GAMG's
  DIC-GaussSeidel V-cycle ~17 ms/iter. SPUME converges in fewer iters (19 vs 43)
  but each is ~3x costlier, and the GPU's fine-SpMV bandwidth win doesn't close it
  on this **2D, ~5-nnz/row** matrix (a worst case; the 6.2x was dense 3D poisson).
  So "superior vs stock GAMG" now needs EITHER (a) a large **3D** bandwidth-bound
  case (untested -- where the fine SpMV dominates and the GPU should win), or (b)
  a cheaper per-iteration (fewer Chebyshev steps, lighter K-cycle, GPU-resident
  coarse). Honest status: SPUME GPU is CORRECT and runs large sims end-to-end, but
  is not yet faster than stock GAMG on a 2D real case; the bandwidth thesis is
  unproven on real meshes until a large 3D case is run.
  rocprof roofline is blocked by a gfx1151 PMC-counter limitation (bandwidth
  stays model-over-kernel-time, ADR-0013).

## M4 — Explicit engine showcase (ADR-0010)

LBM and artificial-compressibility engines from the same IR for eligible
case classes; physics-level validation gates (acoustic CFL,
compressibility error below FV discretization error, per case family).

Target: 10-50x on eligible classes. DoD: one validated external-aero case
with the FV cross-check published alongside.

## M5 — Metal backend (ADR-0009)

FP32 preconditioner engine + LBM on Metal, one M4 Max, 2-3 month box.
DoD: the M0/M2 test suite passing via the Metal path; a single honest
cross-platform results table.

## M6 — Split system (n CPU + m GPU)

VRAM-resident fields, halo-only PCIe with compression fused into pack
kernels, interior/halo overlap, CPU ranks as coarse-level/IO servers.
Ensemble scheduling for DoE sweeps (one resident instance per GPU).

DoD: scaling study on >=2 dGPUs; halo compression on/off ablation.

## Cross-cutting rules

- Series A upstream MRs (ADR-0008) proceed in parallel from M2 onward.
- Assembly-level or ISA work only where the portable version measures
  <90% of roofline (ADR-0003).
- Every performance number in this file gets replaced by a measured one,
  with hardware/kernel/driver recorded, as milestones land.
