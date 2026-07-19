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
  clean; the develop build+contract step is **blocked** — develop.openfoam.com
  now requires auth (HTTP 401 for anonymous clone), which also breaks the
  nightly canary's anonymous clone. Documented with a token-based fix.

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
