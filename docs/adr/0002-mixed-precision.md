# ADR 0002: Mixed-precision numerics architecture

- Status: Accepted
- Date: 2026-07-12

## Context

FVM CFD is memory-bandwidth-bound (arithmetic intensity ~0.1-0.2 FLOP/byte).
Halving bytes in the solver is the cheapest large speedup. But solution
accuracy must remain FP64-class, and pure-FP32 pressure Poisson solves stall
on large or high-aspect-ratio meshes.

## Decision

- FP64 truth: all state fields, geometry, fluxes, equation assembly, outer
  Krylov iterations, and residual evaluation stay FP64.
- Reduced precision (FP32, FP16, block-scaled formats) is permitted only
  inside preconditioners, under a flexible outer Krylov method (FGMRES/FCG
  style, iterative-refinement theory per Carson-Higham). The preconditioner
  steers; it never defines the answer.
- Diagonal equilibration is mandatory before any precision demotion
  (stretched boundary-layer meshes exceed FP32 dynamic range).
- Coarsest GAMG direct solve stays FP64.
- Large-offset scalars (absolute enthalpy etc.) are solved in delta form.
- Correctness equivalence class: results may differ from an FP64 reference
  at rounding-reordering level only — the same class as changing MPI rank
  count. Tests compare within this class, never bitwise, except the
  deterministic-reduction debug mode (fixed-order tree sums), which must be
  bitwise reproducible across thread and rank counts.

## Consequences

Roughly halved bytes through 60-80% of runtime at zero accuracy cost.
The FP32-interior split is also exactly what Apple GPUs (no FP64) require,
making the Metal backend feasible (ADR-0009).

## Rejected alternatives

- WM_PRECISION_OPTION=SP (pure FP32 build): changes results; convergence
  failures on ill-conditioned pressure systems.
- FP64 everywhere: leaves the largest single bandwidth lever unused.
- BF16 preconditioner on the XDNA2 NPU: dataflow engines cannot do
  random-access sparse gathers, and the NPU shares the same LPDDR5X pool —
  frees no bandwidth.
