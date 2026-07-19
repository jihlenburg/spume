<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (C) 2026 Joern Ihlenburg -->

# Design: M2b — FP32 mixed-precision preconditioner in spumePCG

- Status: Draft (autonomous session 2026-07-19; user AFK, full delegation)
- Scope: Run the ADR-0002 mixed-precision architecture inside OpenFOAM — an
  **FP32 preconditioner under the FP64 flexible outer Krylov** — via `spumePCG`,
  and verify it preserves FP64 solution accuracy within the rounding-reorder
  equivalence class. **Correctness only; no speedup is claimed** (bandwidth
  measurement needs uProf counter evidence — ADR-0013 — and is a separate step).

## Why this serves the end goal

This is the project's core thesis (ADR-0002): halving the bytes moved through
60–80% of runtime at zero accuracy cost, by keeping FP64 truth in the outer
Krylov and demoting only the preconditioner. It is the single largest
bandwidth lever toward the M3 flagship, and the exact split Apple GPUs (no FP64)
need (ADR-0009). M2a made a SPUME solver run in OpenFOAM at FP64; M2b flips on
the mixed precision that makes it bandwidth-first — proven correct first, made
fast (with counters) later.

## Numerics policy compliance (hard rules)

- Reduced precision lives **only inside the preconditioner**, under a flexible
  outer Krylov (FCG) — satisfied: the outer `fcg` and the operator `a` stay
  FP64; only the preconditioner's internal SpMVs/vectors are FP32.
- **Diagonal equilibration is mandatory before demotion** — satisfied:
  `make_eq_operator<float>` equilibrates in FP64 (`S A S`, unit diagonal) and
  narrows to FP32 only at the final store. It is the *only* public route to
  FP32 coefficients.
- **Default behaviour unchanged** — the FP32 path is opt-in via a dictionary
  key; the default `spumePCG` stays FP64 (the M2a Stage-1 diagonal Jacobi).

## Design

Reuse the M0 mixed-precision engine (already unit/regression tested:
`fcg(a_fp64, ChebyshevPrecond<float>(op32), …)`). Two owned changes:

1. **`src/bridge/ldu_to_sell`** — add `assemble_csr(...)` with the same LDU
   inputs and convention as `assemble_sell`, returning the `Csr` the FP32
   equilibrated operator is built from (`make_eq_operator<float>` takes a
   `Csr`). Same COO assembly, `coo_to_csr` instead of `sell_from_coo`.
   - Produces: `Csr assemble_csr(std::span<const int> lowerAddr,
     std::span<const int> upperAddr, std::span<const double> diag,
     std::span<const double> upper, std::span<const double> lower, int nCells)`.
   - Unit-tested (no OpenFOAM): the `Csr` reconstructs the same operator as the
     `Sell` on a known small system.

2. **`spumePCG`** — a dictionary key `spumePreconditioner`:
   - `jacobi` (default) — FP64 exact diagonal Jacobi (current M2a behaviour).
   - `chebyshevFP32` — FP32 Chebyshev semi-iteration on the equilibrated
     operator (`ChebyshevPrecond<float>`), with optional `chebyshevSteps`
     (default 5) and `chebyshevEta` (default 30). The negated SPD coefficients
     (M2a sign handling) feed `assemble_csr` → `make_eq_operator<float>`.
   The outer solve stays `fcg(a_fp64, precond, b, x)`; only the preconditioner
   object changes.

## Testing

- **Unit (no OpenFOAM):** `assemble_csr` reconstructs the same matrix as
  `assemble_sell` (SpMV on the CSR-built SELL equals the hand-computed `A·x`).
- **Equivalence gate:** extend `run_spumepcg_equivalence.sh` (or a sibling) to
  run `spumePCG` with `spumePreconditioner chebyshevFP32` and assert it stays
  within the reorder-tolerance class of reference PCG on pitzDaily-pimple. The
  FP32 preconditioner steers but does not define the answer, so the FP64
  solution must remain within class — this is the correctness proof of ADR-0002
  running in OpenFOAM.
- No default change; no test weakened; no perf claim without counters.

## Non-goals

- **No speedup number.** Iteration counts and FP32 usage are correctness
  evidence, not performance. The bandwidth win (uProf GB/s vs roofline per
  kernel) is a later M2 slice.
- FP32-GAMG specifically (vs Chebyshev), block-scaled formats, and the JIT CPU
  backend are later slices.
