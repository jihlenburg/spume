<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (C) 2026 Joern Ihlenburg -->

# Design: M2a — runtime-selectable SPUME solver integration seam

- Status: Draft (autonomous session 2026-07-19; user AFK, full delegation)
- Scope: The **first slice of M2** — integrate the M0 engine behind a
  runtime-selectable OpenFOAM solver, default behavior unchanged. The
  performance work that completes M2 (fused pipelined PCG, FP32-GAMG-in-FP64
  preconditioner, NT stores / gather prefetch via the CPU generator backend,
  renumbering / CCD-aware rank mapping) rides on later slices and is out of
  scope here.

## Why this serves the end goal

M3 (the flagship) needs SPUME's bandwidth-first solvers running *inside* real
OpenFOAM cases. The prerequisite is a seam where a SPUME solver can be selected
per case with **zero upstream core diff** and fall back to the reference by
default (ADR-0001, ADR-0004). This slice builds that seam and proves it against
the M1 equivalence oracle — nothing here claims a speedup (no counter evidence
yet, so no performance claim; ADR-0013).

## Integration mechanism (confirmed against v2606)

`Foam::lduMatrix::solver` exposes runtime selection tables `symMatrix` and
`asymMatrix`. A solver registers with
`lduMatrix::solver::addsymMatrixConstructorToTable<T>` (exactly as upstream
`PCG` does) and implements:

```cpp
virtual solverPerformance solve(scalarField& psi,
                                const scalarField& source,
                                const direction cmpt) const;
```

A SPUME-owned library that registers the name `spumePCG` is loaded by a case
via `libs (...)` in `system/controlDict` (or `fvSolution`) and selected via
`fvSolution/solvers/p { solver spumePCG; }`. No file under `vendor/` changes —
this is the "runtime selection tables load them with zero core diff" path from
ADR-0001. Default cases never mention `spumePCG`, so default behavior is
untouched (numerics policy).

## Architecture

Three owned units, each with one responsibility:

1. **`src/compat/ldu.hpp`** — shim boundary for the LDU/solver API
   (`lduMatrix`, `lduMatrix::solver`, `solverPerformance`, `scalarField`).
   No leaf/lib includes upstream LDU headers directly (invariant #2).

2. **`src/bridge/ldu_to_sell.{hpp,cpp}`** — pure translation between
   OpenFOAM's LDU addressing (`lowerAddr`, `upperAddr`, `diag`, `upper`,
   `lower`) and SPUME's `Sell<double>` (SoA SELL-C-8). Depends only on
   `spume::core` types + plain arrays passed in — *no* OpenFOAM types in its
   interface, so it is unit-testable without OpenFOAM (contract-tested against
   the LDU invariants the M1 suite already asserts).
   - Produces: `Sell<double> assemble_sell(labelUList lowerAddr,
     labelUList upperAddr, scalarField diag, scalarField upper,
     scalarField lower, label nCells)` — but expressed over `std::span` of
     `int`/`double` so the signature carries no OpenFOAM type (the caller in
     unit (3) does the `UList`→`span` adaptation).

3. **`applications/libs/spumeFoamSolvers/`** — the runtime-loaded library.
   `spumePCG : public lduMatrix::solver` registers in the `symMatrix` table;
   its `solve()` bridges the matrix to `Sell` (unit 2) and calls
   `spume::fcg` / `spume::cg` (M0), then maps the SPUME `SolveResult` back to
   `solverPerformance`. Reaches upstream only through `src/compat/ldu.hpp`.

## Staging (each step independently verifiable via the M1 oracle)

- **Stage 0 — passthrough (bitwise gate).** `spumePCG::solve` delegates to
  upstream `PCG` internally. A case selecting `spumePCG` is then
  **bitwise-identical** to the same case with `PCG`, provable by the existing
  `bitwise` oracle. This proves registration + library loading with zero
  numeric risk. (Deliverable of the M2a plan.)
- **Stage 1 — real SPUME CG under the reorder-tolerance oracle.** Swap
  `solve()` internals to the M0 `cg`/`fcg` over the bridged `Sell`. A SPUME CG
  differs from upstream PCG at the rounding-reorder level (different reduction
  order / recurrence), *not* bitwise — so this stage **requires the
  `reorder-tolerance` comparator mode**, which is the M1 stub. Verified
  equivalent within the ADR-0002 class on the proxy + motorBike cases.
- **Later M2 slices (separate specs):** fused pipelined PCG; FP32-GAMG-inside-
  FP64-Krylov preconditioner + diagonal equilibration; CPU generator backend
  (NT stores, gather prefetch); renumbering / CCD-aware rank mapping; **counter
  evidence per kernel** (uProf GB/s vs roofline) — the actual speedup and its
  proof.

## The comparator prerequisite (do first)

Stage 1 cannot be judged without `reorder-tolerance` mode in
`tests/regression/check_equivalence.py` (M1 left it a documented stub). It
compares OpenFOAM ascii field files within a relative+absolute tolerance
(the rounding-reorder equivalence class), field-type-aware (scalar vs vector
components), never bitwise. This is pure Python, TDD'd, dependency-free — the
first, safe M2 commit — and is the gate every subsequent M2 numeric change is
judged against.

## Testing

- `ldu_to_sell` unit tests (no OpenFOAM): round-trip a known small LDU system
  → `Sell` → SpMV equals a hand-computed `A·x`; symmetry handling; empty/tiny.
- `reorder-tolerance` comparator self-test: identical fields pass at zero tol;
  a within-tolerance perturbation passes; an out-of-tolerance one fails;
  vector fields compared per component.
- Stage 0: `spumePCG` bitwise-equal to `PCG` on `pitzDaily-pimple` (extend the
  equivalence runner to parametrize the pressure solver).
- Stage 1: `spumePCG` within `reorder-tolerance` of `PCG` on the proxies.
- No default-behavior change; no test weakened; no perf claim without counters.

## Non-goals / risks

- No speedup is claimed or implied by M2a; it is plumbing + equivalence.
- The `ldu_to_sell` bridge must preserve the LDU sign convention
  (OpenFOAM stores `-` off-diagonals for the discretised Laplacian); the unit
  test pins this with a hand-computed product so a sign error can't pass.
