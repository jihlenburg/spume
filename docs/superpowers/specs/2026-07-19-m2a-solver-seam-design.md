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

## Stage 1 implementation notes (discovered 2026-07-19)

Building Stages 0 (done, merged-ready) surfaced three concrete requirements for
Stage 1 that were not obvious up front. Recording them so Stage 1 is a
de-risked, well-scoped next task.

1. **Convergence tolerance dominates, not machine rounding.** Two *different*
   Krylov solvers (M0 CG vs upstream PCG) stopped at a loose `relTol` (the
   pitzDaily fixture uses `relTol 0.01` for `p`) converge to each other only at
   the ~`relTol` level — their fields differ ~1%, which `reorder-tolerance`
   (rtol 1e-6) correctly rejects. The Stage-1 gate must therefore drive **both**
   ref and test to **tight** convergence (`relTol 0`, `tolerance 1e-9`) via the
   `foamDictionary` overrides in `run_spumepcg_equivalence.sh`, so both reach
   the same true linear solution and the fields then agree at the tolerance
   floor (~1e-8), well inside `reorder-tolerance`. This is a harness change, not
   a change to any default. `run_spumepcg_equivalence.sh` already parametrises
   the solver override — add the tight-tolerance override for the Stage-1 run.

2. **Cross-build-system linkage.** `libspumeFoamSolvers` (wmake) must reach
   `spume::assemble_sell` + `spume::cg` (CMake `spume-bridge`/`spume-core`).
   Two viable routes: (a) compile the required core+bridge `.cpp` **into** the
   wmake library via `Make/files` (self-contained, no path coupling — preferred
   for a runtime-loaded plugin), or (b) link the CMake-built static libs from
   `build/<preset>/src/{core,bridge}` (needs a build-dir variable from
   `foam-env.sh`; the nightly job already builds them before ctest). Either way
   the wmake TU that includes `bridge/ldu_to_sell.hpp` needs **C++20**
   (`std::span`); append `-std=c++2a` in `Make/options` (last `-std` wins over
   wmake's default `-std=c++17`), plus `-fopenmp` for the core's OpenMP paths.

3. **Interface guard.** `assemble_sell` captures only `diag`/`upper`/`lower`;
   OpenFOAM's `Amul` also adds **coupled-interface** contributions
   (processor/cyclic patches) via `interfaceBouCoeffs_`/`interfaceIntCoeffs_`.
   Serial single-region cases (pitzDaily) have no active coupled interfaces, so
   the bridged operator is complete. Stage 1 must **detect active coupled
   interfaces and delegate to the reference PCG** in that case (a correctness
   fallback, consistent with ADR-0004's reference-default) until halo coupling
   is handled in a later M2 slice. Guard on whether any `interfaces_` entry is a
   coupled interface with non-zero boundary coefficients.

Stage 1 status: **not implemented in the 2026-07-19 session** — stopped at the
verified Stage-0 boundary rather than commit unverified numerics. Tasks 1–4
(comparator mode, LDU shim, `ldu_to_sell` bridge, spumePCG Stage-0 passthrough)
are complete and tested.
