<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (C) 2026 Joern Ihlenburg -->

# M2a — SPUME solver seam — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** Integrate the M0 engine behind a runtime-selectable OpenFOAM solver
(`spumePCG`), default behavior unchanged, proven against the M1 equivalence
oracle. No speedup is claimed (no counter evidence yet — ADR-0013).

**Architecture:** `src/compat/ldu.hpp` (shim) → `src/bridge/ldu_to_sell`
(OpenFOAM-free translation to `Sell<double>`) → `applications/libs/
spumeFoamSolvers` (`spumePCG : lduMatrix::solver`, registered in the symMatrix
table). Staged: passthrough (bitwise) → real M0 CG (reorder-tolerance).

**Tech Stack:** OpenFOAM v2606 (wmake lib), C++20, CMake/ctest, Python 3.

## Global Constraints

- OpenFOAM API floor v2606; vendored tree untouched (patch budget ≤2000, ADR-0001); upstream reached only via `src/compat/` (invariant #2).
- SPDX + `Copyright (C) 2026 Joern Ihlenburg` on every new file.
- DCO-signed conventional commits, one concern, no tool attribution (ADR-0011).
- Numerics defaults unchanged; reduced precision only inside preconditioners under a flexible outer Krylov, diagonal equilibration first (ADR-0002). Default solver = reference (ADR-0004).
- No perf claim without uProf counter evidence vs roofline (ADR-0013).
- Branch: `milestone2/solver-seam`.

---

### Task 1: `reorder-tolerance` comparator mode (the M2 gate) — TDD

The M1 oracle only does `bitwise`; Stage-1 SPUME CG differs from upstream PCG
at the rounding-reorder level, so implement the documented stub first. Pure
Python, no OpenFOAM.

**Files:** Modify `tests/regression/check_equivalence.py`,
`tests/regression/test_check_equivalence.py`.

**Interfaces:**
- Produces: `fields_equal(a, b, "reorder-tolerance", rtol=1e-6, atol=1e-9)` →
  parses OpenFOAM ascii internalField numeric blocks (scalar or vector) from
  both files and compares element-wise within `|x-y| <= atol + rtol*|y|`;
  non-numeric structure (dimensions, boundaryField keywords) must still match
  after banner strip. Returns bool.
- `compare_cases(..., mode, rtol, atol)` threads the tolerances through.

- [ ] **Step 1: Failing tests.** Add: identical vector field passes at default
  tol; a `1e-8` relative perturbation passes; a `1e-2` perturbation fails; a
  structural difference (different boundary keyword) fails even within tol.
- [ ] **Step 2: Run, verify fail** (`python3 test_check_equivalence.py`).
- [ ] **Step 3: Implement.** Add a numeric-block parser (regex for `(...)`
  vector tuples and bare scalars inside the `internalField nonuniform List<...>`
  body), compare element-wise; fall back to structural equality for header/BC
  text. Replace the `NotImplementedError` branch.
- [ ] **Step 4: Run, verify pass** (all tests green).
- [ ] **Step 5: Commit** `test(regression): implement reorder-tolerance comparator mode`.

---

### Task 2: `src/compat/ldu.hpp` shim

**Files:** Create `src/compat/ldu.hpp`; update `src/compat/README.md`.

- [ ] **Step 1** Add the header including `lduMatrix.H`, `lduMatrix::solver`,
  `solverPerformance.H`, `scalarField.H` behind `#include "compat/foam.hpp"`;
  document it as the LDU/solver shim boundary. Commit
  `feat(compat): add LDU/solver shim header`.

---

### Task 3: `ldu_to_sell` bridge (OpenFOAM-free) — TDD

**Files:** Create `src/bridge/ldu_to_sell.{hpp,cpp}`, `src/bridge/CMakeLists.txt`,
`tests/unit/test_ldu_to_sell.cpp`; wire into `src/CMakeLists.txt`, `tests/unit/CMakeLists.txt`.

**Interfaces:**
- Produces: `spume::Sell<double> assemble_sell(std::span<const int> lowerAddr,
  std::span<const int> upperAddr, std::span<const double> diag,
  std::span<const double> upper, std::span<const double> lower, int nCells)`.
  Symmetric matrices pass `lower == upper` (empty `lower` means "use upper").

- [ ] **Step 1: Failing test** — build a 3-cell 1-D Laplacian in LDU form by
  hand, assemble to `Sell`, assert `Sell` SpMV against a known `A·x` vector
  (pins the off-diagonal sign convention).
- [ ] **Step 2: Verify fail.** `ctest --preset all -R ldu_to_sell` → fail.
- [ ] **Step 3: Implement** the addressing walk (`lowerAddr[i]`/`upperAddr[i]`
  give the cell pair for face `i`; scatter `upper[i]`/`lower[i]` into rows).
- [ ] **Step 4: Verify pass.**
- [ ] **Step 5: Commit** `feat(bridge): translate OpenFOAM LDU to SELL-C`.

---

### Task 4: `spumeFoamSolvers` library — Stage 0 passthrough

**Files:** Create `applications/libs/spumeFoamSolvers/spumePCG.{H,C}`,
`Make/files`, `Make/options`.

**Interfaces:**
- `spumePCG : public Foam::lduMatrix::solver`, registered via
  `addsymMatrixConstructorToTable<spumePCG>`; `solve(psi, source, cmpt)`
  Stage 0 = construct upstream `PCG` with the same controls and delegate.

- [ ] **Step 1** Write the class + registration + `Make` files (link
  `-lfiniteVolume -lOpenFOAM`; `-I$(SPUME_PROJECT_DIR)/src`). `wmake libso`.
- [ ] **Step 2** Extend `run_equivalence_pimple.sh` to accept a pressure-solver
  override; add a `pitzDaily-pimple` variant selecting `solver spumePCG;` +
  `libs (spumeFoamSolvers);`. Assert **bitwise-identical** to `PCG`.
- [ ] **Step 3: Commit** `feat(solvers): add spumePCG runtime solver (stage 0 passthrough)`.

---

### Task 5: `spumePCG` Stage 1 — real M0 CG under reorder-tolerance

- [ ] **Step 1** Swap `solve()` internals to bridge (Task 3) + `spume::fcg`
  (diagonal Jacobi preconditioner) / `spume::cg`; map iterations + residual to
  `solverPerformance`.
- [ ] **Step 2** Equivalence in `reorder-tolerance` mode (Task 1) on the
  proxies; confirm iteration counts land in a sane band.
- [ ] **Step 3: Commit** `feat(solvers): back spumePCG with the SPUME M0 CG`.

## Self-Review

- Spec coverage: seam mechanism → Tasks 2/4; M0 integration → Tasks 3/5;
  oracle prerequisite → Task 1; default-unchanged → passthrough Stage 0 +
  opt-in `fvSolution`. Perf work explicitly deferred to later M2 slices.
- Placeholders: none — signatures and the sign-convention test are concrete.
- Type consistency: `assemble_sell(span...)` used identically in Tasks 3/5;
  `fields_equal(..., mode, rtol, atol)` consistent Tasks 1/4/5.
