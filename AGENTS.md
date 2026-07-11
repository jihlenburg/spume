# AGENTS.md — SPUME

Binding guidance for AI coding agents in this repository. Read fully before
changing anything. When a task conflicts with a rule here, stop and ask.

## Project

SPUME is a bandwidth-first CFD engine derived from OpenFOAM: mixed-precision
FP64/FP32 solvers, per-case JIT-specialized kernels from a common IR, resident
execution on unified-memory hardware (AMD Zen 5 / RDNA / CDNA, Apple silicon).

Naming: **SPUME** in all prose, docs, and comments; `spume` (lowercase) for
the binary, repo slug, C++ namespace, and CMake targets.

## Architecture invariants

1. **Vendored trunk, owned leaves.** `vendor/openfoam/` is upstream source,
   modified only through the ordered patch stack in `patches/`
   (hard budget: 2,000 lines total). Never edit vendor files directly.
2. **Compat shim.** SPUME code reaches upstream APIs only through
   `src/compat/`. No direct upstream includes anywhere else.
3. **JIT-first.** ISA-specific work (AVX-512 scheduling, RDNA waitcnt /
   cache-policy bits, Metal simdgroup ops) lives in IR generator backends
   under `src/backends/`, never as hand-maintained kernels in solver code.
4. **Runtime dispatch.** Every optimized path has a portable reference
   implementation selectable at runtime. Default behavior = reference.

## Numerics policy (hard rules)

- Solution accuracy is FP64 at the requested tolerance. Reduced precision
  (FP32/FP16/block-scaled) is permitted only inside preconditioners under a
  flexible outer Krylov method.
- Diagonal equilibration is required before any precision demotion.
- Never change default solver behavior, tolerances, or schemes. All new
  behavior is opt-in via dictionary/CLI flags.
- Correctness equivalence class: results may differ from reference at
  rounding-reordering level only (same class as changing MPI rank count).
  Tests compare within this class, never bitwise — except the deterministic-
  reduction debug mode, which must be bitwise reproducible across thread and
  rank counts.

## Performance policy

- An optimization must either (a) move a kernel to the memory-bandwidth
  roofline or (b) lower the roofline by reducing bytes moved. Reject
  instruction-level work that does neither.
- Every performance claim requires counter evidence (uProf / rocprof / Metal
  counters): achieved GB/s vs. measured roofline, per kernel, attached to
  the PR.
- If the portable version already reaches ≥90% of roofline, delete the
  specialized version.
- Benchmarks: fixed power mode, warm-up runs, I/O excluded from the timed
  region, hardware/kernel/driver versions recorded.

## Testing

- `tests/unit/` — kernel and data-structure tests.
- `tests/contract/` — assertions on upstream invariants we depend on (LDU
  addressing, GAMG agglomeration, tutorial iteration counts). Run nightly
  against upstream dev as a merge canary.
- `tests/mms/` — Method of Manufactured Solutions order verification for any
  discretization-touching change.
- `tests/regression/` — iteration counts and force coefficients per solver
  family.
- No merge with failing tests. Never weaken a test or tolerance to pass.

## Licensing and provenance

- GPL-3.0-or-later. Every new file: SPDX header +
  `Copyright (C) 2026 Joern Ihlenburg`.
- Never remove or alter upstream copyright headers.
- All commits DCO-signed (`git commit -s`). The `Signed-off-by:` trailer is
  the committing human's certification of origin and is mandatory.
- Commit authorship is human-only. Never add `Co-Authored-By:` trailers,
  "Generated with" bylines, emoji signatures, or any tool/AI attribution to
  commit messages, PR descriptions, or changelogs. Accountability for every
  line rests with the human who commits it.
- No new dependencies without explicit maintainer approval. No AGPL or
  proprietary code. Never copy code from other projects without approval —
  reimplement from ideas.
- Trademark: never present SPUME as OpenFOAM. Keep the attribution sentence
  in user-facing docs.

## Style

- Vendored code: upstream OpenFOAM style, untouched outside the patch stack.
- SPUME code: C++20, repo clang-format, warnings-as-errors, no raw
  new/delete, spans over raw pointers.
- Commits: conventional commits, imperative, one concern per commit
  (e.g. `perf(spmv): fuse Amul with dot product`).

## Agents must never

- Edit `vendor/` directly, bypass the patch stack, or exceed its budget.
- Change numerics defaults or weaken tests.
- Add Co-Authored-By trailers, "Generated with" bylines, or any tool
  attribution to commits or PRs (see Licensing and provenance).
- Commit generated kernels, benchmark artifacts, or case data.
- Add dependencies or network access in the build without approval.
- Claim performance without counter data.

## Repository configuration

- `.claude/settings.json` (committed) sets `"attribution": {"commit": "",
  "pr": ""}` — do not override or remove it.
- `.githooks/commit-msg` strips any `Co-Authored-By:` trailer as hard
  enforcement; repo setup runs `git config core.hooksPath .githooks`.
  Do not modify or bypass hooks.

## Commands

Filled in as the build lands. Until then: `cmake --preset <name>`,
`ctest --preset unit`. Keep every preset green in CI.
