<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (C) 2026 Joern Ihlenburg -->

# Design: Finish Milestone 1 (vendoring, shim, contract tests, first leaf app)

- Status: Approved (brainstorming, 2026-07-19)
- Scope: Close the four M1 DoD items in `docs/roadmap.md`.
- Approach: **A (baseline-oracle first) with an early rebase dry-run.**

## Why this serves the end goal

M1 is scaffolding for the flagship (M3: motorBike/DrivAer interactive on a
Strix Halo APU at 2.5–4× tuned-CPU OpenFOAM). The deliverable that matters
long-term is a **trusted equivalence oracle**: a harness that proves
`spumePimpleFoam` produces *byte-identical* results to stock `pimpleFoam`
today, and — flipped from bitwise to rounding-reorder tolerance in M2 —
becomes the regression gate that lets mixed-precision, JIT kernels, and
GPU-resident solves land against something known-good without ever touching
upstream core (ADR-0001, ADR-0002, ADR-0003, ADR-0004). Every task below is
justified by that oracle or by keeping the upstream-rebase path cheap
(ADR-0008).

## M1 Definition of Done (from roadmap)

1. motorBike runs through the leaf app **bit-class-identical to stock**.
2. Contract suite green.
3. **One dry-run upstream rebase documented.**
4. (Implicit) leaf app + shim building against a v2606+ tree.

## Decisions locked during brainstorming

| # | Decision |
|---|----------|
| Equivalence cases | Light proxy cases (`pitzDaily`, `TJunction`, pimpleFoam-configured) automated in the **nightly** OpenFOAM CI; full **motorBike** run locally as the milestone headline, recorded in `docs/milestone1.md`. |
| Strictness | **Bitwise-identical** field files (header banner stripped) **plus identical per-timestep residual/iteration counts**. Any diff is a plumbing bug → hard fail. Justified because M1 changes no numerics. |
| Rebase target | Upstream **develop HEAD** (the ref the nightly canary already tracks); apply-cleanliness + ≤2,000-line budget + contract-green recorded in `docs/milestone1.md`. |
| Ordering | Commit plumbing → **early** rebase dry-run → equivalence harness → motorBike → DoD verification + roadmap update. |

## Reality constraints discovered

- **"Automated in CI" = the nightly OpenFOAM job, not per-push.** `ci.yml` has
  no OpenFOAM by design (M0 core only). Anything running `spumePimpleFoam` vs
  stock `pimpleFoam` needs a built OpenFOAM, which exists only in
  `nightly-contract.yml` (cached, 350-min jobs). The equivalence gate slots
  into the `contract-vendored` job, opt-in via `-DSPUME_WITH_OPENFOAM=ON`,
  exactly like the existing `spume-contract-cavity` test.
- **motorBike must be a pimpleFoam configuration.** `spumePimpleFoam` mirrors
  transient `pimpleFoam`; the stock motorBike tutorial is steady `simpleFoam`.
  The run diffs `spumePimpleFoam` vs stock **`pimpleFoam`** on a transient
  motorBike setup, derived from the full install's transient
  `pisoFoam/LES/motorBike` (PISO ≈ PIMPLE with one outer corrector). This is
  the correct baseline: M3's demo is transient external aero on motorBike.
- **The vendored tree is pruned of pimpleFoam tutorials.** Case fixtures are
  therefore **checked into `tests/`** (like `tests/contract/cases/cavity`),
  not pulled from vendored tutorials. Both `spumePimpleFoam` (via `wmake`) and
  stock `pimpleFoam` (present in the vendored solver sources) build against
  the same OpenFOAM tree, so the harness needs no external install.
- **git identity was unset** in the repo; set locally to
  `Joern Ihlenburg <ihlenburg@ihlems.de>` for DCO-signed commits. All work is
  on branch `milestone1/finish`.

## Work breakdown

### Step 1 — Commit the in-flight plumbing

Verify a clean rebuild first (evidence before claims), then commit the working
tree in logical, one-concern commits (conventional-commit style, DCO-signed,
no tool attribution per ADR-0011):

1. `feat(compat): add OpenFOAM compatibility shim headers` — `src/compat/foam.hpp`, `pimple.hpp`, README.
2. `feat(app): add spumePimpleFoam reference leaf application` — `applications/spumePimpleFoam/**` (source + Make; build artifacts already gitignored).
3. `chore(scripts): add foam-env and bootstrap_openfoam helpers` — `scripts/foam-env.sh`, `scripts/bootstrap_openfoam.sh`, `.gitignore` `.openfoam-config/` entry.

`.claude/settings.json` (`defaultMode: acceptEdits`) is a local workflow
preference unrelated to M1 — **excluded** from these commits (left in the
working tree / committed separately only if desired).

Build verification: `. scripts/foam-env.sh && (cd applications/spumePimpleFoam && wmake)` produces the binary with no warnings (warnings-as-errors where the leaf owns the flags).

### Step 2 — Early rebase dry-run (de-risk the fork assumption)

Replay `patches/` onto upstream develop HEAD and record the outcome:

1. Clone `--branch develop` from `develop.openfoam.com`; record the SHA.
2. Apply the patch stack (via the existing `patches/` tooling) onto the clone; capture whether it applies cleanly and the total changed-line count vs the 2,000-line budget (`scripts/check_patch_budget.sh`).
3. Build the rebased tree and run the contract suite against it (this is what the nightly canary already does; the dry-run documents a point-in-time result).
4. Write a **"Upstream rebase dry-run"** section in `docs/milestone1.md`: upstream SHA, apply result, budget usage, contract result, and any shim adjustments needed.

Heavy build note: building upstream develop is multi-hour; run in the
background and document the result when it lands. If the build cannot complete
in the session, document the **patch-apply + budget** result (the cheap,
decision-relevant part) and mark the contract-against-develop step as
in-progress with its command, rather than inventing a result.

### Step 3 — The bitwise-equivalence harness (the durable oracle)

**Location:** `tests/regression/` (OpenFOAM-dependent, guarded by
`SPUME_WITH_OPENFOAM`, mirroring `tests/contract/`).

**Components:**

- `tests/regression/cases/pitzDaily-pimple/` and `tests/regression/cases/TJunction-pimple/` — small, self-contained pimpleFoam fixtures (trimmed `endTime` for speed), derived from the full-install tutorials. Fixtures keep upstream FoamFile headers.
- `tests/regression/run_equivalence_pimple.sh` — mirrors `run_contract_cavity.sh`:
  - Sources the OpenFOAM env (`SPUME_OPENFOAM_DIR`, default vendored tree).
  - Ensures `spumePimpleFoam` is built (`wmake` the leaf app) and stock `pimpleFoam` is available.
  - For each fixture: copy into two work dirs (`spume/`, `stock/`), `blockMesh` both, run `spumePimpleFoam` in one and `pimpleFoam` in the other **serially** (identical decomposition ⇒ identical rounding order).
  - **Compare fields:** for every written time directory, diff each field file byte-for-byte after stripping only the fixed banner comment block (`/*---…---*/`) — no numeric tolerance. Any residual diff → fail.
  - **Compare convergence:** filter each solver log to the `Solving for …` residual/iteration lines (dropping execution-time lines) and assert the two are identical.
- `tests/regression/check_equivalence.py` — the field + log comparison logic (byte compare with banner strip; explicit, testable).
- CMake: add an `SPUME_WITH_OPENFOAM`-guarded test `spume-equivalence-pimple`, label `equivalence`, generous `TIMEOUT`.

**Reusability (the point):** the field comparator takes a mode —
`bitwise` (M1) or `reorder-tolerance` (M2+, the numerics equivalence class
from ADR-0002). M1 wires `bitwise`; M2 flips the flag. This is the same
oracle, not a throwaway.

### Step 4 — motorBike local headline run

1. Build a transient pimpleFoam motorBike case from `pisoFoam/LES/motorBike` geometry (RAS or LES, `nOuterCorrectors ≥ 1`), meshed with `surfaceFeatureExtract` + `snappyHexMesh` (the standard motorBike prep).
2. Run it through the equivalence harness (`spumePimpleFoam` vs stock `pimpleFoam`), serially or with an identical `decomposePar` layout on both sides.
3. Record in `docs/milestone1.md`: cell count, solver, wall time, and the bitwise-identical verdict. Heavy run → background; capture logs.

motorBike is **local-only** (not CI) — it is the milestone headline and the
M3 baseline seed, not a per-run gate.

### Step 5 — DoD verification + roadmap update

- Confirm: leaf app builds; contract suite green; equivalence green on proxies (local) + wired into nightly; motorBike bitwise-identical locally; rebase dry-run documented.
- Wire the equivalence step into `.github/workflows/nightly-contract.yml` (`contract-vendored` job: `ctest … -L 'contract|equivalence'`).
- Update `docs/roadmap.md`: mark M1 DoD items done, replace projected notes with measured facts.
- Write `docs/milestone1.md` as the milestone record (equivalence method, motorBike result, rebase dry-run).

## Testing strategy

- **Unit-ish:** `check_equivalence.py` gets a tiny self-test on synthetic field files (identical → pass; one-byte diff → fail) so the oracle itself is trusted.
- **Proxy equivalence:** `pitzDaily-pimple`, `TJunction-pimple` — bitwise green, local now + nightly CI.
- **Headline:** motorBike bitwise green, local, recorded.
- **Contract:** existing `spume-contract-cavity` stays green (unchanged).
- **No test weakening**, no tolerance introduced where bitwise is achievable (CLAUDE.md).

## Risks / decisions I will make autonomously (user AFK, full trust)

- If upstream-develop build exceeds the session, document the patch-apply + budget result and background the build; do **not** fabricate the contract-against-develop outcome.
- If a field file shows an unavoidable, legitimately non-deterministic header field (e.g. a UUID), extend the banner-strip to that specific line and note it — never widen to a numeric tolerance.
- Fixture `endTime` chosen small enough that nightly CI stays well under the 350-min budget while still writing ≥2 time directories to compare.

## Out of scope

M2 numerics (runtime-selectable SPUME solver classes, fused PCG, FP32-GAMG
preconditioner) — begins only after M1 DoD is green, under its own spec.
