# ADR 0001: Fork topology — own the leaves, vendor the trunk

- Status: Accepted
- Date: 2026-07-12

## Context

SPUME is a hard fork of OpenFOAM that must keep pulling upstream releases.
Merge cost grows with the size of the diff against upstream core. The most
invasive planned change (replacing eager fvm::/fvc:: evaluation with fused
assembly) would be unmergeable if done inside src/.

## Decision

1. Upstream OpenFOAM lives in `vendor/openfoam/` as a vendored subtree,
   modified only through an ordered patch stack in `patches/` with a hard
   budget of 2,000 lines total (hooks and build-system only).
2. All SPUME code reaches upstream APIs through one compat shim
   (`src/compat/`) that absorbs upstream renames per release.
3. Aggressive code lives in SPUME-owned leaves: runtime-loaded solver and
   preconditioner libraries (OpenFOAM's runtime selection tables load them
   with zero core diff) and SPUME-owned solver applications
   (spumePimpleFoam etc.) that call the SPUME assembly engine directly.
   Applications are not pulled from upstream, so churn there is irrelevant.
4. A contract test suite asserts every upstream invariant SPUME depends on
   (LDU addressing, GAMG agglomeration behavior, tutorial iteration counts)
   and runs nightly against upstream dev as a merge canary.

## Consequences

Upstream pulls are a rebase of a small patch stack plus a shim update.
Semantic drift is caught by the contract suite months before a forced
integration. Budget ~20% of ongoing effort for twice-yearly pulls.

## Rejected alternatives

- In-tree fork (modify src/ directly): merge cost compounds per release;
  historically kills forks of this ambition within two releases.
- Plugin-only (no fork): solver plugins alone cannot reach the assembly
  layer or ship SPUME-owned applications; insufficient for the roadmap.
- Upstream-everything: maintainers cannot accept the aggressive layers
  (see ADR-0008 for what does go upstream).
