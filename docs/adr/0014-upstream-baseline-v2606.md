# ADR 0014: Upstream baseline: OpenFOAM v2606 (ESI), pruned tarball import

- Status: Accepted
- Date: 2026-07-12

## Context

Milestone 1 vendors upstream OpenFOAM (ADR-0001). ADR-0008 fixes the
lineage to ESI/OpenCFD (openfoam.com); the open questions were which
release to pin and how the tree physically lands in `vendor/`. At
decision time the current release is v2606 (2026-06-26); the previous is
v2512. The compat shim absorbs upstream renames per release, twice-yearly
pulls are budgeted, and the M3 demo compares against stock OpenFOAM on
identical hardware.

## Decision

- Pin **OpenFOAM v2606** (api=2606, patch=0) as the vendored baseline.
- Import mechanism: **pruned official source tarball**, no upstream git
  history. Provenance = the tarball's recorded SHA-256 plus the upstream
  tag name (`OpenFOAM-v2606`); the manifest lives in `vendor/README.md`.
- Prune list (recorded in the manifest; everything else kept intact):
  `tutorials/` except the motorBike case and its geometry, `plugins/`,
  `modules/`, `doc/`, `etc-mingw/`. `src/` is never pruned internally —
  partial pruning inside the build graph is how vendored trees rot.
- Upstream updates replace the tree wholesale from a new release tarball
  (never file-by-file edits), then the patch stack re-applies in order.
- The M1 dry-run rebase (roadmap DoD) targets upstream `develop` HEAD.

## Consequences

The shim is written against the API generation that will live longest
before the next pull (v2612). The vendored tree adds ~140 MB / ~14.5k
files to the repository — accepted as the price of hermetic builds and a
patch-stack-friendly layout. The M3 comparison baseline stays current
through the M1-M2 window. GitLab tag SHA to be recorded in the manifest
when develop.openfoam.com is reachable from the dev environment (the
proxy currently blocks it; the tarball hash is the binding pin).

## Rejected alternatives

- v2512: mature but superseded — vendoring it means absorbing v2606's
  renames in week one and demoing M3 against a stale baseline.
- Waiting for v2612: builds M1 against a moving target for six months.
- Git submodule: patches would apply at build time, not in-tree;
  contradicts the ADR-0001 patch-stack model and makes builds
  network-dependent.
- Full unpruned import: +260 MB of tutorials/plugins/modules that no
  milestone touches; prune list is small and documented instead.
- Foundation (openfoam.org) release: rejected in ADR-0008.
