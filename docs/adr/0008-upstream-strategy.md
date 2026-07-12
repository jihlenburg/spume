# ADR 0008: Upstream contribution strategy

- Status: Accepted
- Date: 2026-07-12

## Context

Parts of SPUME's work are mergeable upstream and reduce long-term fork
divergence; most are not. Upstream target for performance work is
ESI/OpenCFD (openfoam.com; HPC Technical Committee, benchmark suite,
precedent for external modules like OGL and PETSc4Foam).

## Decision

Three series, by destination:

- Series A — small core MRs, each single-concern, default-off:
  1. Krylov workspace reuse (no numerics change),
  2. fused Amul+dot / update+norm paths behind an optimisationSwitch,
     with the rounding-reordering equivalence class stated in the MR text,
  3. SPDP hardening (diagonal equilibration, delta-form for large-offset
     scalars) framed as robustness fixes,
  4. officially cached addressing metadata (losort/ownerStart) with
     topology-change invalidation.
- Series B — the CPU solver work (SELL-C-8, cell-row kernels, fused
  pipelined PCG, FP32-in-FP64 preconditioner) as an external
  runtime-loadable module aiming for the official modules listing.
  Threading lives here, never in core MRs (maintainers are MPI-only;
  hybrid measured ~10% slower on memory-bound kernels).
- Series C — GPU kernel work (gfx tuning, persistent-kernel PCG,
  HIP-expressible cache hints) goes to Ginkgo/OGL, not OpenFOAM.

Process rules: design issue / HPC TC contact before code; strict upstream
style; no new mandatory dependencies; demonstrated no-regression on
non-AMD hardware; check the target project's AI-disclosure policy per MR
(ADR-0011 applies to SPUME, not to upstream requirements).

## Rejected alternatives

- One large performance MR: dies in review queue regardless of quality.
- OpenMP in core: contradicts maintainers' documented position; burns
  credibility for no gain.
- Foundation (openfoam.org) as primary target: small review team,
  history of absorbing ideas rather than merging performance PRs.
