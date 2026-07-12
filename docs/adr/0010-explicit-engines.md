# ADR 0010: Explicit engines for eligible case classes

- Status: Accepted
- Date: 2026-07-12

## Context

The elliptic pressure solve is the structural reason FVM CFD maps poorly
to GPUs: global coupling, hundreds of synchronizing iterations. For a
subset of case classes it can be legitimately replaced by purely local,
explicit updates that stream perfectly (cf. FluidX3D saturating RDNA
silicon while pressure-based solvers idle in reductions).

## Decision

- SPUME's timestep engine is swappable per case class. OpenFOAM remains
  the shell (meshing, BCs, function objects, post-processing).
- For eligible classes — external aero, ventilation, low-Mach transient —
  offer lattice-Boltzmann and artificial-compressibility /
  weakly-compressible engines emitted by the same JIT.
- Validation is at the physics level, not bitwise: acoustic-CFL and
  compressibility-error terms must be shown below the discretization
  error of the FV reference, per case family, before an engine is
  eligible for that family.
- Expected gain where applicable: 10-50x on the same silicon — the
  largest number in the project, and confined to these classes. The FVM
  path remains the general answer (2.5-4x class on APUs).

## Rejected alternatives

- Replacing the FVM path generally: incompressible internal flows,
  strongly coupled physics, and stiff sources stay pressure-based.
- Parallel-in-time (Parareal/MGRIT) as the big lever: converges poorly
  for advection-dominated turbulence; instead, parallelize the campaign —
  ensemble scheduling of DoE/optimization sweeps, one resident solver
  instance per GPU with deduplicated mesh pages.
