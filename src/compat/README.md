# src/compat — upstream compatibility shim

The single boundary through which SPUME code reaches the OpenFOAM library API
(architecture invariant #2). No file under `src/` outside this directory may
include an upstream `.H` directly; leaf applications include a shim header
from here instead.

## Headers

- `foam.hpp` — the OpenFOAM finite-volume umbrella (`fvCFD.H`) plus the
  API-version pin (`OPENFOAM >= 2606`, ADR-0014). Every other shim header
  includes this one first.
- `pimple.hpp` — upstream API surface for the incompressible PIMPLE solver
  family, consumed by `applications/spumePimpleFoam`. Add a sibling header
  per solver family rather than a raw include in leaf code.

## Scope of the "no direct upstream include" rule

The rule governs upstream **library headers** — the `.H` files that declare
OpenFOAM classes (`fvMesh`, `pimpleControl`, `turbulenceModel`, …). When
upstream churns, those declarations are what break, and this directory is the
one place that has to absorb it.

It does **not** govern the textual **app-assembly fragments** an OpenFOAM
solver includes by convention — `createFields.H`, `UEqn.H`, `pEqn.H`,
`setRootCaseLists.H`, and the like. Those are the standard wmake
solver-assembly mechanism, resolved via `-I` paths at build time, and SPUME
reuses the *vendored* copies in place (never copying them into owned space,
per ADR-0001). They are treated like vendored code, not SPUME source.

## Building against the shim

Leaf `Make/options` add `-I$(SPUME_PROJECT_DIR)/src` so `compat/…` headers
resolve. Source `scripts/foam-env.sh` to set that variable alongside the
OpenFOAM environment.
