# ADR 0007: Naming — SPUME

- Status: Accepted
- Date: 2026-07-12

## Context

The name must be lexically distant from "FOAM" (OPENFOAM(r) is OpenCFD's
registered trademark), unique and greppable, usable as a CLI binary, and
pronounceable in German, English, and Turkish.

## Decision

- Project name: SPUME (spume = foam blown off breaking waves; the heritage
  nod is invisible to trademark concerns).
- Written SPUME (all caps) in all prose, docs, and comments; `spume`
  (lowercase) for repo slug, binary, C++ namespace, CMake targets.
- Not an acronym. (If asked: "Sparse Preconditioned Unified-Memory
  Engine" may appear once in the README as a wink, never expanded in the
  name.)
- User-facing docs carry: "SPUME is based on OpenFOAM technology.
  OPENFOAM(r) is a registered trademark of OpenCFD Ltd. This project is
  not approved or endorsed by OpenCFD Ltd."
- Never present SPUME as OpenFOAM or as an OpenFOAM edition.

## Rejected alternatives

- OpenSPUME: "Open" + a foam word reintroduces the OpenFOAM echo the name
  was chosen to avoid, and the prefix is generic and dating.
- Whitewater (common word, US political baggage), Bora (VW model
  collision), Lodos (strong runner-up; kept as sub-branding candidate),
  Spindrift (beverage brand, sailing team).
- Wind-name scheme (Poyraz, Meltem, Karayel) reserved for sub-components.
