# vendor/ — upstream provenance manifest

## vendor/openfoam — OpenFOAM v2606 (ESI/OpenCFD), pruned

| | |
|---|---|
| Upstream | OpenFOAM® by OpenCFD Ltd (ESI), <https://www.openfoam.com> |
| Release | v2606 (`META-INFO/api-info`: api=2606, patch=0), released 2026-06-26 |
| Upstream tag | `OpenFOAM-v2606` (develop.openfoam.com; tag SHA to be recorded when reachable) |
| Source artifact | `OpenFOAM-v2606.tgz` via the official SourceForge mirror (`downloads.sourceforge.net/project/openfoam/v2606/`) |
| SHA-256 | `2a1310e3ed192cc4c521e1d22dcc176f57bec61160c878dc4348f21d6672294d` |
| License | GPL-3.0-or-later (see `openfoam/LICENSE.md`; upstream headers retained, never edited) |
| Decision record | ADR-0014 |

**Kept**: `src/`, `applications/`, `wmake/`, `bin/`, `etc/`, `META-INFO/`,
`Allwmake`, `LICENSE.md`, `README.md`,
`tutorials/incompressible/simpleFoam/motorBike/`,
`tutorials/resources/geometry/motorBike*.obj.gz`.

**Pruned**: all other `tutorials/`, `plugins/`, `modules/`, `doc/`,
`etc-mingw/`.

## Rules (ADR-0001)

- Never edit files under `vendor/` directly. The only change mechanism is
  the ordered patch stack in `patches/` (hard budget 2,000 lines,
  enforced by `scripts/check_patch_budget.sh` in CI).
- Updates replace the tree wholesale from a new official release
  artifact: verify its checksum, prune per the list above, update this
  manifest and ADR, re-apply the patch stack in order
  (`scripts/apply_patches.sh`).
- Trademark: this project is not approved or endorsed by OpenCFD Ltd.
  OPENFOAM® is a registered trademark of OpenCFD Ltd.
