# tests/contract — upstream invariant assertions

Contract tests assert the upstream invariants SPUME depends on (ADR-0001):
LDU addressing, GAMG behavior, tutorial-class iteration counts. They run
nightly (`.github/workflows/nightly-contract.yml`) twice: against the
vendored tree, and against upstream `develop` as the merge canary.

## Layout

- `src/spumeContractLdu/` — wmake app asserting LDU addressing invariants
  (owner < neighbour, upper-triangular ordering, losort/ownerStart
  consistency) that the Milestone 2 SELL conversion will rely on.
- `cases/cavity/` — 20×20 lid-driven cavity, icoFoam, GAMG on p: the
  iteration-count fixture.
- `expected/*.bands` — recorded iteration-count bands. Numbers are
  compared within bands, never exactly (numerics policy): rounding-level
  drift moves counts by a step; a semantic upstream change moves them out
  of band.
- `run_contract_cavity.sh` — meshes, runs, asserts. Honors
  `SPUME_OPENFOAM_DIR` (defaults to `vendor/openfoam`) so the same suite
  tests any OpenFOAM tree.

## Running locally

Needs a built OpenFOAM (`cd vendor/openfoam && source etc/bashrc &&
./Allwmake -j`), then:

```sh
cmake --preset cpu-release -DSPUME_WITH_OPENFOAM=ON
ctest --test-dir build/cpu-release -L contract --output-on-failure
```

Regular CI skips these (`SPUME_WITH_OPENFOAM` defaults OFF); the nightly
workflow turns them on.
