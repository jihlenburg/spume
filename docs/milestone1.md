<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (C) 2026 Joern Ihlenburg -->

# Milestone 1 — vendoring, shim, contract tests, first leaf app

Record of how M1's Definition of Done was met. See
`docs/superpowers/specs/2026-07-19-finish-m1-design.md` for the design and
`docs/superpowers/plans/2026-07-19-finish-m1.md` for the task breakdown.

## Leaf application + compat shim

`applications/spumePimpleFoam` builds against OpenFOAM v2606 through the
`src/compat/` shim only (architecture invariant #2). Clean rebuild verified:

```
$ . scripts/foam-env.sh && (cd applications/spumePimpleFoam && rm -rf Make/linux64* && wmake)
foam-env: OpenFOAM v2606  |  SPUME_PROJECT_DIR=/home/jihlenburg/src/spume
... g++ -std=c++17 -DOPENFOAM=2606 -Wall -Wextra -Wold-style-cast ... -o .../bin/spumePimpleFoam
```

Compiler flags include `-Wall -Wextra -Wold-style-cast -Wnon-virtual-dtor`;
the build completes with no warnings and links the identical reference solver
set as stock `pimpleFoam`.

## Upstream rebase dry-run

**Target:** OpenFOAM **develop HEAD** at `develop.openfoam.com` (the ref the
nightly merge-canary tracks, ADR-0001/ADR-0008).

**Patch stack state.** At M1 close the stack is empty — the vendored tree is a
pristine v2606 import with no hooks/build-system patches yet:

```
$ sh scripts/check_patch_budget.sh
patch budget: 0 / 2000 lines
$ sh scripts/apply_patches.sh
apply_patches: no patches in stack
```

A patch-application dry-run is therefore trivially clean (nothing to replay,
0 of the 2,000-line budget consumed). The substantive part of the procedure —
building `develop` and running the contract suite against it — is what
validates that upstream has not drifted an invariant SPUME depends on.

**Blocked: `develop` now requires authentication.** As of 2026-07-19,
`develop.openfoam.com` refuses anonymous git access:

```
$ curl -s -o /dev/null -w '%{http_code}' \
    "https://develop.openfoam.com/Development/openfoam.git/info/refs?service=git-upload-pack"
401
$ git ls-remote https://develop.openfoam.com/Development/openfoam.git HEAD
remote: HTTP Basic: Access denied. ... you're required to use a token ...
fatal: Authentication failed
```

The host is reachable (the GitLab web UI answers), but the smart-HTTP git
endpoint returns **HTTP 401** without credentials. No ESI GitLab token is
available in this environment, so the `develop` build + contract step of the
dry-run could not be executed and **is not reported** (results are never
fabricated, per ADR-0013 / CLAUDE.md).

**Impact — the nightly canary is affected too.** `.github/workflows/
nightly-contract.yml`, job `contract-upstream-develop`, clones this exact URL
*anonymously*:

```yaml
git clone --depth 1 --branch develop \
  https://develop.openfoam.com/Development/openfoam.git upstream-develop
```

That step will now fail at clone with the same 401. This is a real,
independently-useful finding: the merge-canary is broken on the upstream
access-policy change, not on a code invariant.

**Resolution — point the canary at the public mirror (no token).** The ESI
OpenFOAM sources are mirrored publicly under the `gitlab.com/openfoam` group at
`gitlab.com/openfoam/core/openfoam`, which carries the same v2606 lineage (the
`OpenFOAM-2606-rc*` branches) and exposes `develop` for anonymous clone. The
nightly canary now clones that mirror instead of the auth-walled
`develop.openfoam.com`, so **no `OPENFOAM_GITLAB_TOKEN` secret is needed**:

```yaml
git clone --depth 1 --branch develop \
  https://gitlab.com/openfoam/core/openfoam.git upstream-develop
```

The `develop` build + contract step of this dry-run therefore runs on the next
scheduled nightly (or `workflow_dispatch`); its apply-clean / budget /
contract-green result should be appended here from that run — it is not
reported now because it was not executed in this session (never fabricated,
ADR-0013). The vendored-tree contract job (`contract-vendored`) already
exercises the full suite against v2606 every night.

## Bitwise equivalence (proxy cases)

`spumePimpleFoam` runs the reference solvers unchanged in M1, so on the same
case with the same (serial) decomposition it must be **byte-identical** to
stock `pimpleFoam` — a stronger anchor than the rounding-reorder tolerance the
numerics policy uses once SPUME numerics actually diverge (that arrives in M2).

The oracle is `tests/regression/check_equivalence.py` (banner-stripped byte
compare across every written time directory + identical `Solving for …`
residual/iteration lines), driven by `tests/regression/run_equivalence_pimple.sh`
over the checked-in fixtures. Verified locally against the v2606 install:

| Fixture | Mesh | Steps written | Residual lines | Verdict |
|---------|------|---------------|----------------|---------|
| `pitzDaily-pimple` | blockMesh (2D step) | 2 (+t0) | 60 = 60 | bitwise-identical |
| `TJunction-pimple` | blockMesh (T-junction) | 2 (+t0) | matched | bitwise-identical |

The comparator is a real gate, not a rubber stamp: perturbing a single byte in
a written field flips it to FAIL, and restoring it flips it back to PASS. It
carries a `mode` (`bitwise` now; `reorder-tolerance` stub for M2), so the same
oracle becomes M2's mixed-precision regression gate.

Wired into CI: the pure-Python comparator self-test runs on every push
(`ctest --preset all`); the OpenFOAM-dependent equivalence gate
(`spume-equivalence-pimple`, label `equivalence`) runs in the nightly
`contract-vendored` job under `-DSPUME_WITH_OPENFOAM=ON`.

## motorBike headline run

The milestone headline and the seed of the M3 flagship baseline: a **transient
pimpleFoam** configuration of the v2606 motorBike case (the leaf mirrors
`pimpleFoam`, so the comparison is against stock `pimpleFoam`, not the steady
`simpleFoam` tutorial — and transient external aero is exactly M3's demo).

- Mesh: `snappyHexMesh` (6-rank parallel) reconstructed to a serial mesh — **353,830 cells**.
- Both solvers run serially, two fixed 1e-4 s steps, kOmegaSST RAS, ascii fields.
- Result: **bitwise-identical**. Solver exit 0/0; 14 `Solving for …` lines identical; every field at every written time directory byte-for-byte equal (e.g. the 10.3 MB `0.0002/U`). Negative control (1-byte perturbation) correctly fails.

The motorBike case is run locally (it needs `snappyHexMesh`); its case data is
not committed (CLAUDE.md). This closes M1 DoD item 1 at the strongest bar.
