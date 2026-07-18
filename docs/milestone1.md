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

**Recommendation (needs maintainer action — not done autonomously because it
requires a secret):**

1. Register a read-only ESI GitLab deploy token and expose it to the nightly
   job as a secret, cloning with
   `https://gitlab-ci-token:${{ secrets.ESI_OF_TOKEN }}@develop.openfoam.com/...`.
2. Re-run the `develop` build + contract to complete this dry-run; append the
   apply-clean / budget / contract-green results here.
3. Until then, the vendored-tree contract job (`contract-vendored`) still
   exercises the full suite against v2606.

## Bitwise equivalence (proxy cases)

_Filled in by the equivalence harness — see below._

## motorBike headline run

_Filled in by the local motorBike run — see below._
