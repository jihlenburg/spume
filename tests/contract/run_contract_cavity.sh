#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
#
# Contract runner: cavity/GAMG fixture (ADR-0001 invariant 4-of-4:
# tutorial-class iteration counts) plus the LDU addressing contract app.
#
# SPUME_OPENFOAM_DIR selects the OpenFOAM tree to test (default: the
# vendored one). The nightly canary points this at upstream develop.

set -eu

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OF_DIR="${SPUME_OPENFOAM_DIR:-$ROOT/vendor/openfoam}"
WORK="${SPUME_CONTRACT_WORK:-$ROOT/build/contract}"

[ -f "$OF_DIR/etc/bashrc" ] || {
    echo "contract: no OpenFOAM tree at $OF_DIR" >&2
    exit 1
}

# wmake needs bash; run the whole OpenFOAM part in one bash -c so the
# sourced environment stays coherent.
bash -c '
    set -eu
    set +u; . "'"$OF_DIR"'/etc/bashrc" > /dev/null 2>&1 || true; set -u

    command -v blockMesh > /dev/null || { echo "contract: blockMesh not built" >&2; exit 1; }
    command -v icoFoam   > /dev/null || { echo "contract: icoFoam not built" >&2; exit 1; }

    # Build the contract app against whichever tree is being tested.
    if ! command -v spumeContractLdu > /dev/null; then
        wmake "'"$ROOT"'/tests/contract/src/spumeContractLdu"
    fi

    rm -rf "'"$WORK"'/cavity"
    mkdir -p "'"$WORK"'"
    cp -r "'"$ROOT"'/tests/contract/cases/cavity" "'"$WORK"'/cavity"
    cd "'"$WORK"'/cavity"

    blockMesh > log.blockMesh 2>&1
    spumeContractLdu > log.spumeContractLdu 2>&1
    grep -q "SPUME-CONTRACT-LDU: OK" log.spumeContractLdu
    echo "contract: LDU addressing OK"

    icoFoam > log.icoFoam 2>&1
    grep -q "^End" log.icoFoam
'

python3 "$ROOT/tests/contract/check_bands.py" \
    "$WORK/cavity/log.icoFoam" \
    "$ROOT/tests/contract/expected/cavity.bands"

echo "contract: cavity fixture OK"
