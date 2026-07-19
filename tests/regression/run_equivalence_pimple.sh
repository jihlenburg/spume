#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
#
# M1 equivalence gate: prove spumePimpleFoam is byte-identical to stock
# pimpleFoam on every pimpleFoam fixture under tests/regression/cases/*-pimple.
# The leaf runs the reference solvers unchanged, so on the same case with the
# same (serial) decomposition every written field file must match to the byte
# and the per-timestep residual/iteration lines must be identical. Any diff is
# a plumbing bug -> hard fail (never a tolerance; AGENTS.md).
#
# SPUME_OPENFOAM_DIR selects the OpenFOAM tree (default: the vendored one, as
# the nightly CI uses; point it at a prebuilt install for local runs). Mirrors
# tests/contract/run_contract_cavity.sh.

set -eu

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) ||
    ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OF_DIR="${SPUME_OPENFOAM_DIR:-$ROOT/vendor/openfoam}"
WORK="${SPUME_EQUIV_WORK:-$ROOT/build/equivalence}"
CASES="$ROOT/tests/regression/cases"

[ -f "$OF_DIR/etc/bashrc" ] || {
    echo "equivalence: no OpenFOAM tree at $OF_DIR" >&2
    exit 1
}

export ROOT OF_DIR WORK CASES

# wmake and the solvers need bash with the OpenFOAM environment sourced; run
# the whole thing in one bash -c so that environment stays coherent.
bash -c '
    set -eu
    set +u; . "$OF_DIR/etc/bashrc" > /dev/null 2>&1 || true; set -u
    export SPUME_PROJECT_DIR="$ROOT"

    command -v blockMesh  > /dev/null || { echo "equivalence: blockMesh not built" >&2; exit 1; }
    command -v pimpleFoam > /dev/null || { echo "equivalence: stock pimpleFoam not built" >&2; exit 1; }

    if ! command -v spumePimpleFoam > /dev/null; then
        echo "equivalence: building spumePimpleFoam against $OF_DIR"
        wmake "$ROOT/applications/spumePimpleFoam"
    fi

    rc=0
    for casedir in "$CASES"/*-pimple; do
        [ -d "$casedir" ] || continue
        name=$(basename "$casedir")
        echo "== $name =="

        rm -rf "$WORK/$name"
        mkdir -p "$WORK/$name"
        cp -r "$casedir" "$WORK/$name/spume"
        cp -r "$casedir" "$WORK/$name/stock"

        ( cd "$WORK/$name/spume" && blockMesh > log.blockMesh 2>&1 && spumePimpleFoam > log.solver 2>&1 )
        ( cd "$WORK/$name/stock" && blockMesh > log.blockMesh 2>&1 && pimpleFoam       > log.solver 2>&1 )

        # Per-timestep residual/iteration lines must match exactly.
        if ! diff \
            "$WORK/$name/spume/log.solver" "$WORK/$name/stock/log.solver" \
            > /dev/null 2>&1; then
            # logs differ only in banner (exe name); compare the numeric lines.
            if ! diff \
                <(grep "Solving for" "$WORK/$name/spume/log.solver") \
                <(grep "Solving for" "$WORK/$name/stock/log.solver") > /dev/null; then
                echo "equivalence: $name residual/iteration lines differ" >&2
                rc=1
                continue
            fi
        fi

        if python3 "$ROOT/tests/regression/check_equivalence.py" \
            "$WORK/$name/spume" "$WORK/$name/stock"; then
            echo "equivalence: $name OK"
        else
            echo "equivalence: $name FIELDS DIFFER" >&2
            rc=1
        fi
    done
    exit "$rc"
'
