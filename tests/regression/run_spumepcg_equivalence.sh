#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
#
# M2 solver-seam gate: prove the runtime-selectable spumePCG solver produces
# the same solution as the reference PCG on a real case. Runs stock pimpleFoam
# twice on the same fixture, changing only the pressure solver:
#   ref/  : solvers/p/solver = PCG
#   test/ : solvers/p/solver = spumePCG   (+ libs (spumeFoamSolvers))
# then compares the written fields.
#
# Comparison mode (SPUME_EQUIV_MODE, default reorder-tolerance):
#   - Stage 0 (passthrough) is bitwise-identical, so it also passes the looser
#     reorder-tolerance class.
#   - Stage 1 (real M0 CG) differs from PCG only at rounding-reorder level, so
#     the permanent gate uses reorder-tolerance (ADR-0002). Run with
#     SPUME_EQUIV_MODE=bitwise to assert the stronger Stage-0 property.
#
# SPUME_OPENFOAM_DIR selects the OpenFOAM tree (default: vendored), as in the
# other regression/contract runners.

set -eu

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) ||
    ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OF_DIR="${SPUME_OPENFOAM_DIR:-$ROOT/vendor/openfoam}"
WORK="${SPUME_SPUMEPCG_WORK:-$ROOT/build/spumepcg}"
CASE="$ROOT/tests/regression/cases/pitzDaily-pimple"
MODE="${SPUME_EQUIV_MODE:-reorder-tolerance}"

[ -f "$OF_DIR/etc/bashrc" ] || {
    echo "spumePCG: no OpenFOAM tree at $OF_DIR" >&2
    exit 1
}

export ROOT OF_DIR WORK CASE MODE

bash -c '
    set -eu
    set +u; . "$OF_DIR/etc/bashrc" > /dev/null 2>&1 || true; set -u
    export SPUME_PROJECT_DIR="$ROOT"

    command -v pimpleFoam > /dev/null || { echo "spumePCG: stock pimpleFoam not built" >&2; exit 1; }

    # Build the SPUME solver library into FOAM_USER_LIBBIN if not present.
    if [ ! -e "$FOAM_USER_LIBBIN/libspumeFoamSolvers.so" ]; then
        echo "spumePCG: building libspumeFoamSolvers"
        wmake libso "$ROOT/applications/libs/spumeFoamSolvers"
    fi

    rm -rf "$WORK"; mkdir -p "$WORK"
    cp -r "$CASE" "$WORK/ref"
    cp -r "$CASE" "$WORK/test"

    # ref: reference PCG (symmetric) with a DIC preconditioner.
    foamDictionary -entry solvers/p/solver        -set PCG "$WORK/ref/system/fvSolution"  > /dev/null
    foamDictionary -entry solvers/p/preconditioner -set DIC "$WORK/ref/system/fvSolution" > /dev/null

    # test: spumePCG, same controls, load the SPUME solver library.
    foamDictionary -entry solvers/p/solver        -set spumePCG "$WORK/test/system/fvSolution" > /dev/null
    foamDictionary -entry solvers/p/preconditioner -set DIC     "$WORK/test/system/fvSolution" > /dev/null
    foamDictionary -entry libs -set "(spumeFoamSolvers)" "$WORK/test/system/controlDict" > /dev/null

    ( cd "$WORK/ref"  && blockMesh > log.blockMesh 2>&1 && pimpleFoam > log.solver 2>&1 )
    ( cd "$WORK/test" && blockMesh > log.blockMesh 2>&1 && pimpleFoam > log.solver 2>&1 )

    if python3 "$ROOT/tests/regression/check_equivalence.py" "$WORK/ref" "$WORK/test" --mode="$MODE"; then
        echo "spumePCG: PCG-equivalent ($MODE) OK"
    else
        echo "spumePCG: DIFFERS from PCG ($MODE)" >&2
        exit 1
    fi
'
