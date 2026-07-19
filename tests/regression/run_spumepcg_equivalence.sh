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
# CMake build tree with the position-independent core/bridge static libs the
# solver library links (built before this test in the nightly CI job).
SPUME_BUILD_DIR="${SPUME_BUILD_DIR:-$ROOT/build/cpu-release}"

[ -f "$OF_DIR/etc/bashrc" ] || {
    echo "spumePCG: no OpenFOAM tree at $OF_DIR" >&2
    exit 1
}
[ -e "$SPUME_BUILD_DIR/src/core/libspume-core.a" ] || {
    echo "spumePCG: CMake static libs missing under $SPUME_BUILD_DIR — build the" >&2
    echo "spumePCG: cpu-release preset first (cmake --build --preset cpu-release)." >&2
    exit 1
}

export ROOT OF_DIR WORK CASE MODE SPUME_BUILD_DIR

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

    # Write explicit fvSolution for each case: only the pressure solver name
    # differs (PCG vs spumePCG); pFinal reuses $p (both correctors, incl. the
    # final one that writes the field). Both drive p to a TIGHT tolerance with
    # no relative early-out, because two different Krylov methods agree only to
    # the linear-solve tolerance, not machine rounding (Stage 1 note in the
    # design spec) — a comparison-harness setting, not a default change. The
    # U/k/epsilon solvers are identical in both, so they cannot introduce a
    # difference; they use the same tight settings for a clean comparison.
    write_fvsolution()  # $1 = case dir, $2 = pressure solver name
    {
        cat > "$1/system/fvSolution" <<EOF
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    p
    {
        solver          $2;
        preconditioner  DIC;
        tolerance       1e-10;
        relTol          0;
        maxIter         5000;
    }
    pFinal { \$p; }
    "(U|k|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-9;
        relTol          0;
    }
    "(U|k|epsilon)Final" { \$U; }
}
PIMPLE
{
    nNonOrthogonalCorrectors 0;
    nCorrectors         2;
}
EOF
    }

    write_fvsolution "$WORK/ref"  PCG
    write_fvsolution "$WORK/test" spumePCG
    # test loads the SPUME solver library.
    foamDictionary -entry libs -set "(spumeFoamSolvers)" "$WORK/test/system/controlDict" > /dev/null

    # Write extra digits so the comparison sees the true numerical agreement of
    # the two solvers (~1e-8), not the 6-digit ascii write truncation (~1e-5 on
    # O(1) values) that dominates at the fixture default.
    for d in ref test; do
        foamDictionary -entry writePrecision -set 12 "$WORK/$d/system/controlDict" > /dev/null
    done

    ( cd "$WORK/ref"  && blockMesh > log.blockMesh 2>&1 && pimpleFoam > log.solver 2>&1 )
    ( cd "$WORK/test" && blockMesh > log.blockMesh 2>&1 && pimpleFoam > log.solver 2>&1 )

    # Tolerance calibration (measured 2026-07-19 on pitzDaily-pimple, both
    # solvers converged tight, writePrecision 12): the fields agree to ~1e-8
    # absolute / ~1e-9 relative — two different Krylov methods reaching the same
    # SPD solution. rtol 1e-6 + atol 1e-7 clears that with margin; a genuinely
    # wrong solver diverges by O(1e-2) or more, orders above this floor.
    if python3 "$ROOT/tests/regression/check_equivalence.py" \
        "$WORK/ref" "$WORK/test" --mode="$MODE" --rtol=1e-6 --atol=1e-7; then
        echo "spumePCG: PCG-equivalent ($MODE) OK"
    else
        echo "spumePCG: DIFFERS from PCG ($MODE)" >&2
        exit 1
    fi
'
