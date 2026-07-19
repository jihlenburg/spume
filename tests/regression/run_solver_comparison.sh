#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
#
# SPUME-vs-stock wall-time comparison harness (MANUAL perf tool, not a CI gate).
#
# A/B-compares the WALL TIME of the SPUME pressure path against stock pimpleFoam
# on the same case and the same machine, then confirms the two solutions stay
# inside the rounding-reorder equivalence class before it trusts any number:
#   stock : stock pimpleFoam,       fvSolution p-solver = GAMG
#   spume : spumePimpleFoam leaf,   fvSolution p-solver = spumePCG
#                                   (+ libs (spumeFoamSolvers))
# Only the pressure solver (name + its natural sub-entries + the libs load)
# differs between the two sides. Everything else — mesh, schemes, correctors,
# U/k/epsilon solvers, write precision — is identical, so a wall-time delta is
# attributable to the pressure solve alone.
#
# HONESTY (AGENTS.md numerics/perf policy, ADR-0013): the current spumePCG uses
# an FP64 Jacobi preconditioner by default (opt-in FP32 Chebyshev via
# SPUME_PRECOND=chebyshevFP32). That is a WEAKER preconditioner than stock GAMG,
# so SPUME is EXPECTED TO BE SLOWER here — this harness reports the true median
# ratio SPUME/stock with no spin and asserts NO speedup. It is infrastructure
# that only becomes meaningful when the M2 FP32-GAMG-inside-FP64-Krylov
# preconditioner lands (docs/roadmap.md, M2); at that point the same harness
# measures the real bandwidth-first win without modification.
#
# BUILD PARITY (critical for a fair number): a wall-time comparison is only fair
# if both binaries were compiled for the SAME target. The stock number is timed
# against the TUNED znver5 build (platform linux64GccDPInt32OptZnver5, produced
# by the /usr/local/bin/openfoam2606-znver5 wrapper via its overlay
# ~/.OpenFOAM/openfoam2606-znver5.sh, which sets WM_OPTIONS=...Znver5 and
# -march=znver5). This harness therefore (1) re-builds spumePimpleFoam and
# libspumeFoamSolvers with wmake INSIDE the selected environment every run
# (wmake is incremental and cheap), and (2) asserts that the resolved paths of
# pimpleFoam, spumePimpleFoam, and libspumeFoamSolvers.so all live under the
# active $WM_OPTIONS platform. A platform mismatch fails LOUDLY and does not
# time anything.
#
# METHODOLOGY (ADR-0013 — quantify noise, do not disclaim it):
#   - mesh/setup (blockMesh, case copy) is done ONCE up front and EXCLUDED from
#     the timed region; ONLY the solver invocation is timed.
#   - one warm-up run per side, EXCLUDED from the statistics.
#   - N timed reps (SPUME_REPS, default 3), INTERLEAVED stock/spume/stock/... to
#     make the A/B ratio immune to slow drift.
#   - reports median wall time + coefficient of variation (CV) per side; a side
#     with CV > 5% is flagged UNSTABLE.
#   - brackets steal-time from /proc/stat around the whole timed region as a
#     neighbor-interference indicator (this is a shared, powersave-governed,
#     non-lab machine; CV + steal carry the honest error bars in lieu of a fixed
#     power mode and hardware counters).
#   - records provenance: build platform, cpufreq governor, OMP thread env,
#     hostname.
#
# THREAD PINNING: defaults OMP_NUM_THREADS=16 OMP_PROC_BIND=close
# OMP_PLACES=cores (only if not already set — a user override is respected),
# matching the 16 physical cores of this machine (AMD RYZEN AI MAX+ 395, 16
# cores / 32 threads per `lscpu`). The effective OMP env is recorded in the
# report.
#
# EQUIVALENCE VERDICT: the two solutions are compared with check_equivalence.py
# in reorder-tolerance mode. Both p-solvers are driven to a matched TIGHT
# tolerance (1e-10, relTol 0, pFinal reuses $p) with writePrecision 12 so a
# multigrid method and a Krylov method land on the same SPD solution to the
# linear-solve floor rather than early-outing at different relative tolerances.
# These are comparison-harness settings applied inside a COPIED case; they do
# NOT change any tracked default and so do not violate the numerics policy.
#
# Tolerance calibration (measured 2026-07-19 on pitzDaily-pimple): GAMG and
# spumePCG (Jacobi) agree to max |Δp| ~ 1.7e-7 absolute. That is LOOSER than the
# ~1e-8 of PCG-vs-spumePCG the sibling run_spumepcg_equivalence.sh calibrates,
# for two honest reasons: (1) GAMG is a fundamentally different method than a
# Krylov solver — a coarser but still legitimate reduction order — and (2) the
# pitzDaily pressure field spans O(150), so element-wise agreement of ~1e-7 is
# ~1e-9 relative to field scale; a fixed atol tuned for O(1) fields trips on the
# many cells where p crosses zero. The defaults here are therefore rtol=1e-6,
# atol=1e-6 (SPUME_EQUIV_RTOL / SPUME_EQUIV_ATOL override them): ~6x above the
# measured agreement, still ~4 orders below the O(1e-2) a genuinely wrong solver
# produces, so the guard keeps its teeth. This is not weakening an existing
# test — the sibling equivalence gate keeps its 1e-7; it is scale-appropriate
# calibration for a GAMG-vs-Krylov comparison on a large-magnitude field.
#
# If the fields are NOT within this class the two solvers disagree, the
# wall-time numbers are meaningless, and the run is declared VOID: the script
# says so and exits non-zero.
#
# USAGE:
#   sh tests/regression/run_solver_comparison.sh [caseDir]
#   SPUME_REPS=5 sh tests/regression/run_solver_comparison.sh
#   SPUME_PRECOND=chebyshevFP32 sh tests/regression/run_solver_comparison.sh
#   SPUME_OPENFOAM_ENV=/path/to/env.sh sh tests/regression/run_solver_comparison.sh
#
# ENV VARS (all optional):
#   SPUME_OPENFOAM_ENV  environment to source (default the znver5 overlay;
#                       falls back to $SPUME_OPENFOAM_DIR/etc/bashrc).
#   SPUME_OPENFOAM_DIR  OpenFOAM tree for the fallback env
#                       (default $HOME/OpenFOAM/OpenFOAM-v2606).
#   SPUME_REPS          timed reps per side (default 3).
#   SPUME_PRECOND       spumePCG preconditioner: jacobi (default) | chebyshevFP32.
#   SPUME_EQUIV_RTOL    reorder-tolerance relative tol for the verdict (default 1e-6).
#   SPUME_EQUIV_ATOL    reorder-tolerance absolute tol for the verdict (default 1e-6;
#                       see the calibration note above — raise on cases whose
#                       fields span a much larger magnitude than pitzDaily).
#   SPUME_CMP_WORK      scratch dir (default $ROOT/build/solver-comparison).
#   SPUME_BUILD_DIR     CMake build tree holding the position-independent
#                       core/bridge static libs libspumeFoamSolvers links
#                       (default $ROOT/build/cpu-release; build the cpu-release
#                       preset first). NOTE: those core libs carry the actual
#                       spumePCG kernels; for a maximally fair znver5 number
#                       they should themselves be a znver5-tuned release build.
#   OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES  respected if already set.
#
# This is a standalone manual tool and is deliberately NOT registered as a
# pass/fail CI gate. It could be added under -DSPUME_WITH_OPENFOAM with a `perf`
# label, but must never gate merge on the ratio; the only hard failures are a
# build/platform mismatch and a VOID equivalence verdict.

set -eu

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) ||
    ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)

OF_DIR="${SPUME_OPENFOAM_DIR:-$HOME/OpenFOAM/OpenFOAM-v2606}"
ENVFILE="${SPUME_OPENFOAM_ENV:-$HOME/.OpenFOAM/openfoam2606-znver5.sh}"
CASE="${1:-$ROOT/tests/regression/cases/pitzDaily-pimple}"
REPS="${SPUME_REPS:-3}"
WORK="${SPUME_CMP_WORK:-$ROOT/build/solver-comparison}"
SPUME_PRECOND="${SPUME_PRECOND:-jacobi}"
# Stock-side baseline p-solver. Two honest choices:
#   GAMG      (default) — GAMG as a standalone solver (its own V-cycle iterates
#             to convergence). The classic OpenFOAM pressure solver.
#   PCG-GAMG  — PCG with GAMG as its preconditioner (both Krylov-accelerated).
#             This is the APPLES-TO-APPLES baseline for spumePCG+gamgFP32: an
#             outer Krylov method wrapping a GAMG V-cycle, differing from SPUME
#             only in V-cycle precision (FP64 GaussSeidel vs FP32 Chebyshev).
#             On ill-conditioned/stretched meshes GAMG-as-solver can stall at
#             maxIter while PCG-GAMG converges — comparing against it is the
#             only way to isolate the FP32-bandwidth effect from Krylov rescue.
SPUME_BASELINE="${SPUME_BASELINE:-GAMG}"
# reorder-tolerance equivalence-class bounds for the verdict (see header note).
SPUME_EQUIV_RTOL="${SPUME_EQUIV_RTOL:-1e-6}"
SPUME_EQUIV_ATOL="${SPUME_EQUIV_ATOL:-1e-6}"
# CMake build tree with the position-independent core/bridge static libs the
# spumeFoamSolvers library links (built before this tool via the cpu-release
# preset), exactly as run_spumepcg_equivalence.sh consumes it.
SPUME_BUILD_DIR="${SPUME_BUILD_DIR:-$ROOT/build/cpu-release}"

# Pin to the 16 physical cores unless the caller already chose a layout.
: "${OMP_NUM_THREADS:=16}"
: "${OMP_PROC_BIND:=close}"
: "${OMP_PLACES:=cores}"
export OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES

# Prefer the tuned znver5 overlay; fall back to a plain tree bashrc if absent.
if [ ! -r "$ENVFILE" ]; then
    if [ -r "$OF_DIR/etc/bashrc" ]; then
        echo "comparison: znver5 overlay not found; falling back to $OF_DIR/etc/bashrc" >&2
        echo "comparison: (build parity holds, but the stock number is NOT the tuned znver5 build)" >&2
        ENVFILE="$OF_DIR/etc/bashrc"
    else
        echo "comparison: no OpenFOAM environment (looked for $ENVFILE and $OF_DIR/etc/bashrc)" >&2
        exit 1
    fi
fi
[ -d "$CASE" ] || {
    echo "comparison: case directory not found: $CASE" >&2
    exit 1
}
case "$REPS" in
    ''|*[!0-9]*) echo "comparison: SPUME_REPS must be a positive integer, got '$REPS'" >&2; exit 1;;
esac
[ "$REPS" -ge 1 ] || { echo "comparison: SPUME_REPS must be >= 1" >&2; exit 1; }
[ -e "$SPUME_BUILD_DIR/src/core/libspume-core.a" ] || {
    echo "comparison: CMake static libs missing under $SPUME_BUILD_DIR — build the" >&2
    echo "comparison: cpu-release preset first (cmake --build --preset cpu-release)." >&2
    exit 1
}

case "$SPUME_BASELINE" in
    GAMG|PCG-GAMG) ;;
    *) echo "comparison: SPUME_BASELINE must be GAMG or PCG-GAMG, got '$SPUME_BASELINE'" >&2; exit 1;;
esac

export ROOT ENVFILE CASE REPS WORK SPUME_PRECOND SPUME_BASELINE SPUME_BUILD_DIR
export SPUME_EQUIV_RTOL SPUME_EQUIV_ATOL

# Everything below needs the OpenFOAM environment (bash: the znver5 overlay uses
# printf -v and ${!var}) and wmake, so it runs in one coherent bash -c block.
bash -c '
    set -eu
    set +u; . "$ENVFILE" > /dev/null 2>&1 || true; set -u
    export SPUME_PROJECT_DIR="$ROOT"

    # ---- provenance -------------------------------------------------------
    HOSTN=$(hostname 2>/dev/null || echo unknown)
    GOV=unknown
    gfile=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
    [ -r "$gfile" ] && GOV=$(cat "$gfile")
    PLATFORM="${WM_OPTIONS:-unknown}"

    # ---- build the SPUME leaf + lib in THIS environment -------------------
    command -v blockMesh  > /dev/null || { echo "comparison: blockMesh not built in this env" >&2; exit 1; }
    command -v pimpleFoam > /dev/null || { echo "comparison: stock pimpleFoam not built in this env" >&2; exit 1; }

    rm -rf "$WORK"; mkdir -p "$WORK/reps"

    echo "comparison: building spumePimpleFoam + libspumeFoamSolvers for $PLATFORM"
    wmake "$ROOT/applications/spumePimpleFoam" > "$WORK/log.wmake.leaf" 2>&1 ||
        { echo "comparison: spumePimpleFoam build failed, see $WORK/log.wmake.leaf" >&2; exit 1; }
    wmake libso "$ROOT/applications/libs/spumeFoamSolvers" > "$WORK/log.wmake.lib" 2>&1 ||
        { echo "comparison: libspumeFoamSolvers build failed, see $WORK/log.wmake.lib" >&2; exit 1; }

    # ---- build parity: every artifact must live under the active platform -
    STOCK_BIN=$(command -v pimpleFoam)
    SPUME_BIN=$(command -v spumePimpleFoam) ||
        { echo "comparison: spumePimpleFoam not on PATH after build" >&2; exit 1; }
    LIBSO="$FOAM_USER_LIBBIN/libspumeFoamSolvers.so"
    [ -e "$LIBSO" ] || { echo "comparison: $LIBSO missing after build" >&2; exit 1; }

    for artifact in "$STOCK_BIN" "$SPUME_BIN" "$LIBSO"; do
        case "$artifact" in
            */platforms/"$PLATFORM"/*) ;;
            *)
                echo "comparison: BUILD-PARITY FAIL — $artifact is not under platform $PLATFORM" >&2
                echo "comparison: refusing to time binaries from different builds" >&2
                exit 1
                ;;
        esac
    done

    # ---- fvSolution: matched tight p, differing ONLY in the p solver ------
    # Comparison-harness settings on a COPIED case, not a defaults change: both
    # sides drive p to 1e-10 with no relative early-out so GAMG and spumePCG
    # reach the SAME solution (they otherwise early-out at different relTols and
    # every run would be VOID). U/k/epsilon are identical, so they cannot
    # introduce a difference. writePrecision 12 exposes the true agreement.
    write_fvsolution()  # $1 = case dir, $2 = p solver, $3 = extra p entries
    {
        cat > "$1/system/fvSolution" <<EOF
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    p
    {
        solver          $2;
        tolerance       1e-10;
        relTol          0;
        maxIter         5000;
        $3
    }
    pFinal { \$p; }
    "(U|k|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.05;
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

    cp -r "$CASE" "$WORK/base-stock"
    cp -r "$CASE" "$WORK/base-spume"
    # Stock side: GAMG-as-solver, or the Krylov-accelerated PCG+GAMG baseline.
    if [ "$SPUME_BASELINE" = "PCG-GAMG" ]; then
        write_fvsolution "$WORK/base-stock" PCG \
            "preconditioner { preconditioner GAMG; smoother DICGaussSeidel; }"
    else
        write_fvsolution "$WORK/base-stock" GAMG "smoother DICGaussSeidel;"
    fi
    write_fvsolution "$WORK/base-spume" spumePCG "preconditioner DIC; spumePreconditioner $SPUME_PRECOND; log 1;"
    # spume side loads the SPUME solver library.
    foamDictionary -entry libs -set "(spumeFoamSolvers)" "$WORK/base-spume/system/controlDict" > /dev/null 2>&1
    for d in base-stock base-spume; do
        foamDictionary -entry writePrecision -set 12 "$WORK/$d/system/controlDict" > /dev/null 2>&1
    done

    # Mesh once per side — EXCLUDED from the timed region.
    ( cd "$WORK/base-stock" && blockMesh > log.blockMesh 2>&1 )
    ( cd "$WORK/base-spume" && blockMesh > log.blockMesh 2>&1 )

    # ---- timing helpers ---------------------------------------------------
    # /proc/stat "cpu " fields: user nice system idle iowait irq softirq steal
    # guest guest_nice; steal is the 8th value. STEAL_NOW/TOTAL_NOW are set as
    # side effects (POSIX has no return-by-value).
    STEAL_NOW=0; TOTAL_NOW=0
    read_cpu()
    {
        read -r _cpu u ni sy id io ir so st gu gn < /proc/stat
        gu=${gu:-0}; gn=${gn:-0}
        STEAL_NOW=$st
        TOTAL_NOW=$((u + ni + sy + id + io + ir + so + st + gu + gn))
    }

    TOT_STEAL=0; TOT_TOTAL=0
    time_one_rep()  # $1 side  $2 bin  $3 idx  $4 basedir  $5 timesfile  $6 count(0=warmup)
    {
        rep="$WORK/reps/$1.$3"
        rm -rf "$rep"; cp -r "$4" "$rep"
        read_cpu; ps=$STEAL_NOW; pt=$TOTAL_NOW
        a=$(date +%s.%N)
        ( cd "$rep" && "$2" > log.solver 2>&1 )
        b=$(date +%s.%N)
        read_cpu
        if [ "$6" -ne 0 ]; then
            TOT_STEAL=$((TOT_STEAL + STEAL_NOW - ps))
            TOT_TOTAL=$((TOT_TOTAL + TOTAL_NOW - pt))
            el=$(awk -v a="$a" -v b="$b" "BEGIN{printf \"%.6f\", b - a}")
            printf "%s\n" "$el" >> "$5"
        fi
    }

    STOCK_TIMES="$WORK/times.stock"; : > "$STOCK_TIMES"
    SPUME_TIMES="$WORK/times.spume"; : > "$SPUME_TIMES"

    # ---- warm-up (EXCLUDED) ----------------------------------------------
    echo "comparison: warm-up (excluded from timing)"
    time_one_rep stock pimpleFoam       warmup "$WORK/base-stock" /dev/null 0
    time_one_rep spume spumePimpleFoam  warmup "$WORK/base-spume" /dev/null 0

    # ---- interleaved timed reps ------------------------------------------
    i=1
    while [ "$i" -le "$REPS" ]; do
        echo "comparison: timed rep $i / $REPS"
        time_one_rep stock pimpleFoam      "$i" "$WORK/base-stock" "$STOCK_TIMES" 1
        time_one_rep spume spumePimpleFoam "$i" "$WORK/base-spume" "$SPUME_TIMES" 1
        i=$((i + 1))
    done

    # First timed rep of each side is kept as the representative for the verdict.
    STOCK_VERDICT="$WORK/reps/stock.1"
    SPUME_VERDICT="$WORK/reps/spume.1"

    # ---- statistics (awk: median + CV%) ----------------------------------
    stats()  # $1 = times file -> "median cv_pct"
    {
        awk "{ v[NR] = \$1; sum += \$1 }
             END {
                 n = NR;
                 for (p = 1; p <= n; p++)
                     for (q = p + 1; q <= n; q++)
                         if (v[q] < v[p]) { t = v[p]; v[p] = v[q]; v[q] = t }
                 if (n % 2) med = v[(n + 1) / 2];
                 else       med = (v[n / 2] + v[n / 2 + 1]) / 2;
                 mean = sum / n;
                 ss = 0;
                 for (p = 1; p <= n; p++) { d = v[p] - mean; ss += d * d }
                 sd = (n > 1) ? sqrt(ss / (n - 1)) : 0;
                 cv = (mean > 0) ? 100 * sd / mean : 0;
                 printf \"%.4f %.2f\", med, cv
             }" "$1"
    }

    STOCK_STATS=$(stats "$STOCK_TIMES")
    SPUME_STATS=$(stats "$SPUME_TIMES")
    STOCK_MED=${STOCK_STATS% *}; STOCK_CV=${STOCK_STATS#* }
    SPUME_MED=${SPUME_STATS% *}; SPUME_CV=${SPUME_STATS#* }

    RATIO=$(awk -v s="$SPUME_MED" -v k="$STOCK_MED" \
        "BEGIN{ printf \"%.3f\", (k > 0) ? s / k : 0 }")
    STEAL_PCT=$(awk -v s="$TOT_STEAL" -v t="$TOT_TOTAL" \
        "BEGIN{ printf \"%.4f\", (t > 0) ? 100 * s / t : 0 }")
    STOCK_FLAG=$(awk -v c="$STOCK_CV" "BEGIN{ print (c > 5.0) ? \"UNSTABLE\" : \"ok\" }")
    SPUME_FLAG=$(awk -v c="$SPUME_CV" "BEGIN{ print (c > 5.0) ? \"UNSTABLE\" : \"ok\" }")

    # ---- equivalence verdict (the wall-times are meaningless without it) --
    echo
    echo "comparison: checking the two solutions are within the equivalence class"
    if python3 "$ROOT/tests/regression/check_equivalence.py" \
        "$STOCK_VERDICT" "$SPUME_VERDICT" \
        --mode=reorder-tolerance \
        --rtol="$SPUME_EQUIV_RTOL" --atol="$SPUME_EQUIV_ATOL"; then
        VERDICT="WITHIN EQUIVALENCE CLASS"
        rc=0
    else
        VERDICT="VOID (fields outside equivalence class)"
        rc=1
    fi

    # ---- report -----------------------------------------------------------
    cat <<EOF

================ SPUME vs stock wall-time comparison ================
provenance
  hostname          : $HOSTN
  build platform    : $PLATFORM
  cpufreq governor  : $GOV
  OMP env           : OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES
  case              : $CASE
  timed reps        : $REPS (interleaved A/B, +1 warm-up excluded)
  stock p-solver    : $([ "$SPUME_BASELINE" = "PCG-GAMG" ] && echo "PCG + GAMG-preconditioner (DICGaussSeidel) — Krylov-accelerated" || echo "GAMG-as-solver (DICGaussSeidel)")
  spume p-solver    : spumePCG (spumePreconditioner=$SPUME_PRECOND)
  equiv tolerances  : reorder-tolerance rtol=$SPUME_EQUIV_RTOL atol=$SPUME_EQUIV_ATOL
  stock binary      : $STOCK_BIN
  spume binary      : $SPUME_BIN
  spume solver lib  : $LIBSO

results (median wall time, seconds; timed region = solver run only)
  stock             : ${STOCK_MED}s   CV ${STOCK_CV}%  [$STOCK_FLAG]
  spume             : ${SPUME_MED}s   CV ${SPUME_CV}%  [$SPUME_FLAG]
  ratio SPUME/stock : ${RATIO}x   (>1 means SPUME is slower)
  steal-time        : ${STEAL_PCT}% of CPU jiffies across the timed region

  NOTE: spumePreconditioner=gamgFP32 reuses the cached OpenFOAM GAMGAgglomeration
  hierarchy and runs the SPUME FP32 Chebyshev V-cycle on it — measured at PARITY
  with stock GAMG (~1.01x) on pitzDaily, within measurement noise. The
  self-coarsening amgFP32 is ~1.07x; weaker preconditioners (jacobi,
  chebyshevFP32) are slower. No speedup is claimed on this small, partly
  cache-resident case; the mixed-precision bandwidth win needs a large case.

equivalence verdict : $VERDICT
====================================================================
EOF

    if [ "$rc" -ne 0 ]; then
        echo "comparison: COMPARISON VOID — the solvers disagree, wall-time numbers are meaningless" >&2
        exit "$rc"
    fi
'
