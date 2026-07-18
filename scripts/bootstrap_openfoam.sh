#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
#
# Bootstrap a build of the vendored OpenFOAM tree against system libraries,
# without editing anything under vendor/ (ADR-0001, ADR-0014).
#
# What it does:
#   1. Verifies the host toolchain (compilers, MPI, flex/bison, scotch dev).
#   2. Works around the pruned-tree quirk: top-level Allwmake ends with
#      `cd modules && wmake -all`; with modules/ pruned it exits non-zero
#      after a successful build. We create an untracked modules/ directory.
#   3. Generates a repo-local FOAM_CONFIG_ETC override that points OpenFOAM's
#      scotch config at the system install (scotch-system), so no ThirdParty
#      tree is needed. The override shadows vendor/openfoam/etc/config.sh/scotch
#      via foamEtcFile search order WITHOUT touching the vendored file.
#   4. Optionally installs missing OS packages (--install-deps) and/or runs
#      the build (--build).
#
# The build uses the system OpenMPI (WM_MPLIB=SYSTEMOPENMPI, the vendored
# default) and the system Scotch/PT-Scotch. Nothing here is committed by the
# build itself; generated config lives in an ignored scratch dir.
#
# Usage:
#   scripts/bootstrap_openfoam.sh [--install-deps] [--build] [--jobs N]
#
# After a successful run you can source the emitted env file to get a live
# OpenFOAM shell for the compat shim / leaf app work:
#   source .openfoam-config/envrc

set -euo pipefail

# --- locate the repo and the vendored tree ----------------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
foam_dir="$repo_root/vendor/openfoam"
config_root="$repo_root/.openfoam-config"   # gitignored scratch (see below)
config_etc="$config_root/etc"

[ -f "$foam_dir/etc/bashrc" ] || {
    echo "error: vendored OpenFOAM not found at $foam_dir" >&2
    exit 1
}

# --- options ----------------------------------------------------------------
do_install=0
do_build=0
jobs="$(nproc)"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --install-deps) do_install=1 ;;
        --build)        do_build=1 ;;
        --jobs)         jobs="$2"; shift ;;
        --jobs=*)       jobs="${1#*=}" ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# --- dependency check -------------------------------------------------------
# Package names are Debian/Ubuntu; adjust for other distros.
missing_pkgs=""
need() {  # need <command> <apt-package>
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "  missing: $1  (apt: $2)"
        case " $missing_pkgs " in *" $2 "*) ;; *) missing_pkgs="$missing_pkgs $2" ;; esac
    fi
}

echo "==> Checking host toolchain"
need g++      g++
need clang++  clang
need mpicc    libopenmpi-dev
need mpirun   openmpi-bin
need flex     flex
need bison    bison
need m4       m4
need cmake    cmake
need make     make

# Scotch is used through system dev headers (no ThirdParty tree).
if [ ! -e /usr/include/scotch/scotch.h ] && [ ! -e /usr/include/scotch.h ]; then
    echo "  missing: scotch.h  (apt: libscotch-dev)"
    missing_pkgs="$missing_pkgs libscotch-dev"
fi
if [ ! -e /usr/include/scotch/ptscotch.h ] && [ ! -e /usr/include/ptscotch.h ]; then
    echo "  missing: ptscotch.h  (apt: libptscotch-dev)"
    missing_pkgs="$missing_pkgs libptscotch-dev"
fi

missing_pkgs="$(echo "$missing_pkgs" | xargs -n1 2>/dev/null | sort -u | xargs || true)"
if [ -n "$missing_pkgs" ]; then
    if [ "$do_install" -eq 1 ]; then
        echo "==> Installing missing packages: $missing_pkgs"
        sudo apt-get update -qq
        # shellcheck disable=SC2086
        sudo apt-get install -y $missing_pkgs
    else
        echo
        echo "Missing packages. Install them (needs sudo), then re-run:"
        echo "    sudo apt-get install -y$missing_pkgs"
        echo "or re-run this script with --install-deps"
        exit 1
    fi
else
    echo "    all required tools present"
fi

# --- pruned-tree quirk: empty modules/ so Allwmake exits 0 ------------------
# Untracked; keeps the vendored tree pristine (documented in vendor/README.md).
mkdir -p "$foam_dir/modules"

# --- repo-local FOAM_CONFIG_ETC override for system scotch ------------------
# foamEtcFile searches $FOAM_CONFIG_ETC before $projectDir/etc (bin/foamEtcFile),
# so this shadows the vendored config.sh/scotch without editing vendor/.
mkdir -p "$config_etc/config.sh"
cat > "$config_etc/config.sh/scotch" <<'EOF'
# SPUME-generated: point OpenFOAM at the system Scotch/PT-Scotch install.
# Shadows vendor/openfoam/etc/config.sh/scotch via FOAM_CONFIG_ETC — the
# vendored file is never edited (ADR-0001).
SCOTCH_VERSION=scotch-system
export SCOTCH_ARCH_PATH=/usr
EOF

# Emit a small env file so later interactive work gets the same OpenFOAM shell.
cat > "$config_root/envrc" <<EOF
# SPUME-generated. Source this for a live OpenFOAM shell against the vendored
# tree with system MPI + system Scotch:  source .openfoam-config/envrc
export FOAM_CONFIG_ETC="$config_etc"
source "$foam_dir/etc/bashrc"
EOF

# Keep the scratch dir out of version control.
if [ -f "$repo_root/.gitignore" ] && ! grep -qxF '.openfoam-config/' "$repo_root/.gitignore"; then
    printf '\n# Bootstrap-generated OpenFOAM config/env (system libs)\n.openfoam-config/\n' \
        >> "$repo_root/.gitignore"
fi

echo "==> Config override written: $config_etc/config.sh/scotch (scotch-system)"
echo "    Env file:               $config_root/envrc"

# --- build ------------------------------------------------------------------
if [ "$do_build" -eq 0 ]; then
    echo
    echo "Setup complete. To build:"
    echo "    scripts/bootstrap_openfoam.sh --build --jobs $jobs"
    exit 0
fi

echo "==> Building vendored OpenFOAM (jobs=$jobs, WM_MPLIB=SYSTEMOPENMPI)"
export FOAM_CONFIG_ETC="$config_etc"
# OpenFOAM's etc/bashrc and Allwmake are neither errexit- nor nounset-safe
# (they source scripts that legitimately return non-zero); relax both.
set +eu
# shellcheck disable=SC1090
source "$foam_dir/etc/bashrc"
echo "    WM_PROJECT_DIR=$WM_PROJECT_DIR"
echo "    WM_MPLIB=$WM_MPLIB  WM_COMPILER=$WM_COMPILER"

cd "$foam_dir"
# -k: keep going so one failure does not mask overall progress in the log.
./Allwmake -j "$jobs" -s -k
build_rc=$?
echo "==> Allwmake finished (exit $build_rc)"
exit "$build_rc"
