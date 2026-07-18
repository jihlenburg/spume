#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
#
# Source (do NOT execute) this file to get an OpenFOAM environment for
# building/running SPUME leaf applications (applications/) and contract tests
# (tests/contract/). It also exports SPUME_PROJECT_DIR, which leaf-app
# Make/options use to locate the compat shim (src/compat) and the vendored
# solver fragments.
#
# Usage:
#     . scripts/foam-env.sh
#     cd applications/spumePimpleFoam && wmake
#
# Override the OpenFOAM install by exporting OPENFOAM_BASHRC first:
#     OPENFOAM_BASHRC=/path/to/OpenFOAM-vXXXX/etc/bashrc . scripts/foam-env.sh
#
# For a clean-runner build from the vendored tree instead of a pre-built
# install, use scripts/bootstrap_openfoam.sh (that is CI's path).

SPUME_PROJECT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export SPUME_PROJECT_DIR

: "${OPENFOAM_BASHRC:=$HOME/OpenFOAM/OpenFOAM-v2606/etc/bashrc}"

if [ ! -f "$OPENFOAM_BASHRC" ]; then
    echo "foam-env: OpenFOAM bashrc not found at: $OPENFOAM_BASHRC" >&2
    echo "foam-env: set OPENFOAM_BASHRC to a v2606+ install, or build the" >&2
    echo "foam-env: vendored tree with scripts/bootstrap_openfoam.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
. "$OPENFOAM_BASHRC"

echo "foam-env: OpenFOAM $WM_PROJECT_VERSION  |  SPUME_PROJECT_DIR=$SPUME_PROJECT_DIR"
