#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
#
# Apply the ordered patch stack in patches/ to the vendored tree (ADR-0001).
# Idempotence: refuses to apply a patch that no longer applies cleanly —
# fix the patch, never the vendored file.

set -eu

cd "$(git rev-parse --show-toplevel)"

applied=0
for p in patches/[0-9]*.patch; do
    [ -e "$p" ] || {
        echo "apply_patches: no patches in stack"
        exit 0
    }
    echo "applying $p"
    git apply --check "$p" || {
        echo "apply_patches: $p does not apply cleanly" >&2
        exit 1
    }
    git apply "$p"
    applied=$((applied + 1))
done
echo "apply_patches: $applied patch(es) applied"
