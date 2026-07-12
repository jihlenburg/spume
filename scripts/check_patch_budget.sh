#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
#
# Enforce the ADR-0001 patch-stack budget: all patches/*.patch together
# stay under 2,000 lines, and vendor/ is never modified outside the stack
# (vendored files must match a pristine import plus applied patches; this
# check covers the budget half — pristineness is enforced by review and
# the wholesale-replacement update rule in vendor/README.md).

set -eu

cd "$(git rev-parse --show-toplevel)"

budget=2000
total=0
for p in patches/[0-9]*.patch; do
    [ -e "$p" ] || break
    lines=$(wc -l < "$p")
    total=$((total + lines))
done

echo "patch budget: $total / $budget lines"
if [ "$total" -gt "$budget" ]; then
    echo "patch budget EXCEEDED (ADR-0001 hard budget)" >&2
    exit 1
fi
