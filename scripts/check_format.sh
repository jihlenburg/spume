#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
#
# Verify that all SPUME-owned C++ sources match the repo .clang-format.
# Override the binary with CLANG_FORMAT (e.g. CLANG_FORMAT=clang-format-18).

set -u

CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"

files=$(git ls-files '*.cpp' '*.hpp' '*.h' | grep -v -e '^vendor/' -e '^tests/thirdparty/')
if [ -z "$files" ]; then
    echo "format check: no files"
    exit 0
fi

# shellcheck disable=SC2086 # word splitting of the file list is intended
if "$CLANG_FORMAT" --dry-run -Werror $files; then
    echo "format check: OK ($("$CLANG_FORMAT" --version))"
else
    echo "format check: FAILED — run: $CLANG_FORMAT -i <files>" >&2
    exit 1
fi
