#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
#
# Verify that every SPUME-owned source/build file carries the SPDX header.
# Vendored code (vendor/, tests/thirdparty/) is exempt: upstream headers are
# retained there instead.

set -u

fail=0
for f in $(git ls-files \
    '*.cpp' '*.hpp' '*.h' '*.c' '*.sh' '*.py' '*.cmake' '*.yml' '*.yaml' \
    'CMakeLists.txt' '*/CMakeLists.txt'); do
    case "$f" in
    vendor/* | tests/thirdparty/*) continue ;;
    esac
    if ! head -n 5 "$f" | grep -q 'SPDX-License-Identifier: GPL-3.0-or-later'; then
        echo "missing SPDX header: $f" >&2
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "SPDX check: OK"
fi
exit "$fail"
