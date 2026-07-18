#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
"""Bitwise field/log equivalence comparator for SPUME leaf applications.

M1 proves spumePimpleFoam produces results *byte-identical* to stock
pimpleFoam: the leaf runs the reference solvers unchanged, so on the same
case with the same decomposition every written field file must match to the
byte (rounding order is identical). Any difference is a plumbing bug, not a
numerics difference — hence bitwise, never a tolerance (AGENTS.md).

The comparator is written to outlive M1. `fields_equal` takes a mode:

    bitwise            -- M1: banner-stripped byte compare (this milestone)
    reorder-tolerance  -- M2+: rounding-reorder equivalence class (ADR-0002);
                          a documented stub until SPUME numerics diverge.

When M2 introduces mixed-precision solvers the fields will differ at the
rounding-reorder level, and this same harness flips to `reorder-tolerance`
as its regression gate — the oracle is reused, not rebuilt.

CLI:
    check_equivalence.py <caseA> <caseB> [--mode bitwise]

Compares every field file in each common numeric time directory of the two
cases. Exit 0 iff all match; non-zero with a summary otherwise.
"""

import os
import re
import sys

# Leading OpenFOAM banner comment block: /*--- ... ---*/ at the top of file.
# Stripped defensively so an (unexpected) banner-only difference never trips
# the byte compare; the field data below it is what must match.
_BANNER_RE = re.compile(r"^\s*/\*.*?\*/\s*", re.DOTALL)

_VALID_MODES = ("bitwise", "reorder-tolerance")


def strip_banner(text):
    """Remove the leading OpenFOAM banner comment block, if present."""
    return _BANNER_RE.sub("", text, count=1)


def fields_equal(a, b, mode):
    """Return True iff field-file contents a and b are equal under `mode`.

    a, b are bytes. `bitwise` decodes latin-1 (lossless for any byte),
    strips the banner from both, and compares the remainder exactly.
    """
    if mode == "bitwise":
        sa = strip_banner(a.decode("latin-1"))
        sb = strip_banner(b.decode("latin-1"))
        return sa == sb
    if mode == "reorder-tolerance":
        raise NotImplementedError(
            "reorder-tolerance mode lands in M2 with the mixed-precision "
            "solvers (ADR-0002); M1 is bitwise."
        )
    raise ValueError(f"unknown comparison mode: {mode!r} (expected one of {_VALID_MODES})")


def residual_lines(log):
    """Return the solver 'Solving for ...' lines, execution-time lines dropped."""
    return [ln.rstrip() for ln in log.splitlines() if "Solving for" in ln]


def _time_dirs(case):
    """Numeric time directories in a case, sorted by value (excludes 'constant')."""
    out = []
    for name in os.listdir(case):
        path = os.path.join(case, name)
        if not os.path.isdir(path):
            continue
        try:
            val = float(name)
        except ValueError:
            continue
        out.append((val, name))
    return [name for _, name in sorted(out)]


def _field_files(time_dir):
    return sorted(
        f for f in os.listdir(time_dir)
        if os.path.isfile(os.path.join(time_dir, f))
    )


def compare_cases(case_a, case_b, mode="bitwise"):
    """Compare all field files in common numeric time dirs. Returns list of diffs."""
    diffs = []
    times_a, times_b = _time_dirs(case_a), _time_dirs(case_b)
    if times_a != times_b:
        diffs.append(f"time directories differ: {times_a} vs {times_b}")
        return diffs
    if not times_a:
        diffs.append("no numeric time directories written — nothing to compare")
        return diffs
    for t in times_a:
        da, db = os.path.join(case_a, t), os.path.join(case_b, t)
        fa, fb = _field_files(da), _field_files(db)
        if fa != fb:
            diffs.append(f"time {t}: field set differs: {fa} vs {fb}")
            continue
        for f in fa:
            with open(os.path.join(da, f), "rb") as fh:
                ba = fh.read()
            with open(os.path.join(db, f), "rb") as fh:
                bb = fh.read()
            if not fields_equal(ba, bb, mode):
                diffs.append(f"time {t}: field {f} differs")
    return diffs


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    mode = "bitwise"
    for a in argv[1:]:
        if a.startswith("--mode="):
            mode = a.split("=", 1)[1]
    if len(args) != 2:
        print("usage: check_equivalence.py <caseA> <caseB> [--mode=bitwise]", file=sys.stderr)
        return 2
    diffs = compare_cases(args[0], args[1], mode)
    if diffs:
        print(f"equivalence FAIL ({mode}):", file=sys.stderr)
        for d in diffs:
            print(f"  - {d}", file=sys.stderr)
        return 1
    print(f"equivalence OK ({mode}): {args[0]} == {args[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
