#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
"""Assert solver-log iteration counts stay inside recorded bands.

Contract-test philosophy (AGENTS.md numerics policy): iteration counts on
a fixed case are compared within a band, never exactly — rounding-level
drift moves them by a step or two; a changed GAMG agglomeration or solver
semantic moves them a lot. The bands file records the expectation:

    first_p_iters   <int>   <abs-slack>
    total_p_iters   <int>   <rel-slack, e.g. 0.10>
    max_continuity  <float>
    p_solves        <int>   0

Usage: check_bands.py <solver-log> <bands-file>
"""

import re
import sys


def fail(msg: str) -> None:
    print(f"check_bands: FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    log_path, bands_path = sys.argv[1], sys.argv[2]

    with open(log_path, encoding="ascii", errors="replace") as f:
        log = f.read()

    p_iters = [
        int(m)
        for m in re.findall(
            r"GAMG:\s+Solving for p,.*No Iterations (\d+)", log
        )
    ]
    if not p_iters:
        fail("no GAMG p-solves found in log")

    cont = [
        float(m)
        for m in re.findall(r"continuity errors.*?max:\s*([0-9eE+.-]+)", log)
    ]
    if not cont:
        fail("no continuity-error lines found in log")

    bands = {}
    with open(bands_path, encoding="ascii") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                key, *vals = line.split()
                bands[key] = vals

    first, first_slack = int(bands["first_p_iters"][0]), int(bands["first_p_iters"][1])
    if abs(p_iters[0] - first) > first_slack:
        fail(f"first p-solve iterations {p_iters[0]} outside {first}±{first_slack}")

    total, rel = int(bands["total_p_iters"][0]), float(bands["total_p_iters"][1])
    got_total = sum(p_iters)
    if abs(got_total - total) > rel * total:
        fail(f"total p iterations {got_total} outside {total}±{rel:.0%}")

    nsolves = int(bands["p_solves"][0])
    if len(p_iters) != nsolves:
        fail(f"{len(p_iters)} p-solves, expected exactly {nsolves}")

    max_cont = float(bands["max_continuity"][0])
    worst = max(abs(c) for c in cont)
    if worst > max_cont:
        fail(f"worst continuity error {worst} > {max_cont}")

    print(
        f"check_bands: OK (first {p_iters[0]}, total {got_total} over "
        f"{len(p_iters)} solves, worst continuity {worst:.3e})"
    )


if __name__ == "__main__":
    main()
