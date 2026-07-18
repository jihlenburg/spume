#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
"""Self-test for the bitwise field/log equivalence comparator.

Dependency-free (stdlib unittest) to match the repo's python convention
(tests/contract/check_bands.py) and the no-new-dependencies rule.

Run: python3 tests/regression/test_check_equivalence.py
"""

import unittest

import check_equivalence as ce

BANNER = (
    "/*--------------------------------*- C++ -*----------------------------------*\\\n"
    "| =========                 |                                                 |\n"
    "\\*---------------------------------------------------------------------------*/\n"
)


class StripBanner(unittest.TestCase):
    def test_removes_leading_banner_block(self):
        self.assertEqual(ce.strip_banner(BANNER + "payload\n"), "payload\n")

    def test_noop_when_no_banner(self):
        self.assertEqual(ce.strip_banner("payload\n"), "payload\n")


class FieldsEqual(unittest.TestCase):
    def test_bitwise_true_when_only_banner_differs(self):
        a = (BANNER + "1.0 2.0 3.0\n").encode()
        b = (BANNER.replace("C++", "C--") + "1.0 2.0 3.0\n").encode()
        self.assertTrue(ce.fields_equal(a, b, "bitwise"))

    def test_bitwise_false_on_data_diff(self):
        a = (BANNER + "1.0\n").encode()
        b = (BANNER + "1.1\n").encode()
        self.assertFalse(ce.fields_equal(a, b, "bitwise"))

    def test_reorder_tolerance_is_stub(self):
        with self.assertRaises(NotImplementedError):
            ce.fields_equal(b"a", b"b", "reorder-tolerance")

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            ce.fields_equal(b"a", b"b", "nonsense")


class ResidualLines(unittest.TestCase):
    def test_keeps_only_solving_for_lines(self):
        log = (
            "Time = 0.005\n"
            "Solving for Ux, Initial residual = 1e-3, Final residual = 1e-7, No Iterations 2\n"
            "ExecutionTime = 0.5 s  ClockTime = 1 s\n"
        )
        self.assertEqual(
            ce.residual_lines(log),
            ["Solving for Ux, Initial residual = 1e-3, Final residual = 1e-7, No Iterations 2"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
