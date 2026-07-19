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

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            ce.fields_equal(b"a", b"b", "nonsense")


def _field(values, bc="zeroGradient"):
    """Minimal OpenFOAM ascii scalar field with the given internal values."""
    body = "\n".join(str(v) for v in values)
    return (
        BANNER
        + "FoamFile\n{\n    class volScalarField;\n    object p;\n}\n"
        + "dimensions [0 2 -2 0 0 0 0];\n"
        + f"internalField nonuniform List<scalar>\n{len(values)}\n(\n{body}\n)\n;\n"
        + "boundaryField\n{\n    inlet { type " + bc + "; }\n}\n"
    ).encode()


class ReorderTolerance(unittest.TestCase):
    def test_identical_passes(self):
        a = _field([1.0, 2.0, 3.0])
        self.assertTrue(ce.fields_equal(a, a, "reorder-tolerance"))

    def test_within_tolerance_passes(self):
        a = _field([1.0, 2.0, 3.0])
        b = _field([1.0, 2.0 + 2e-8, 3.0])  # ~1e-8 relative, under default rtol 1e-6
        self.assertTrue(ce.fields_equal(a, b, "reorder-tolerance"))

    def test_out_of_tolerance_fails(self):
        a = _field([1.0, 2.0, 3.0])
        b = _field([1.0, 2.02, 3.0])  # 1e-2 relative, well over tolerance
        self.assertFalse(ce.fields_equal(a, b, "reorder-tolerance"))

    def test_structural_difference_fails_even_within_tol(self):
        a = _field([1.0, 2.0, 3.0], bc="zeroGradient")
        b = _field([1.0, 2.0, 3.0], bc="fixedValue")  # same numbers, different structure
        self.assertFalse(ce.fields_equal(a, b, "reorder-tolerance"))

    def test_different_value_count_fails(self):
        a = _field([1.0, 2.0, 3.0])
        b = _field([1.0, 2.0])
        self.assertFalse(ce.fields_equal(a, b, "reorder-tolerance"))


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
