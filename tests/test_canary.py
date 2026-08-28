"""Canary smoke test proving the pytest harness works end-to-end.

Exercises only core dependencies (os, numpy) by importing and calling
``fix_even`` from ``liom_toolkit.utils.utils``. This confirms the package
imports cleanly on the base install (no extras) on both Python 3.12 and 3.14,
and that conftest discovery, marker registration (--strict-markers), xdist
(-n auto --dist=loadscope), and coverage collection (--cov=liom_toolkit) all
work together. The full test suite is a later phase; this is the single
tracer test that gates the rest of the foundation work.
"""

from liom_toolkit.utils.utils import fix_even


class TestFixEven:
    """Known-answer tests for fix_even (liom_toolkit/utils/utils.py)."""

    def test_even_number_becomes_odd(self):
        assert fix_even(4) == 5

    def test_odd_number_unchanged(self):
        assert fix_even(5) == 5

    def test_zero_becomes_one(self):
        assert fix_even(0) == 1

    def test_negative_even(self):
        assert fix_even(-4) == -3
