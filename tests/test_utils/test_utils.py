"""Pure-function smoke tests for ``liom_toolkit/utils/utils.py``.

Mirrors the package layout (``tests/test_utils/test_utils.py``) and the
known-answer style established by ``tests/test_canary.py``. Covers the two
core-dep-only pure functions in ``utils.py``: ``fix_even`` (integer parity
fix) and ``convert_to_png_for_saving`` (linear rescale to uint8). No I/O, no
mocking, no optional deps.
"""

import numpy as np

from liom_toolkit.utils.utils import convert_to_png_for_saving, fix_even


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


class TestConvertToPngForSaving:
    """Known-answer tests for convert_to_png_for_saving.

    Uses a non-constant image so the (max - min) divisor is non-zero. The
    constant-image divide-by-zero boundary is intentionally NOT exercised
    here; it is left to a later bug audit.
    """

    def test_uint16_image_rescaled_to_uint8(self):
        img = np.array([[0, 100], [200, 300]], dtype=np.uint16)
        expected = np.array([[0, 85], [170, 255]], dtype=np.uint8)
        out = convert_to_png_for_saving(img)
        assert out.dtype == np.uint8
        assert np.array_equal(out, expected)
