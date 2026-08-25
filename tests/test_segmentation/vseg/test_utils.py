"""Known-answer tests for ``liom_toolkit/segmentation/vseg/utils.py``.

Both tests target functions whose module (``vseg/utils.py``) has module-top
imports of ``torch`` and ``sklearn.metrics``. Per D-02 we do NOT stub those
deps; instead each test calls ``pytest.importorskip`` BEFORE importing from
``vseg.utils`` so the tests run for real on the 3.12-full CI leg (where the
``ai`` extra is installed) and cleanly skip on the 3.14-core leg (where it is
not). The importorskip must precede the ``from liom_toolkit...`` statement —
importing the module before the skips would crash on missing torch.

Coverage:

* ``calculate_metrics`` — known-answer for ``[f1, recall, accuracy, jaccard,
  precision]`` on a small binary mask pair (returns 5 floats; f1 asserted via
  ``pytest.approx``).
* ``add_patch_to_empty_array`` — inserts a 4x4 patch into an 8x8 zero array
  at coords (0, 0) with ``overlap=0``. The ``overlap=0`` boundary avoids the
  multi-branch overlap-blending logic (utils.py:134-172), keeping the
  known-answer deterministic: the patch region equals ``pred_y`` and the
  untouched region stays zero.
"""

import numpy as np
import pytest


def test_calculate_metrics_known_answer():
    """calculate_metrics returns [f1, recall, accuracy, jaccard, precision]."""
    pytest.importorskip("torch")  # vseg/utils.py module-top imports torch
    pytest.importorskip("sklearn")  # calculate_metrics uses sklearn.metrics
    from liom_toolkit.segmentation.vseg.utils import calculate_metrics

    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    result = calculate_metrics(y_true, y_pred)

    assert len(result) == 5
    # f1 is result[0]; verified value from 02-RESEARCH.md
    assert result[0] == pytest.approx(0.6667, abs=1e-3)


def test_add_patch_to_empty_array_inserts_patch():
    """add_patch_to_empty_array writes pred_y into the patch region; overlap=0
    avoids the blending branches so the untouched region stays zero."""
    pytest.importorskip("torch")  # vseg/utils.py module-top imports torch
    from liom_toolkit.segmentation.vseg.utils import add_patch_to_empty_array

    inference = np.zeros((8, 8), dtype=np.float32)
    pred_y = np.ones((4, 4), dtype=np.float32)

    result = add_patch_to_empty_array(
        inference,
        pred_y,
        coords=(0, 0),
        stride=4,
        overlap=0,
        size=(4, 4),
    )

    # Patch region (0:4, 0:4) equals pred_y (all ones)
    assert np.array_equal(result[0:4, 0:4], pred_y)
    # Untouched region (4:8, 4:8) remains all zeros
    assert np.array_equal(result[4:8, 4:8], np.zeros((4, 4), dtype=np.float32))
