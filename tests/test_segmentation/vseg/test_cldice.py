"""Import-smoke test for ``liom_toolkit/segmentation/vseg/cldice.py``.

This is the whole-test xfail characterization of the ``skeletonize_3d``
removal (BUG-01/DEP-01): ``skimage`` 0.26 dropped ``skeletonize_3d`` from
``skimage.morphology`` (PR #7572), so the module-top
``from skimage.morphology import skeletonize, skeletonize_3d`` raises
``ImportError`` and the ``cl_dice`` symbol cannot be imported.

The test body is the *post-Phase-3* correctness assertion (import succeeds,
``cl_dice`` is callable). It is marked ``xfail(strict=True, raises=ImportError)``
so the current broken state is reported as ``XFAIL`` (not hidden via ``skip``),
and the moment Phase 3 replaces ``skeletonize_3d`` with ``skeletonize`` the
import succeeds, the test ``XPASS``es, ``strict=True`` turns that into a
hard ``FAILURE``, and the developer is forced to remove the marker — proving
the fix happened instead of silently passing.

This depends on the conftest narrowing from Plan 01: ``vseg.prediction`` stays
mocked (so the barrel ``vseg/__init__.py`` does not crash on ``import torch``)
while the ``vseg`` package mock carries a real ``__path__`` pointing at the
on-disk ``vseg/`` directory, so Python can locate ``cldice.py`` as a real
submodule and the ``skeletonize_3d`` ``ImportError`` is the one that surfaces
(not ``'liom_toolkit.segmentation.vseg' is not a package`` and not
``ModuleNotFoundError: No module named 'torch'``).
"""

import pytest


@pytest.mark.xfail(
    strict=True,
    raises=ImportError,
    reason="BUG-01/DEP-01: skeletonize_3d removed in skimage 0.26",
)
def test_cldice_imports():
    """Post-Phase-3 correctness: cldice module imports cleanly."""
    from liom_toolkit.segmentation.vseg.cldice import cl_dice

    assert callable(cl_dice)
