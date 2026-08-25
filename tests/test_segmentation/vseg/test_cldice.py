"""Import-smoke test for ``liom_toolkit/segmentation/vseg/cldice.py``.

``skimage`` 0.26 dropped ``skeletonize_3d`` from ``skimage.morphology``
(PR #7572); ``cldice.py`` now imports ``skeletonize`` (which handles both
2D and 3D inputs) and the module imports cleanly. The previous
``xfail(strict=True, raises=ImportError)`` characterization has been
removed now that the migration is complete.

This depends on the conftest narrowing from Plan 01: ``vseg.prediction`` stays
mocked (so the barrel ``vseg/__init__.py`` does not crash on ``import torch``)
while the ``vseg`` package mock carries a real ``__path__`` pointing at the
on-disk ``vseg/`` directory, so Python can locate ``cldice.py`` as a real
submodule.
"""


def test_cldice_imports():
    """cldice module imports cleanly after the skeletonize migration."""
    from liom_toolkit.segmentation.vseg.cldice import cl_dice

    assert callable(cl_dice)
