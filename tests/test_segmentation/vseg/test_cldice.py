"""Import-smoke test for ``liom_toolkit/segmentation/vseg/cldice.py``.

Verifies that ``cl_dice`` imports cleanly and is callable now that
``skeletonize_3d`` has been replaced with ``skeletonize`` (which handles
both 2D and 3D inputs since skimage 0.26).

This depends on the conftest narrowing: ``vseg.prediction`` stays mocked
(so the barrel ``vseg/__init__.py`` does not crash on ``import torch``)
while the ``vseg`` package mock carries a real ``__path__`` pointing at the
on-disk ``vseg/`` directory, so Python can locate ``cldice.py`` as a real
submodule.
"""


def test_cldice_imports():
    """cldice module imports cleanly after the skeletonize migration."""
    from liom_toolkit.segmentation.vseg.cldice import cl_dice

    assert callable(cl_dice)
