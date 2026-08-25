"""Shared pytest fixtures for the liom-toolkit test suite.

Fixtures are generated programmatically (no binary fixtures committed) and
match the synthetic-volume patterns documented in the codebase testing map.
They are shipped in Phase 1 so later phases can consume them without
redefinition.
"""

import os
import sys
import types

import numpy as np
import pytest

# The package's barrel ``__init__.py`` files create an import chain that
# pulls in optional deps and broken imports even when importing a core-only
# module such as ``utils.utils``. Specifically:
#
#   utils.__init__ → utils.io → segmentation.__init__
#     → segmentation.volume_segmentation (bare ``import ants``)
#     → segmentation.vseg               (torch, sklearn, removed zarr.convenience)
#
# Until the lazy-import migration lands and dependency adaptations are applied,
# pre-populate ``sys.modules`` with lightweight stand-ins for the two
# problematic segmentation submodules so the barrel import chain succeeds on
# the base install. The canary only exercises ``fix_even`` (core deps: os,
# numpy) — it never calls into ``volume_segmentation`` or ``vseg``.
#
# Tests that actually exercise an optional dep should gate with
# ``pytest.importorskip("ants")`` / ``"torch"`` and remove the corresponding
# sys.modules entry inside the test body so the real import is attempted.

# Note: ``liom_toolkit.segmentation.vseg.utils`` and
# ``liom_toolkit.segmentation.vseg.cldice`` are intentionally NOT mocked here.
# They are imported for real so Phase-2 tests can exercise ``calculate_metrics``,
# ``add_patch_to_empty_array`` (via ``importorskip("torch")``/``"sklearn"``) and
# characterize the ``cldice`` ``skeletonize_3d`` ImportError. The remaining 7
# entries stay mocked so the barrel ``__init__.py`` chain + base-install import
# smoke test still pass without optional deps.
_VOL_SEG_NAMES = [
    "liom_toolkit.segmentation.volume_segmentation",
    "liom_toolkit.segmentation.vseg",
    "liom_toolkit.segmentation.vseg.prediction",
    "liom_toolkit.segmentation.vseg.dataset",
    "liom_toolkit.segmentation.vseg.model",
    "liom_toolkit.segmentation.vseg.loss",
    "liom_toolkit.segmentation.vseg.validation",
]

for _name in _VOL_SEG_NAMES:
    if _name not in sys.modules:
        _mock = types.ModuleType(_name)
        # Expose no-op callables so star-imports in segmentation.__init__
        # and vseg.__init__ pick up the expected names without error.
        _mock.__all__ = []
        sys.modules[_name] = _mock

# Provide the names that the barrel ``__init__.py`` files expect to
# star-import, so ``from .volume_segmentation import *`` and
# ``from .vseg import *`` succeed. ``__all__`` must list them explicitly,
# otherwise the star-import picks up nothing.
_vol_seg = sys.modules["liom_toolkit.segmentation.volume_segmentation"]
_vol_seg.segment_3d = lambda *a, **k: None
_vol_seg.__all__ = ["segment_3d"]

_vseg = sys.modules["liom_toolkit.segmentation.vseg"]
_vseg.predict_one = lambda *a, **k: None
_vseg.predict_volume = lambda *a, **k: None
_vseg.__all__ = ["predict_one", "predict_volume"]

# Give the mocked ``vseg`` package a real ``__path__`` pointing at the actual
# ``liom_toolkit/segmentation/vseg/`` directory. Without this, importing a real
# submodule (``from liom_toolkit.segmentation.vseg.cldice import cl_dice``)
# raises ``'liom_toolkit.segmentation.vseg' is not a package`` instead of the
# intended ``ImportError: cannot import name 'skeletonize_3d'``. Setting
# ``__path__`` lets Python locate ``cldice.py`` / ``utils.py`` as real
# submodules of the mocked package.
_vseg_pkg = sys.modules["liom_toolkit.segmentation.vseg"]
_vseg_pkg.__path__ = [
    os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "liom_toolkit",
            "segmentation",
            "vseg",
        )
    )
]


@pytest.fixture
def synthetic_volume() -> np.ndarray:
    """A 64x64x64 uint16 volume with a bright sphere.

    A bright sphere (radius**2 <= 100, centered at (32, 32, 32), value 1000)
    on a dark (0) background. Used by segmentation / mask tests.
    """
    vol = np.zeros((64, 64, 64), dtype=np.uint16)
    zz, yy, xx = np.ogrid[:64, :64, :64]
    vol[(zz - 32) ** 2 + (yy - 32) ** 2 + (xx - 32) ** 2 <= 100] = 1000
    return vol


@pytest.fixture
def bimodal_2d() -> np.ndarray:
    """A 128x128 uint8 image with a bright square on a dark background.

    A bright square (value 200) at [32:96, 32:96] on a dark (0) background.
    Used by 2D segmentation / threshold tests.
    """
    img = np.zeros((128, 128), dtype=np.uint8)
    img[32:96, 32:96] = 200
    return img
