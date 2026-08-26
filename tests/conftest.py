"""Shared pytest fixtures for the liom-toolkit test suite.

Fixtures are generated programmatically (no binary fixtures committed) and
match the synthetic-volume patterns documented in the codebase testing map.
They are shipped in Phase 1 so later phases can consume them without
redefinition.
"""

import sys

import numpy as np
import PIL.Image
import pytest


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


@pytest.fixture
def synthetic_ants_image():
    """A tiny 8x8x8 float32 ANTsImage built via ants.from_numpy.

    The fixture body imports ``ants`` and ``numpy`` INSIDE the function so
    this conftest module imports cleanly on core-only installs (ants is an
    extra). Callers MUST call ``pytest.importorskip("ants")`` in the test
    body BEFORE requesting this fixture so the test skips cleanly on the
    3.14-core CI leg instead of erroring at fixture setup.
    """
    import ants

    arr = np.random.rand(8, 8, 8).astype("float32")
    # antspyx 0.6.x setDirection expects a 2D direction matrix
    # (Sequence[Sequence[float]]), not the raveled length-9 form accepted by
    # 0.5.x. Passing np.eye(3) keeps the identity direction in the 0.6.x shape.
    return ants.from_numpy(
        arr, spacing=(1, 1, 1), origin=(0, 0, 0), direction=np.eye(3)
    )


@pytest.fixture
def fake_ants():
    """Inject a MagicMock as the ``ants`` module in ``sys.modules`` for the test.

    Mirrors ``tests/test_conversion/test_use_custom_atlas.py:_install_fake_ants``.
    ``fake.from_numpy`` / ``fake.image_read`` / ``fake.reorient_image2`` return a
    MagicMock image whose ``.numpy()`` returns a small real numpy array. The fake
    is popped from ``sys.modules`` on teardown so it does not leak across tests.

    Note: mock-orchestration tests that use ``patch("liom_toolkit.registration.register.ants")``
    do NOT need this fixture — the per-module patch is the safer pattern for
    lazy-imported ants (AGENTS §5). ``fake_ants`` is for tests that exercise code
    doing a bare ``import ants`` at module top or inside a function-scope try/except
    where the per-module patch target is inconvenient.
    """
    from unittest.mock import MagicMock

    fake = MagicMock()
    fake_image = MagicMock()
    fake_image.numpy.return_value = np.zeros((4, 4, 4), dtype=np.float32)
    fake_image.orientation = "RAS"
    fake.from_numpy.return_value = fake_image
    fake.image_read.return_value = fake_image
    fake.reorient_image2.return_value = fake_image
    sys.modules["ants"] = fake
    try:
        yield fake
    finally:
        sys.modules.pop("ants", None)


@pytest.fixture(autouse=True)
def _reset_pil_max_image_pixels():
    """Reset PIL.Image.MAX_IMAGE_PIXELS to the package's finite limit around every test.

    PIL's MAX_IMAGE_PIXELS is a module-level global that guards against
    decompression-bomb images. A test that mutates it (e.g. setting it to
    None to allow a huge synthetic image, or to a small value to test the
    guard) would leak that mutation into subsequent tests because module
    globals are process-wide. This autouse fixture resets the global to
    the package's chosen finite limit (2_000_000_000) before and after
    every test so no cross-test contamination can occur.
    """
    PIL.Image.MAX_IMAGE_PIXELS = 2_000_000_000
    yield
    PIL.Image.MAX_IMAGE_PIXELS = 2_000_000_000

