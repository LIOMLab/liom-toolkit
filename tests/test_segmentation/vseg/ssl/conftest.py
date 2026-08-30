"""Pytest configuration for the SSL pretraining test subpackage.

Mirrors the autouse fixtures in ``tests/conftest.py`` (PIL decompression-bomb
guard reset + DDP env-var unset) so SSL tests inherit the same cross-test
isolation, and adds the synthetic OME-Zarr + synthetic vessel-slice fixtures
the corpus / masking tests consume. All disk fixtures write into ``tmp_path``
only (AGENTS section 5 -- never into the repo tree).
"""

import os

import numpy as np
import PIL.Image
import pytest


@pytest.fixture(autouse=True)
def _reset_pil_max_image_pixels():
    """Reset PIL.Image.MAX_IMAGE_PIXELS to the package's finite limit around every test.

    Sibling of the top-level ``tests/conftest.py`` fixture: a test that
    mutates the global would leak it into subsequent tests. The package keeps
    a finite (not ``None``) limit as a decompression-bomb guard (AGENTS
    section 2), so the reset target is the same ``2_000_000_000`` value.
    """
    PIL.Image.MAX_IMAGE_PIXELS = 2_000_000_000
    yield
    PIL.Image.MAX_IMAGE_PIXELS = 2_000_000_000


@pytest.fixture(autouse=True)
def _unset_ddp_env_vars():
    """Unset torchrun / DDP env vars in teardown so they do not leak between tests.

    The pretraining loop reads ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` /
    ``MASTER_ADDR`` / ``MASTER_PORT`` to decide whether to enter the
    distributed path. A stale value leaking from a prior test (or a
    subprocess) would silently flip a ``ddp=False`` call into the
    distributed branch. Sibling of the top-level fixture.
    """
    yield
    for env_var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(env_var, None)


@pytest.fixture
def synthetic_ome_zarr(tmp_path) -> str:
    """A small (2, 8, 16, 16) uint16 2-channel OME-Zarr volume written via save_zarr.

    Channel 0 is a bright sphere (value 1000) centered in the volume; channel
    1 is a dimmer off-center structure (value 400). Both channels are
    non-constant so z-score per-channel normalization is well-defined, and
    the volume is small enough that the full-pyramid write + dask read-back
    is fast on CPU. Returns the zarr store path for ``da.from_zarr`` /
    ``_level0_component`` resolution.
    """
    from liom_toolkit.conversion.conversion import save_zarr

    vol = np.zeros((2, 8, 16, 16), dtype=np.uint16)
    zz, yy, xx = np.ogrid[:8, :16, :16]
    # Channel 0: bright centered sphere.
    vol[0][(zz - 4) ** 2 + (yy - 8) ** 2 + (xx - 8) ** 2 <= 16] = 1000
    # Channel 1: dimmer off-center structure (non-constant, non-identical to ch0).
    vol[1][(zz - 2) ** 2 + (yy - 5) ** 2 + (xx - 5) ** 2 <= 9] = 400
    zarr_path = str(tmp_path / "ssl_vol.zarr")
    # 4D chunks: one channel per chunk so the slice read is a single chunk.
    save_zarr(vol, zarr_path, scales=(6.5, 6.5, 6.5), chunks=(1, 8, 16, 16))
    return zarr_path


@pytest.fixture
def synthetic_vessel_slice() -> np.ndarray:
    """A (1, 16, 16) float32 slice with a thin bright line for Frangi-responsive masking tests.

    A single-channel 2D slice with a horizontal bright line (value 1.0) on a
    dark (0.0) background. The line is thin (1px tall) so Frangi / Hessian
    vesselness filters respond strongly to it -- this is the proxy the
    vessel-aware masking transform biases hole-center sampling toward. Shape
    ``(1, 16, 16)`` keeps the channel dim (axis 0) so the fixture mirrors the
    corpus output layout (C, H, W).
    """
    sl = np.zeros((1, 16, 16), dtype=np.float32)
    sl[0, 7:8, 1:15] = 1.0
    return sl
