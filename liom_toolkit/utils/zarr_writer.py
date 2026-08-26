"""Streaming OME-Zarr writer for live acquisition (NGFF v0.5).

This module provides a chunk-by-chunk OME-Zarr writer so a live-acquisition
caller (e.g. a lightsheet microscope controller) can write a whole-stack
OME-Zarr **without buffering the full volume in RAM**. Frames are written
into a pre-allocated level-0 array as they arrive; the multiscale pyramid is
downsampled from disk after the last frame via Dask (no eager ``.compute()``).

Adapted from the linumpy pattern (NOT a verbatim port). Diverges from linumpy
on three points, all required by the repo's correctness rules
(AGENTS.md §2 — no silent wrong data):

* **Channel-axis scale = 1.0**, not linumpy's ``0.0`` bug. A ``0.0`` channel
  scale mislabels every channel's physical coordinate.
* **Anisotropic Y/X-only downsample.** Z stays at base resolution (matches
  the repo's ``_DOWNSAMPLE_AXES = {"y","x"}`` convention). linumpy downsamples
  Z too — diverge.
* **AnalysisOmeZarrWriter keeps raw L0.** Target-resolution levels are
  appended as L1..Ln beyond raw L0. linumpy *replaces* L0 with a downsampled
  target-res array via temp+move (destructive) — do NOT replicate that.

Target stack: NGFF v0.5, ``ome-zarr>=0.18.0``, ``zarr>=3.0``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import dask.array as da
import numpy as np
import zarr
from ome_zarr.dask_utils import resize as da_resize
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscales_metadata

from .io import _NGFF_LENGTH_UNITS, generate_axes_dict, validate_n_levels

__all__ = [
    "OmeZarrWriter",
    "AnalysisOmeZarrWriter",
    "create_directory",
    "create_transformation_dict",
]


def create_directory(store_path: Path, overwrite: bool = False) -> Path:
    """Create the directory at *store_path*, optionally removing an existing one.

    Symlink-aware: ``is_symlink()`` is True even for dangling symlinks while
    ``exists()`` follows the link. ``shutil.rmtree`` raises ``OSError`` on
    symlinks, so handle them separately (unlink the link only; the link's
    target is NOT deleted).

    :param store_path: Path to the directory to create.
    :param overwrite: If True, remove an existing directory/symlink at
        *store_path* before creating a fresh one. If False and the path
        exists, raise ``FileExistsError``.
    :raises FileExistsError: When *store_path* exists (as a real directory or
        a symlink) and ``overwrite=False``. ``FileExistsError`` is the
        stdlib-correct exception for "path already exists" (NOT
        ``ValueError`` — and never ``assert``, which is stripped under
        ``python -O``).
    :return: The created directory path.
    """
    directory = Path(store_path)
    if directory.is_symlink():
        if overwrite:
            # Unlink the symlink only — never rmtree a symlink (would either
            # raise OSError or follow the link and delete the target).
            directory.unlink()
        else:
            raise FileExistsError(
                f"Path {directory.as_posix()} already exists as a symlink. "
                "Set overwrite=True to overwrite."
            )
    elif directory.exists():
        if overwrite:
            shutil.rmtree(directory)
        else:
            raise FileExistsError(
                f"Directory {directory.as_posix()} already exists. "
                "Set overwrite=True to overwrite."
            )
    directory.mkdir(parents=True)
    return directory


def create_transformation_dict(
    n_levels: int,
    voxel_size: Sequence[float],
    ndims: int = 4,
    downscale_factor: int = 2,
) -> list[list[dict]]:
    """Per-level ``coordinateTransformations`` for ``write_multiscales_metadata``.

    Each entry is ``[{"type": "scale", "scale": [s_c, s_z, s_y, s_x]}]`` for
    4D (or ``[s_z, s_y, s_x]`` for 3D — no channel element).

    Semantics (NGFF v0.5, anisotropic LSFM):

    * **Channel scale = 1.0** (linumpy's bug was ``0.0`` — a 0 channel scale
      mislabels every channel coordinate).
    * **Z stays at base** (matches ``_DOWNSAMPLE_AXES = {"y","x"}``; only Y/X
      are downsampled in this repo).
    * **Y/X grow cumulatively** by ``downscale_factor`` per level: level ``i``
      Y/X scale = ``voxel_size_yx * downscale_factor ** i`` (cumulative from
      L0, NOT compounded from the previous level — avoids rounding error).

    **Scale list order MUST match axes order** (``c, z, y, x`` for 4D). Getting
    the order wrong is the silent-wrong-coordinate failure mode AGENTS.md §2
    warns about — the on-disk array shape is correct but the physical
    coordinates are mislabeled. This helper takes ``ndims`` explicitly and
    branches on it so the scale list length always matches the axes length.

    :param n_levels: Number of levels (INCLUDING L0). Returns ``n_levels``
        entries.
    :param voxel_size: ``(z, y, x)`` base voxel size in ``unit``.
    :param ndims: 3 or 4 — selects whether a channel element is prepended.
    :param downscale_factor: Per-level Y/X downsample factor (default 2).
    :return: A list of ``n_levels`` ``[{"type":"scale","scale":[...]}]``
        entries, one per pyramid level (L0 first).
    """
    if ndims not in (3, 4):
        raise ValueError(f"ndims must be 3 or 4, got {ndims!r}.")
    if len(voxel_size) != 3:
        raise ValueError(
            f"voxel_size must be a 3-element (z, y, x) sequence, got len={len(voxel_size)}."
        )

    z_base, y_base, x_base = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
    transforms: list[list[dict]] = []
    for i in range(n_levels):
        if ndims == 4:
            scale = [
                1.0,
                z_base,
                y_base * downscale_factor**i,
                x_base * downscale_factor**i,
            ]
        else:  # ndims == 3
            scale = [
                z_base,
                y_base * downscale_factor**i,
                x_base * downscale_factor**i,
            ]
        transforms.append([{"type": "scale", "scale": scale}])
    return transforms
