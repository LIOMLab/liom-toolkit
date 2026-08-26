"""Streaming OME-Zarr writer for live acquisition (NGFF v0.5).

This module provides a chunk-by-chunk OME-Zarr writer so a live-acquisition
caller (e.g. a lightsheet microscope controller) can write a whole-stack
OME-Zarr **without buffering the full volume in RAM**. Frames are written
into a pre-allocated level-0 array as they arrive; the multiscale pyramid is
downsampled from disk after the last frame via Dask (no eager materialization).

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


class OmeZarrWriter:
    """Streaming OME-Zarr writer for live acquisition (NGFF v0.5).

    Frames are written into a pre-allocated level-0 array as they arrive via
    ``writer[:, z_idx, :, :] = frame``; the multiscale pyramid is downsampled
    from disk after the last frame via Dask (no eager materialization). This
    lets a live-acquisition caller write a whole-stack OME-Zarr **without
    buffering the full volume in RAM**.

    Typical usage::

        writer = OmeZarrWriter(
            store_path="stack.ome.zarr",
            shape=(n_channels, n_planes, ysize, xsize),  # (c, z, y, x), 4D
            chunk_shape=(1, 1, ysize, xsize),            # one plane per chunk
            dtype=np.uint16,
            overwrite=True,
            downscale_factor=2,
            unit="micrometer",
        )
        writer[:, z_idx, :, :] = frame        # per-frame streaming write
        writer.finalize(
            res=[stack_step_um, pixel_y_um, pixel_x_um],  # µm (z, y, x)
            n_levels=4,
            omero_channels=[{"label": "555 nm", "color": "00FF00", ...}],
        )

    Diverges from the linumpy pattern (see module docstring): channel scale
    = 1.0 (not 0.0), anisotropic Y/X-only downsample (Z stays at base), and
    ``exact=True`` on ``require_array`` so a shape mismatch raises instead
    of silently reusing an existing L0 array (no silent wrong-data fallback,
    AGENTS.md §2).
    """

    def __init__(
        self,
        store_path: str,
        shape: tuple[int, ...],
        chunk_shape: tuple[int, ...],
        dtype,
        overwrite: bool = True,
        downscale_factor: int = 2,
        unit: str = "micrometer",
        shards: tuple[int, ...] | None = None,
    ) -> None:
        """Pre-allocate the level-0 zarr array for streaming writes.

        :param store_path: Filesystem path to the output zarr group. This MUST
            be a plain filesystem path, NOT a ``file://`` URL — passing a
            ``file://`` URL would orphan-create a literal ``file:/...``
            directory in the CWD (``Path("file:///tmp/x.zarr")`` is parsed as
            the relative path ``file:/tmp/x.zarr``) while ``parse_url`` opens
            the real store at the URL's path, leaving the two out of sync.
            Use ``liom_toolkit.utils.io.load_zarr`` (which goes through
            ``parse_url``) to read URL-located stores.
        :param shape: Shape of the dataset. Expected ordering is
            ``(c, z, y, x)`` for 4D or ``(z, y, x)`` for 3D.
        :param chunk_shape: Chunk size on disk. One plane per chunk
            (``(1, 1, y, x)`` for 4D) is the streaming-friendly default.
        :param dtype: Data type of the dataset.
        :param overwrite: If True, remove an existing directory/symlink at
            *store_path* before creating a fresh one. If False and the path
            exists, raise ``FileExistsError`` (no silent clobber).
        :param downscale_factor: Per-level Y/X downsample factor (default 2).
            Must be >= 2 (a factor of 0 crashes with ``ZeroDivisionError``, 1
            produces duplicate pyramid levels, and a negative factor produces
            negative scales).
        :param unit: NGFF UDUNITS-2 length unit for the spatial axes
            (default ``"micrometer"``, repo convention — NOT linumpy's mm).
        :param shards: Shard shape (``None`` for no sharding).
        :raises ValueError: If ``unit`` is not a known NGFF length unit,
            ``shape`` is not 3D or 4D, ``store_path`` is a ``file://`` URL,
            or ``downscale_factor`` is < 2.
        """
        if unit not in _NGFF_LENGTH_UNITS:
            raise ValueError(
                f"Unsupported unit {unit!r}; use a NGFF UDUNITS-2 length unit "
                f"(one of {sorted(_NGFF_LENGTH_UNITS)})."
            )
        ndims = len(shape)
        if ndims not in (3, 4):
            raise ValueError(f"shape must be 3D or 4D, got len(shape)={ndims}.")
        if downscale_factor < 2:
            raise ValueError(
                f"downscale_factor must be >= 2 (0 raises ZeroDivisionError, "
                f"1 produces duplicate levels, negative produces negative "
                f"scales); got {downscale_factor!r}."
            )
        # Reject file:// URLs: create_directory takes a filesystem path, and
        # Path("file:///tmp/x.zarr") is parsed as the relative path
        # "file:/tmp/x.zarr" — create_directory would mkdir that literal path
        # in the CWD while parse_url opens the real store at the URL's path,
        # leaving the two out of sync (orphan dir in CWD). parse_url handles
        # URL-located stores for the *read* path (load_zarr); the streaming
        # writer requires a plain filesystem path.
        if isinstance(store_path, str) and store_path.startswith("file://"):
            raise ValueError(
                f"store_path must be a filesystem path, not a file:// URL "
                f"(got {store_path!r}); create_directory would orphan a "
                f"literal 'file:/...' directory in the CWD. Pass a plain path."
            )

        self.shape = tuple(int(v) for v in shape)
        self.downscale_factor = downscale_factor
        self.unit = unit
        self.store_path_str = str(store_path)
        self.fmt = CurrentFormat()
        self.axes = generate_axes_dict(ndims, unit=unit)

        # Symlink-aware directory creation (FileExistsError on collision when
        # overwrite=False — no silent clobber).
        create_directory(Path(store_path), overwrite=overwrite)

        store = parse_url(store_path, mode="w").store
        self.root = zarr.group(store=store)

        # Pre-allocate L0 with exact=True so an existing "0" array with a
        # different shape/chunks/dtype raises instead of being silently
        # reused (no silent wrong-data fallback, AGENTS.md §2).
        self.root.require_array(
            "0",
            shape=self.shape,
            exact=True,
            chunks=tuple(int(v) for v in chunk_shape),
            shards=shards,
            dtype=dtype,
            chunk_key_encoding=self.fmt.chunk_key_encoding,
            dimension_names=[ax["name"] for ax in self.axes],
        )

    def __setitem__(self, key, value) -> None:
        """Write *value* at *key* into the pre-allocated level-0 array.

        This is the per-frame streaming write path — the caller never has to
        hold the full volume in RAM.
        """
        self.root["0"][key] = value

    def __getitem__(self, key):
        """Read a slice from the level-0 array (cheap read-back)."""
        return self.root["0"][key]

    @property
    def ndim(self) -> int:
        """Number of dimensions of the level-0 array."""
        return self.root["0"].ndim

    @property
    def dtype(self) -> np.dtype:
        """Data type of the level-0 array."""
        return self.root["0"].dtype

    def finalize(
        self,
        res: Sequence[float],
        n_levels: int,
        omero_channels: list[dict] | None = None,
    ) -> None:
        """Downsample the pyramid from disk and write NGFF v0.5 multiscales metadata.

        Reads L0 via ``da.from_zarr``, downsamples Y/X only (Z stays at base
        resolution — matches ``_DOWNSAMPLE_AXES``) per level via
        ``da_resize``, and writes each level via ``da.to_zarr``. No eager
        No eager materialization — Dask materializes lazily at the write boundary.

        Each iteration reads from L0 (cumulative factor from base, NOT
        compounded from the previous level) to avoid rounding error.

        Metadata is written via ``write_multiscales_metadata`` with a single
        ``metadata=`` kwarg containing both the provenance fields and an
        ``"omero"`` sub-key (NGFF v0.5 — omero MUST be nested inside the
        ``metadata=`` kwarg, NOT passed as a separate ``omero=`` kwarg).

        :param res: ``(z, y, x)`` base voxel size in ``unit`` (µm by default).
        :param n_levels: Number of downsample levels (excluding L0). Clamped
            by ``validate_n_levels`` to what the Y/X shapes can support.
        :param omero_channels: Optional list of omero channel dicts
            (``{"label", "color" (6-char hex, no #), "active", "wavelength",
            "window": {"min","max","start","end"}}``). Written to
            ``root.attrs["ome"]["omero"]["channels"]`` by the ome-zarr writer.
        :raises ValueError: If ``res`` is not a 3-element sequence or any
            element is not positive (a negative/zero voxel size is meaningless
            and would silently produce wrong physical coordinates —
            AGENTS.md §2).
        """
        if len(res) != 3:
            raise ValueError(f"res must be a 3-element (z, y, x) sequence, got len={len(res)}.")
        if any(v <= 0 for v in res):
            raise ValueError(f"res values must be positive, got {tuple(res)!r}.")
        res = (float(res[0]), float(res[1]), float(res[2]))

        axis_names = [ax["name"] for ax in self.axes]
        n_levels_clamped = validate_n_levels(
            n_levels, self.shape, axis_names, downscale_factor=self.downscale_factor
        )

        # Anisotropic on-disk downsample loop: Y/X only (Z stays at base).
        # Index convention: 4D (c, z, y, x) -> Y=2, X=3; 3D (z, y, x) -> Y=1, X=2.
        src = da.from_zarr(self.root.store_path / "0")
        y_idx, x_idx = (2, 3) if self.ndim == 4 else (1, 2)
        for i in range(1, n_levels_clamped + 1):
            new_shape = list(src.shape)
            new_shape[y_idx] = src.shape[y_idx] // (self.downscale_factor**i)
            new_shape[x_idx] = src.shape[x_idx] // (self.downscale_factor**i)
            down = da_resize(src, tuple(new_shape), preserve_range=True, anti_aliasing=False)
            da.to_zarr(
                arr=down,
                url=str(self.root.store_path),
                component=str(i),
                zarr_format=self.fmt.zarr_format,
                chunk_key_encoding=self.fmt.chunk_key_encoding,
            )

        # Per-level coordinateTransformations: n_levels_clamped + 1 entries
        # (including L0). Channel scale = 1.0, Z constant, Y/X cumulative.
        per_level_scales = create_transformation_dict(
            n_levels_clamped + 1,
            res,
            ndims=self.ndim,
            downscale_factor=self.downscale_factor,
        )
        datasets = [
            {"path": str(i), "coordinateTransformations": per_level_scales[i]}
            for i in range(n_levels_clamped + 1)
        ]

        provenance = {
            "method": "streaming_dask_resize",
            "version": "1.0",
            "args": {
                "downscale_factor": self.downscale_factor,
                "downsample_axes": ["y", "x"],
            },
        }
        metadata = dict(provenance)
        if omero_channels is not None:
            # omero MUST be nested inside the single metadata= kwarg — the
            # ome-zarr writer pops it from metadata["metadata"]["omero"] and
            # writes it to root.attrs["ome"]["omero"]. Passing omero as a
            # separate top-level kwarg does NOT work.
            metadata = {**provenance, "omero": {"channels": omero_channels}}

        write_multiscales_metadata(
            self.root,
            datasets,
            fmt=self.fmt,
            axes=self.axes,
            name="stack",
            metadata=metadata,
        )


class AnalysisOmeZarrWriter(OmeZarrWriter):
    """Streaming writer that appends custom target-resolution pyramid levels.

    Subclass of :class:`OmeZarrWriter` — inherits the pre-allocate + L0
    streaming write path. Use :meth:`finalize_with_resolutions` (instead of
    :meth:`OmeZarrWriter.finalize`) to build a pyramid at specific target
    resolutions (e.g. 10/25/50/100 µm) optimized for downstream analysis.

    **Diverges from linumpy (per the repo's correctness rules):** linumpy's
    ``AnalysisOmeZarrWriter`` *replaces* L0 with a downsampled target-res
    array via temp+move (destructive — raw data is lost). This class keeps
    raw L0 untouched at ``base_res`` and appends target levels as L1..Ln
    beyond raw L0. Raw data is never destroyed.

    Usage::

        writer = AnalysisOmeZarrWriter(
            store_path="stack.ome.zarr",
            shape=(1, n_planes, ysize, xsize),
            chunk_shape=(1, 1, ysize, xsize),
            dtype=np.uint16,
            overwrite=True,
            unit="micrometer",
        )
        writer[:, z_idx, :, :] = frame        # raw frames into L0
        writer.finalize_with_resolutions(
            base_res=(6.5, 6.5, 6.5),        # µm (z, y, x) at L0
            target_resolutions_um=(10, 25, 50, 100),
            make_isotropic=True,
        )
    """

    def finalize_with_resolutions(
        self,
        base_res: tuple[float, float, float],
        target_resolutions_um: tuple[float, ...] = (10, 25, 50, 100),
        make_isotropic: bool = True,
        omero_channels: list[dict] | None = None,
    ) -> None:
        """Append target-resolution pyramid levels beyond raw L0.

        L0 stays raw at ``base_res`` (untouched — do NOT re-downsample "0").
        Each target in ``target_resolutions_um`` becomes a level L1..Ln,
        downsampled FROM L0 (raw) — NOT from the previous target level.

        Targets that would upscale ANY dim are dropped (they would silently
        invent data via interpolation — AGENTS.md §2). For
        ``make_isotropic=True`` a target is valid only if
        ``target_um >= max(base_res)`` (so every dim's per-dim scale factor is
        >= 1); for ``make_isotropic=False`` the uniform scale
        ``target_um / min(base_res)`` is >= 1 whenever
        ``target_um >= min(base_res)``, so the existing ``min_base`` filter
        applies.

        Per-level scale dicts record the **ACTUAL per-dim voxel**
        (``base_res_d * sf``), NOT the target_um. For ``make_isotropic=True``
        the actual voxel equals the target for every dim; for
        ``make_isotropic=False`` the actual voxel is anisotropic (aspect
        ratio preserved) — recording the target_um there would be the
        silent-wrong-coordinate failure mode.

        :param base_res: ``(z, y, x)`` base voxel size in µm at L0.
        :param target_resolutions_um: Target resolutions in µm (default
            ``(10, 25, 50, 100)``). Targets that would upscale any dim are
            dropped (see above).
        :param make_isotropic: If True (default), each dim is scaled
            independently to reach the target resolution (isotropic output
            voxels, aspect ratio changes). If False, all dims scale uniformly
            by ``target_um / min(base_res)`` (aspect ratio preserved,
            anisotropic output voxels).
        :param omero_channels: Optional omero channel dicts (same shape as
            :meth:`OmeZarrWriter.finalize`).
        :raises ValueError: If ``base_res`` is not a 3-element sequence or
            any element is not positive (a negative/zero base voxel is
            meaningless and would crash on ``target_um / b`` or silently
            produce wrong physical coordinates — AGENTS.md §2).
        """
        if len(base_res) != 3:
            raise ValueError(
                f"base_res must be a 3-element (z, y, x) sequence, got len={len(base_res)}."
            )
        if any(v <= 0 for v in base_res):
            raise ValueError(f"base_res values must be positive, got {tuple(base_res)!r}.")
        base_res = (float(base_res[0]), float(base_res[1]), float(base_res[2]))

        # Drop targets that would upscale (silently invent data, AGENTS.md §2).
        # The validity check is per-dim, not against the minimum base: a target
        # is valid only if every dim's per-dim scale factor ``sf_d = target_um /
        # base_res_d`` is >= 1 (i.e. the target downsamples that dim, never
        # upscales it).
        #
        # For ``make_isotropic=True`` each dim is scaled independently to reach
        # the target, so a target is valid iff ``target_um >= max(base_res)``
        # (otherwise the thicker dims would have ``sf_d < 1`` and would be
        # silently upscaled by ``da_resize`` — exactly the silent-data-invention
        # failure mode this writer is meant to forbid).
        #
        # For ``make_isotropic=False`` the uniform scale is
        # ``sf = target_um / min(base_res)``; ``target_um >= min(base_res)``
        # already guarantees ``sf >= 1`` for every dim, so the existing
        # ``min_base`` filter is correct on that path.
        min_base = min(base_res)
        max_base = max(base_res)
        if make_isotropic:
            valid_targets = [float(t) for t in target_resolutions_um if t >= max_base]
        else:
            valid_targets = [float(t) for t in target_resolutions_um if t >= min_base]

        # L0 stays raw (untouched). Downsample each target FROM L0.
        src = da.from_zarr(self.root.store_path / "0")
        # shape[1:] drops the channel axis for 4D; for 3D shape is already (z,y,x).
        spatial_shape = self.shape[1:] if self.ndim == 4 else self.shape

        per_level_scales: list[list[float]] = []
        # L0 scale: channel=1.0 (4D) + base_res; or just base_res (3D).
        if self.ndim == 4:
            per_level_scales.append([1.0, *base_res])
        else:
            per_level_scales.append(list(base_res))

        for i, target_um in enumerate(valid_targets, start=1):
            if make_isotropic:
                sf = [target_um / b for b in base_res]
            else:
                u = target_um / min_base
                sf = [u, u, u]
            target_shape_3d = [max(1, int(s / f)) for s, f in zip(spatial_shape, sf)]
            # ACTUAL per-dim voxel at this level — NOT target_um.
            target_voxel = [b * f for b, f in zip(base_res, sf)]

            full_target_shape = (
                (self.shape[0], *target_shape_3d) if self.ndim == 4 else tuple(target_shape_3d)
            )
            # anti_aliasing=True for analysis pyramids — smoother than the
            # streaming L1..Ln power-of-2 path (which uses False).
            down = da_resize(src, full_target_shape, preserve_range=True, anti_aliasing=True)
            da.to_zarr(
                arr=down,
                url=str(self.root.store_path),
                component=str(i),
                zarr_format=self.fmt.zarr_format,
                chunk_key_encoding=self.fmt.chunk_key_encoding,
            )

            if self.ndim == 4:
                per_level_scales.append([1.0, *target_voxel])
            else:
                per_level_scales.append(list(target_voxel))

        datasets = [
            {"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": scale}]}
            for i, scale in enumerate(per_level_scales)
        ]

        provenance = {
            "method": "analysis_target_resolution",
            "version": "1.0",
            "args": {
                "base_res": list(base_res),
                "target_resolutions_um": list(valid_targets),
                "make_isotropic": make_isotropic,
            },
        }
        metadata = dict(provenance)
        if omero_channels is not None:
            metadata = {**provenance, "omero": {"channels": omero_channels}}

        write_multiscales_metadata(
            self.root,
            datasets,
            fmt=self.fmt,
            axes=self.axes,
            name="stack",
            metadata=metadata,
        )
