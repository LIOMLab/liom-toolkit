"""SSL corpus builder: multi-plane slice extraction + z-score per-channel + MIP QC.

This module is the data layer the masked-inpainting pretraining loop
consumes. It builds a slice corpus from OME-Zarr volumes by:

1. Resolving the level-0 (full-resolution) dask array via the existing
   ``_level0_component`` NGFF resolver (do not hand-roll a path resolver).
2. Extracting 2D multi-channel slices along the coronal (Z), sagittal (Y),
   or axial (X) plane (D-03 — 50/25/25 plane mix).
3. Normalizing each slice per-channel with z-score (D-03d — z-score, NOT
   contrast-limited adaptive histogram equalization; the non-linear
   remapping of histogram-equalization destroys the intensity relationships
   the inpainting reconstruction loss relies on).
4. Quality-checking each volume via its max-intensity projection so
   catastrophic-signal-loss brains are skipped.

Both channels (555nm + 647nm) are preserved — the channel dim is never
indexed away (D-03b). All paths are parameters (``volume_paths``); there is
no hardcoded lab default. Validation uses ``if ...: raise ValueError(...)``
with the offending value in the message (AGENTS section 2 — never ``assert``
for validation, never a silent NaN/zero-filled fallback).
"""

from __future__ import annotations

from typing import cast

import dask.array as da
import numpy as np
from numpy.typing import NDArray

# torch is in the [ai] extra. The upfront ImportError here is the honest
# signal on an io-only install -- the message names [ai,benchmark] (the
# torch + MONAI path the SSL pretraining stack needs). The `from e` chain
# preserves the underlying error for debugging (AGENTS section 2). MONAI
# itself is imported function-scope in the masking module, not here.
try:
    import torch  # ruff: ignore[unused-import] -- imported for the guard side-effect; the [ai] extra is the honest signal
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError(
        "Please install liom-toolkit[ai,benchmark] to use the SSL corpus builder."
    ) from e

from liom_toolkit.segmentation.vseg.dataset import _level0_component

__all__ = [
    "SSLCorpus",
    "extract_plane_slice",
    "mip_qc",
    "z_score_per_channel",
]


def extract_plane_slice(volume: da.Array, axis: int, index: int) -> NDArray[np.generic]:
    """Extract a 2D multi-channel slice from a (C, Z, Y, X) volume along the given axis.

    Parameters
    ----------
    volume : da.Array
        The 4D OME-Zarr volume ``(C, Z, Y, X)`` as a dask array.
    axis : int
        ``1``=coronal (Z), ``2``=sagittal (Y), ``3``=axial (X) — indexing
        into ``(C, Z, Y, X)``. Axis 0 (the channel dim) is never sliced
        away (D-03b — both 555nm + 647nm channels are kept).
    index : int
        The slice index along ``axis``.

    Returns
    -------
    NDArray
        The 2D multi-channel slice ``(C, spatial_1, spatial_2)`` as a real
        numpy array (torch needs a concrete array, not a dask array).

    Raises
    ------
    ValueError
        If ``axis`` is outside ``[1, volume.ndim - 1]`` (the channel axis 0
        must be preserved) or ``index`` is out of bounds for ``axis``.
    """
    if axis < 1 or axis >= volume.ndim:
        raise ValueError(
            f"axis must be in [1, {volume.ndim - 1}] (channel axis 0 is preserved), got axis={axis}"
        )
    size_along_axis = int(volume.shape[axis])
    if index < 0 or index >= size_along_axis:
        raise ValueError(f"index {index} out of bounds for axis {axis} (size {size_along_axis})")
    sl = [slice(None)] * volume.ndim
    sl[axis] = index
    # Materialize the Dask slice to a real NumPy array -- torch.tensor()
    # below (in the pretraining loop) requires a concrete array, not a
    # Dask array (removing this .compute() would pass a Dask Array to
    # torch.tensor, which raises TypeError). This is a genuine
    # Dask->PyTorch boundary.
    return np.asarray(volume[tuple(sl)].compute())


def z_score_per_channel(slice_2d: NDArray[np.generic]) -> NDArray[np.floating]:
    """Z-score normalize a multi-channel 2D slice per channel.

    Each channel is normalized independently: ``out[c] = (x - mean) / std``.
    Per-channel (not global) normalization keeps the 555nm and 647nm
    intensity scales from bleeding into each other (D-03d).

    Parameters
    ----------
    slice_2d : NDArray
        A ``(C, H, W)`` multi-channel slice.

    Returns
    -------
    NDArray
        The z-scored slice (float), same shape as the input, with each
        channel having mean ~0 and std ~1.

    Raises
    ------
    ValueError
        If any channel has zero std (a constant channel) — dividing would
        produce NaN/inf. The error names the offending channel so the
        caller can act on it (AGENTS section 2 — no silent NaN fallback).
    """
    if slice_2d.ndim != 3:
        raise ValueError(
            f"z_score_per_channel expects a (C, H, W) 3D slice, got ndim={slice_2d.ndim}"
        )
    out = np.empty(slice_2d.shape, dtype=np.result_type(slice_2d, np.float32))
    for c in range(slice_2d.shape[0]):
        channel = slice_2d[c].astype(np.float64, copy=False)
        std = float(channel.std())
        # A constant channel has std exactly 0.0; the tiny tolerance guards
        # against floating-point noise on a near-constant channel that would
        # otherwise divide by a vanishingly small number and produce inf.
        if std < 1e-12:
            raise ValueError(
                f"zero-std channel {c} cannot be z-scored (constant channel — "
                f"mean={float(channel.mean())}); drop the channel or filter the slice"
            )
        out[c] = (channel - float(channel.mean())) / std
    return out


def mip_qc(volume: da.Array, threshold: float = 1.0) -> bool:
    """Max-intensity-projection quality check: flag catastrophic-signal-loss volumes.

    A volume whose max-intensity projection is at or below ``threshold`` is
    treated as catastrophic signal loss (e.g. a failed acquisition or a
    misloaded file) and skipped — feeding empty slices to the pretraining
    loop teaches the network nothing. The check is on the raw volume (pre-
    normalization) so the threshold is in the volume's native intensity
    units.

    Parameters
    ----------
    volume : da.Array
        The 4D OME-Zarr volume ``(C, Z, Y, X)`` as a dask array.
    threshold : float
        The minimum max-intensity value for a volume to pass. Defaults to
        ``1.0`` so an all-zero or all-near-zero volume fails.

    Returns
    -------
    bool
        ``True`` if the volume's max intensity is above ``threshold``
        (keep), ``False`` otherwise (skip).
    """
    # .max() on a dask array returns a 0-d dask array; .compute() here is a
    # genuine scalar-materialization boundary (we need a Python float to
    # compare against the threshold, not a dask scalar proxy).
    max_val = float(volume.max().compute())
    return max_val > threshold


class SSLCorpus(Dataset):
    """SSL slice corpus builder over a set of OME-Zarr volumes.

    Holds the volume paths, the dominant sampling axis, and the coronal /
    sagittal / axial plane mix (D-03 — 50/25/25 by default). The actual
    slice sampling / dataset length logic is added by later plans; this
    constructor establishes the parameter contract: all paths are
    parameters (``volume_paths``), there is no hardcoded lab default
    (AGENTS section 1), and the plane mix must sum to 1.0.

    Parameters
    ----------
    volume_paths : list[str]
        Paths to the OME-Zarr stores to build the corpus from. There is no
        default — the caller always supplies the paths.
    axis : int
        The default sampling axis (``1``=coronal, ``2``=sagittal,
        ``3``=axial) used when a single-axis corpus is requested.
    plane_mix : tuple[float, float, float]
        The (coronal, sagittal, axial) sampling proportions (D-03). Must
        sum to 1.0 within a small tolerance. Defaults to ``(0.5, 0.25,
        0.25)``.

    Raises
    ------
    ValueError
        If ``volume_paths`` is empty, ``axis`` is out of range, or
        ``plane_mix`` does not sum to 1.0.
    """

    def __init__(
        self,
        volume_paths: list[str],
        axis: int = 1,
        plane_mix: tuple[float, float, float] = (0.5, 0.25, 0.25),
    ) -> None:
        if not volume_paths:
            raise ValueError(
                "volume_paths must be a non-empty list of OME-Zarr store paths (got an empty list)"
            )
        if axis < 1 or axis > 3:
            raise ValueError(
                f"axis must be 1 (coronal), 2 (sagittal), or 3 (axial), got axis={axis}"
            )
        mix_sum = float(sum(plane_mix))
        if abs(mix_sum - 1.0) > 1e-6:
            raise ValueError(f"plane_mix must sum to 1.0, got {plane_mix} (sum={mix_sum})")
        if len(plane_mix) != 3:
            raise ValueError(
                f"plane_mix must be a 3-tuple (coronal, sagittal, axial), "
                f"got length {len(plane_mix)}"
            )
        self.volume_paths: list[str] = list(volume_paths)
        self.axis: int = axis
        self.plane_mix: tuple[float, float, float] = plane_mix

    def __len__(self) -> int:
        """Return the number of volumes in the corpus.

        The full per-slice dataset length (sum of valid slices across all
        volumes after MIP QC) is computed by later plans; the base length
        is the volume count.

        Returns
        -------
        int
            The number of OME-Zarr stores in the corpus.
        """
        return len(self.volume_paths)

    def _load_volume(self, path: str) -> da.Array:
        """Resolve and read the level-0 dask array for one OME-Zarr store.

        Reuses ``_level0_component`` so the NGFF v0.5 ``s0`` vs v0.4 ``0``
        resolution naming is handled by the existing resolver (do not
        hand-roll a path resolver). ``typing.cast`` narrows the
        ``da.from_zarr`` return (which the type checker infers as
        ``Unknown`` across the dask collection-expression / legacy-array
        split) to the declared ``da.Array`` return type without an
        ``assert`` (AGENTS section 2 — assert is not validation).

        Returns
        -------
        da.Array
            The level-0 ``(C, Z, Y, X)`` dask array for the store.
        """
        component = _level0_component(path)
        return cast("da.Array", da.from_zarr(path, component=component))
