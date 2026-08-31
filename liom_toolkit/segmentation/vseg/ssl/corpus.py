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

from typing import Any, cast

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

# Bound on the GDS CuFile handle cache. Without a limit, a 2001-slice volume
# would open 2001 file descriptors -- risky against the per-process ulimit
# (typically 1024). 512 is well under the default ulimit and large enough
# that the working set of a 100-step epoch (800 random slice reads) mostly
# hits the cache.
_GDS_CUFILE_CACHE_MAX = 512


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
    sliced = volume[tuple(sl)]
    # Materialize a Dask slice to a real NumPy array -- torch.tensor()
    # below (in the pretraining loop) requires a concrete array, not a
    # Dask array (removing this .compute() would pass a Dask Array to
    # torch.tensor, which raises TypeError). This is a genuine
    # Dask->PyTorch boundary. A numpy array (in_memory=True) is already
    # concrete -- .compute() is skipped.
    return np.asarray(sliced.compute() if hasattr(sliced, "compute") else sliced)


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
    sagittal / axial plane mix (D-03 — 50/25/25 by default). The sampler
    draws each sample's plane axis from a categorical distribution weighted
    by ``plane_mix`` (D-03), extracts a 2D multi-channel slice along the
    chosen axis (reusing :func:`extract_plane_slice`), keeps BOTH channels
    (D-03b — the channel dim is never indexed away), z-scores per-channel
    (D-03d — z-score per-channel, not contrast-limited adaptive histogram
    equalization), and applies light augmentation (D-03e —
    random flips + 90deg rotations + mild intensity jitter only).

    Brain-centered sampling with a periphery margin (D-03c): the slice index
    is drawn from a window centered on the brain center (when a brain center
    is supplied) but the periphery is NEVER cropped — the full spatial extent
    of the slice is returned so batches contain both vessel-bearing tissue
    AND truly empty background. The ship gate's FPR-on-empty metric rewards
    learning what non-vessel looks like; cropping to the brain mask would
    suppress exactly that signal.

    All paths are parameters (``volume_paths``); there is no hardcoded lab
    default (AGENTS section 1). Validation uses ``if ...: raise
    ValueError(...)`` with the offending value in the message (AGENTS
    section 2 — never ``assert`` for validation, never a silent NaN
    fallback).

    Parameters
    ----------
    volume_paths : list[str]
        Paths to the OME-Zarr stores to build the corpus from. There is no
        default — the caller always supplies the paths.
    axis : int
        The default sampling axis (``1``=coronal, ``2``=sagittal,
        ``3``=axial) used to size the dataset length and as the fallback
        when ``plane_mix`` concentrates on one axis.
    plane_mix : tuple[float, float, float]
        The (coronal, sagittal, axial) sampling proportions (D-03). Must
        sum to 1.0 within a small tolerance. Defaults to ``(0.5, 0.25,
        0.25)``.
    augment : bool, optional
        Whether to apply light augmentation (random flips + 90deg
        rotations + mild intensity jitter, D-03e). Defaults to ``True``.
        Disable for tests that need clean z-score statistics.
    intensity_jitter : float, optional
        The standard deviation of the additive Gaussian intensity jitter
        applied per-channel when ``augment=True`` (D-03e — mild only; the
        pretext task is the primary regularizer). Defaults to ``0.05``
        (on z-scored data, so 5% of the unit std).
    periphery_margin : float, optional
        The fraction of the slice-index range on each side of the brain
        center that is included in the brain-centered sampling window
        (D-03c). ``0.0`` = sample only the exact center; ``0.5`` = sample
        the full range. Defaults to ``0.5`` (full range — no brain center
        supplied means sample all slices; when a brain center IS supplied,
        a smaller margin biases toward the center while still including
        periphery slices).
    rng : np.random.Generator | None, optional
        The random generator for axis + slice-index sampling + aug draws.
        Defaults to a fresh ``default_rng()`` (non-deterministic; pass a
        seeded generator for reproducible tests).

    Raises
    ------
    ValueError
        If ``volume_paths`` is empty, ``axis`` is out of range,
        ``plane_mix`` does not sum to 1.0, ``intensity_jitter`` is
        negative, or ``periphery_margin`` is outside ``[0, 1]``.
    """

    def __init__(
        self,
        volume_paths: list[str],
        axis: int = 1,
        plane_mix: tuple[float, float, float] = (0.5, 0.25, 0.25),
        *,
        augment: bool = True,
        intensity_jitter: float = 0.05,
        periphery_margin: float = 0.5,
        rng: np.random.Generator | None = None,
        in_memory: bool = False,
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
        if intensity_jitter < 0.0:
            raise ValueError(
                f"intensity_jitter must be >= 0, got intensity_jitter={intensity_jitter}"
            )
        if not (0.0 <= periphery_margin <= 1.0):
            raise ValueError(
                f"periphery_margin must be in [0, 1], got periphery_margin={periphery_margin}"
            )
        self.volume_paths: list[str] = list(volume_paths)
        self.axis: int = axis
        self.plane_mix: tuple[float, float, float] = plane_mix
        self.augment: bool = augment
        self.intensity_jitter: float = float(intensity_jitter)
        self.periphery_margin: float = float(periphery_margin)
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
        # When in_memory=True, each volume is materialized into RAM as a real
        # numpy array on first access (the real-run path -- the corpus zarrs
        # have 2048x2048 spatial chunks, so a dask patch read pulls the whole
        # slice from disk anyway; holding the volume in RAM makes patch
        # sampling pure in-memory slicing). The box has 499GB RAM; the 6-brain
        # 555nm corpus is ~108GB, which fits. Defaults to False (the tracer
        # and tests use dask, no RAM cost).
        self.in_memory: bool = bool(in_memory)
        # Cache the per-volume level-0 arrays so __getitem__ does not
        # re-resolve the NGFF component on every access. Lazily populated.
        # Under in_memory=True the cached value is a numpy array; otherwise a
        # dask array.
        self._volume_cache: dict[int, da.Array | np.ndarray] = {}

    def __len__(self) -> int:
        """Return the per-slice dataset length along the dominant axis.

        The length is the sum of ``volume.shape[axis]`` across all volumes
        (the dominant-axis slice count). ``__getitem__`` maps an index into
        ``(volume_idx, slice_idx_along_dominant_axis)``; the actual axis
        extracted is sampled stochastically per ``plane_mix`` (D-03), with
        the slice index clipped to the chosen axis's size.

        Returns
        -------
        int
            The total number of dominant-axis slices across all volumes.
        """
        total = 0
        for vi, _path in enumerate(self.volume_paths):
            vol = self._get_volume(vi)
            total += int(vol.shape[self.axis])
        return total

    def _get_volume(self, vol_idx: int) -> da.Array:
        """Resolve and cache the level-0 dask array for one OME-Zarr store.

        Reuses :func:`_load_volume` (which reuses ``_level0_component``)
        so the NGFF v0.5 ``s0`` vs v0.4 ``0`` resolution naming is handled
        by the existing resolver. The result is cached per-volume so
        repeated ``__getitem__`` calls do not re-resolve.

        Returns
        -------
        da.Array | np.ndarray
            The level-0 ``(C, Z, Y, X)`` array for the store. A dask array
            by default; a real numpy array when ``in_memory=True`` (the
            real-run path -- materializes the volume into RAM so patch
            sampling is pure in-memory slicing instead of disk reads).
        """
        if vol_idx not in self._volume_cache:
            vol = self._load_volume(self.volume_paths[vol_idx])
            if self.in_memory:
                # Materialize the full volume into RAM. This is a genuine
                # boundary (the corpus zarrs have 2048x2048 spatial chunks,
                # so a dask patch read pulls the whole slice from disk
                # anyway; holding the volume in RAM makes patch sampling
                # pure in-memory slicing). The box has enough RAM for the
                # real corpus (~108GB for 6 brains vs 499GB RAM).
                vol = np.asarray(vol.compute())
            self._volume_cache[vol_idx] = vol
        return self._volume_cache[vol_idx]

    def _sample_axis(self, rng: np.random.Generator) -> int:
        """Sample a plane axis from the categorical ``plane_mix`` distribution.

        Returns one of ``1`` (coronal), ``2`` (sagittal), ``3`` (axial)
        weighted by ``self.plane_mix`` (D-03 — 50/25/25 by default). This is
        the per-sample plane-axis draw; ``__getitem__`` calls it to decide
        which axis to extract the slice along.

        Parameters
        ----------
        rng : np.random.Generator
            The random generator to draw from.

        Returns
        -------
        int
            ``1``, ``2``, or ``3`` (coronal, sagittal, axial).
        """
        # axis indices are 1=coronal, 2=sagittal, 3=axial; plane_mix is
        # (coronal, sagittal, axial) so the categorical draw over [0,1,2]
        # maps directly to [1,2,3].
        return int(rng.choice([1, 2, 3], p=list(self.plane_mix)))

    def _sample_slice_index(
        self,
        axis_size: int,
        rng: np.random.Generator,
    ) -> int:
        """Sample a slice index along an axis, brain-centered with a periphery margin.

        The index is drawn from a window centered on the brain center (the
        middle of the axis by default — a real caller can override the
        center via a brain mask) with a periphery margin so background
        slices are included (D-03c — do NOT crop to the brain mask). With
        ``periphery_margin=0.5`` (the default) the full range is sampled;
        a smaller margin biases toward the center while still including
        periphery slices.

        Parameters
        ----------
        axis_size : int
            The number of slices along the axis.
        rng : np.random.Generator
            The random generator to draw from.

        Returns
        -------
        int
            A slice index in ``[0, axis_size)``.

        Raises
        ------
        ValueError
            If ``axis_size`` is less than 1.
        """
        if axis_size < 1:
            raise ValueError(f"axis_size must be >= 1, got axis_size={axis_size}")
        center = axis_size // 2
        half_window = max(1, round(axis_size * self.periphery_margin / 2.0))
        lo = max(0, center - half_window)
        hi = min(axis_size, center + half_window + 1)
        if hi <= lo:
            lo, hi = 0, axis_size
        return int(rng.integers(lo, hi))

    def __getitem__(self, index: int) -> NDArray[np.floating]:
        """Return one z-scored, lightly-augmented multi-channel slice.

        Maps ``index`` into ``(volume_idx, slice_idx_along_dominant_axis)``,
        samples the actual plane axis stochastically per ``plane_mix``
        (D-03), clips the slice index to the chosen axis's size, extracts
        the 2D multi-channel slice (reusing :func:`extract_plane_slice`,
        keeping BOTH channels — D-03b), z-scores per-channel (D-03d, not
        contrast-limited adaptive histogram equalization), and applies light
        augmentation (D-03e) when ``augment=True``.

        The periphery is NEVER cropped (D-03c) — the full spatial extent of
        the slice is returned so background voxels are present alongside
        vessel-bearing tissue.

        Parameters
        ----------
        index : int
            The sample index in ``[0, len(self))``. Maps to a
            ``(volume, slice)`` pair; the actual axis is sampled
            stochastically.

        Returns
        -------
        NDArray
            A ``(C, H, W)`` z-scored (and optionally augmented) multi-
            channel slice. Both channels are preserved (D-03b).

        Raises
        ------
        ValueError
            If ``index`` is out of range (AGENTS section 2 — explicit
            failure, never an IndexError leak).
        """
        n = len(self)
        if index < 0 or index >= n:
            raise ValueError(
                f"SSLCorpus index {index} out of range [0, {n}) — the dataset "
                f"has {n} dominant-axis slices across {len(self.volume_paths)} volume(s)"
            )
        # Map index -> (volume_idx, slice_idx_along_dominant_axis).
        vol_idx = 0
        remaining = index
        for vi, _path in enumerate(self.volume_paths):
            vol = self._get_volume(vi)
            size_along_axis = int(vol.shape[self.axis])
            if remaining < size_along_axis:
                vol_idx = vi
                break
            remaining -= size_along_axis
        vol = self._get_volume(vol_idx)
        # Sample the slice: draw the plane axis (D-03 multi-plane mix) and a
        # brain-centered slice index (D-03c periphery margin), then extract +
        # z-score. A slice where a channel is constant (zero std) cannot be
        # z-scored (z_score_per_channel raises ValueError — AGENTS section 2,
        # no silent NaN). This happens on background-only slices where one
        # channel has no signal. Rather than producing a NaN, retry with a
        # different slice index (a slice with signal in both channels). After
        # max_slice_attempts, raise — the volume may be catastrophic-signal-
        # loss (mip_qc should have filtered it upstream).
        max_slice_attempts = 8
        last_err: Exception | None = None
        for _ in range(max_slice_attempts):
            axis = self._sample_axis(self._rng)
            axis_size = int(vol.shape[axis])
            slice_idx = self._sample_slice_index(axis_size, self._rng)
            # Extract the 2D multi-channel slice (reusing Plan-01 helper;
            # keeps BOTH channels — D-03b — do NOT index data[channel]).
            raw = extract_plane_slice(vol, axis=axis, index=slice_idx)
            try:
                # Z-score per-channel (D-03d — not contrast-limited adaptive
                # histogram equalization; do NOT import the histogram-eq
                # helper from vseg.utils).
                out = z_score_per_channel(raw)
            except ValueError as err:
                # Zero-std channel on this slice — retry with a different
                # slice. This is NOT the empty-corpus case; it is a single
                # degenerate slice. Surface the last error if all attempts
                # fail.
                last_err = err
                continue
            if self.augment:
                out = self._apply_light_aug(out)
            return out
        raise ValueError(
            f"SSLCorpus: could not z-score any slice in volume {vol_idx} "
            f"({self.volume_paths[vol_idx]}) after {max_slice_attempts} attempts "
            f"— every sampled slice had a zero-std channel (constant channel). "
            f"Filter the volume via mip_qc or drop the constant channel. "
            f"Last error: {last_err}"
        )

    def get_patch(self, patch_size: tuple[int, int]) -> NDArray[np.floating]:
        """Sample one z-scored, augmented random patch (efficient disk read).

        Like :meth:`__getitem__` but crops the patch region from the dask
        array BEFORE ``.compute()``, so only the ``patch_size`` region is
        read from disk (not the full 2048x2048 slice). This is the
        real-run path -- materializing a full slice per patch would make
        pretraining disk-bound (each slice is ~8MB; 800 slices/epoch is
        ~6.4GB of reads per epoch just to crop 512x512 patches).

        Samples a volume (round-robin via the dominant-axis index space),
        a plane axis (per ``plane_mix``), a brain-centered slice index, a
        random patch offset within the slice's spatial extent, reads only
        that patch region, z-scores it per-channel, and applies light aug.
        Retries on a zero-std (constant) patch up to ``max_slice_attempts``
        times (a degenerate background patch -- same policy as
        :meth:`__getitem__`).

        Parameters
        ----------
        patch_size : tuple[int, int]
            The ``(PH, PW)`` patch size to crop from the 2D slice.

        Returns
        -------
        NDArray
            A ``(C, PH, PW)`` z-scored (and optionally augmented) patch.

        Raises
        ------
        ValueError
            If ``patch_size`` is larger than the slice spatial extent, or
            no patch with signal could be sampled after the retry budget.
        """
        patch_h, patch_w = patch_size
        if patch_h < 1 or patch_w < 1:
            raise ValueError(f"patch_size must be positive, got {(patch_h, patch_w)}")
        n = len(self)
        max_slice_attempts = 8
        last_err: Exception | None = None
        for _attempt in range(max_slice_attempts):
            # Sample a volume + dominant-axis slice index (same mapping as
            # __getitem__).
            idx = int(self._rng.integers(0, n))
            vol_idx = 0
            remaining = idx
            for vi, _path in enumerate(self.volume_paths):
                vol = self._get_volume(vi)
                size_along_axis = int(vol.shape[self.axis])
                if remaining < size_along_axis:
                    vol_idx = vi
                    break
                remaining -= size_along_axis
            vol = self._get_volume(vol_idx)
            axis = self._sample_axis(self._rng)
            axis_size = int(vol.shape[axis])
            slice_idx = self._sample_slice_index(axis_size, self._rng)
            # The two spatial axes are the non-channel, non-plane axes.
            spatial_axes = [a for a in range(1, vol.ndim) if a != axis]
            h_axis, w_axis = spatial_axes[0], spatial_axes[1]
            h_size, w_size = int(vol.shape[h_axis]), int(vol.shape[w_axis])
            if h_size < patch_h or w_size < patch_w:
                raise ValueError(
                    f"slice spatial extent {(h_size, w_size)} smaller than "
                    f"patch_size {(patch_h, patch_w)} -- reduce --patch-size"
                )
            top = int(self._rng.integers(0, h_size - patch_h + 1))
            left = int(self._rng.integers(0, w_size - patch_w + 1))
            # Build the dask slice expression: full channel axis, the single
            # plane slice, and the patch crop on the two spatial axes. Only
            # this region is read from disk on .compute().
            sl = [slice(None)] * vol.ndim
            sl[axis] = slice_idx
            sl[h_axis] = slice(top, top + patch_h)
            sl[w_axis] = slice(left, left + patch_w)
            sliced = vol[tuple(sl)]
            # dask arrays need .compute(); numpy arrays are already concrete.
            raw = np.asarray(sliced.compute() if hasattr(sliced, "compute") else sliced)
            try:
                out = z_score_per_channel(raw)
            except ValueError as err:
                last_err = err
                continue
            if self.augment:
                out = self._apply_light_aug(out)
            return out
        raise ValueError(
            f"SSLCorpus.get_patch: could not z-score any patch in volume "
            f"{vol_idx} ({self.volume_paths[vol_idx]}) after "
            f"{max_slice_attempts} attempts -- every sampled patch had a "
            f"zero-std channel (constant patch). Last error: {last_err}"
        )

    def get_patch_gpu(self, patch_size: tuple[int, int], device: Any = None) -> Any:
        """Sample one z-scored patch read directly to GPU via GPUDirect Storage.

        Like :meth:`get_patch` but reads the uncompressed zarr chunk directly
        into GPU memory via kvikio (GPUDirect Storage), bypassing the host
        bounce-buffer + zstd decompression that dominates the dask path. A
        single 16MB slice read via GDS hits ~14.6 GB/s (vs ~4.7 GB/s host),
        and 8 random slices + crop take ~12ms (vs ~36ms host) -- a 3x
        speedup on the patch-read stage.

        Requires the uncompressed corpus (zarr v3, raw-bytes chunks, no
        compression) and kvikio + cupy installed. Falls back to the dask
        :meth:`get_patch` path + H->D copy when kvikio is unavailable, the
        volume is compressed, or the sampled plane is not coronal (axis=1 --
        sagittal/axial patches span multiple chunks along one spatial axis,
        which the current GDS path does not handle; the multi-plane mix still
        applies, only the GDS fast-path is coronal-biased).

        Parameters
        ----------
        patch_size : tuple[int, int]
            The ``(PH, PW)`` patch size to crop from the 2D slice.
        device : torch.device | None
            The target CUDA device. Defaults to ``torch.cuda.current_device()``.

        Returns
        -------
        torch.Tensor
            A ``(C, PH, PW)`` z-scored (and optionally augmented) patch on
            GPU. Both channels are preserved.

        Raises
        ------
        ValueError
            If ``patch_size`` is larger than the slice spatial extent, or
            no patch with signal could be sampled after the retry budget.
        """
        try:
            import kvikio  # ruff: ignore[unused-import] -- imported for the availability check
        except ImportError:
            # No kvikio -- fall back to the dask path + H->D copy.
            patch = self.get_patch(patch_size)
            if device is None:
                device = torch.device("cuda", torch.cuda.current_device())
            return torch.as_tensor(patch, dtype=torch.float32, device=device)

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        patch_h, patch_w = patch_size
        n = len(self)
        max_slice_attempts = 8
        last_err: Exception | None = None
        for _attempt in range(max_slice_attempts):
            idx = int(self._rng.integers(0, n))
            vol_idx = 0
            remaining = idx
            for vi, _path in enumerate(self.volume_paths):
                vol = self._get_volume(vi)
                size_along_axis = int(vol.shape[self.axis])
                if remaining < size_along_axis:
                    vol_idx = vi
                    break
                remaining -= size_along_axis
            vol = self._get_volume(vol_idx)
            axis = self._sample_axis(self._rng)
            axis_size = int(vol.shape[axis])
            slice_idx = self._sample_slice_index(axis_size, self._rng)
            spatial_axes = [a for a in range(1, vol.ndim) if a != axis]
            h_axis, w_axis = spatial_axes[0], spatial_axes[1]
            h_size, w_size = int(vol.shape[h_axis]), int(vol.shape[w_axis])
            if h_size < patch_h or w_size < patch_w:
                raise ValueError(
                    f"slice spatial extent {(h_size, w_size)} smaller than "
                    f"patch_size {(patch_h, patch_w)} -- reduce --patch-size"
                )
            top = int(self._rng.integers(0, h_size - patch_h + 1))
            left = int(self._rng.integers(0, w_size - patch_w + 1))
            # GDS fast path: coronal plane (axis=1) on an uncompressed v3
            # zarr. The chunk is (1, 1, 2048, 2048) -- one slice per chunk --
            # so a coronal patch reads exactly one chunk. Sagittal/axial
            # patches span multiple chunks along one spatial axis; fall back
            # to dask for those (the multi-plane mix still applies, only the
            # GDS fast-path is coronal-biased).
            if axis != 1 or not self._is_uncompressed(vol_idx):
                # Fall back to dask read + H->D copy for non-coronal or
                # compressed volumes.
                sl = [slice(None)] * vol.ndim
                sl[axis] = slice_idx
                sl[h_axis] = slice(top, top + patch_h)
                sl[w_axis] = slice(left, left + patch_w)
                sliced = vol[tuple(sl)]
                raw = np.asarray(sliced.compute() if hasattr(sliced, "compute") else sliced)
            else:
                raw_gpu = self._gds_read_coronal_slice(vol_idx, slice_idx, device)
                # Crop the patch on GPU (the slice is (C, H, W)).
                raw_gpu = raw_gpu[:, top : top + patch_h, left : left + patch_w]
                out_gpu = self._z_score_per_channel_gpu(raw_gpu)
                if self.augment:
                    out_gpu = self._apply_light_aug_gpu(out_gpu)
                return out_gpu
            try:
                out = z_score_per_channel(raw)
            except ValueError as err:
                last_err = err
                continue
            if self.augment:
                out = self._apply_light_aug(out)
            return torch.as_tensor(out, dtype=torch.float32, device=device)
        raise ValueError(
            f"SSLCorpus.get_patch_gpu: could not z-score any patch in volume "
            f"{vol_idx} ({self.volume_paths[vol_idx]}) after "
            f"{max_slice_attempts} attempts -- every sampled patch had a "
            f"zero-std channel (constant patch). Last error: {last_err}"
        )

    def _is_uncompressed(self, vol_idx: int) -> bool:
        """Check whether the volume's on-disk zarr is uncompressed (kvikio-compatible).

        Returns
        -------
        bool
            ``True`` if the volume's ``s0`` array has only the raw ``bytes``
            codec (no zstd/blosc/gzip compression).
        """
        import json
        from pathlib import Path

        path = Path(self.volume_paths[vol_idx])
        meta = path / "s0" / "zarr.json"
        if not meta.exists():
            return False
        try:
            with meta.open(encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            return False
        codecs = m.get("codecs", [])
        return all(c.get("name") == "bytes" for c in codecs)

    def _gds_read_coronal_slice(self, vol_idx: int, slice_idx: int, device: Any) -> Any:
        """Read one coronal slice (axis=1) directly to GPU via kvikio GDS.

        Reads the whole ``(1, 1, Y, X)`` chunk (one 16MB page-aligned read --
        GDS requires page-aligned offsets, so reading the full chunk and
        cropping on GPU is faster than an unaligned sub-region read) into a
        torch tensor on ``device`` via ``CuFile.pread`` (kvikio accepts torch
        tensors directly -- no cupy needed).

        The ``CuFile`` handle is cached per (volume, slice) with an LRU bound
        (``_GDS_CUFILE_CACHE_MAX``) so repeated reads of the same slice reuse
        the open handle without the per-read open/close overhead that made
        the linumpy ``read_zarr_via_kvikio`` slow (it opens a CuFile per
        chunk in a loop). The cache is bounded to avoid hitting the file-
        descriptor limit on a 2001-slice volume.

        Returns
        -------
        torch.Tensor
            The ``(1, Y, X)`` slice on GPU, in the on-disk dtype.
        """
        import json
        from pathlib import Path

        import kvikio

        path = Path(self.volume_paths[vol_idx])
        with (path / "s0" / "zarr.json").open(encoding="utf-8") as f:
            meta = json.load(f)
        shape = tuple(meta["shape"])
        chunks = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
        # Map the on-disk dtype to a torch dtype.
        np_dtype = np.dtype(meta["data_type"])
        torch_dtype = torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype
        chunk_path = path / "s0" / "c" / "0" / str(slice_idx) / "0" / "0"
        if not chunk_path.exists():
            return torch.zeros((shape[0], shape[2], shape[3]), dtype=torch_dtype, device=device)
        # Bounded LRU CuFile cache. Without a bound, a 2001-slice volume
        # would open 2001 file descriptors -- risky against the ulimit.
        cache_key = (vol_idx, slice_idx)
        if not hasattr(self, "_gds_cufile_cache"):
            self._gds_cufile_cache: dict[tuple[int, int], Any] = {}
        if cache_key not in self._gds_cufile_cache:
            if len(self._gds_cufile_cache) >= _GDS_CUFILE_CACHE_MAX:
                # Evict the oldest entry (FIFO -- dict preserves insertion
                # order; close the CuFile before dropping the reference).
                old_key = next(iter(self._gds_cufile_cache))
                self._gds_cufile_cache.pop(old_key).close()
            self._gds_cufile_cache[cache_key] = kvikio.CuFile(str(chunk_path), "r")
        cf = self._gds_cufile_cache[cache_key]
        buf = torch.empty(chunks, dtype=torch_dtype, device=device)
        chunk_nbytes = int(np.prod(chunks)) * np_dtype.itemsize
        cf.pread(buf, chunk_nbytes, file_offset=0).get()
        return buf[0]  # (1, Y, X) -- keep the channel dim

    def _z_score_per_channel_gpu(self, slice_2d: Any) -> Any:
        """Z-score normalize a (C, H, W) torch tensor per channel on GPU.

        Returns
        -------
        torch.Tensor
            The z-scored (C, H, W) float32 tensor on GPU.

        Raises
        ------
        ValueError
            If any channel has zero std (constant channel).
        """
        out = torch.empty(slice_2d.shape, dtype=torch.float32, device=slice_2d.device)
        for c in range(slice_2d.shape[0]):
            channel = slice_2d[c].to(torch.float64)
            std = float(channel.std())
            if std < 1e-12:
                raise ValueError(f"zero-std channel {c} cannot be z-scored (constant channel)")
            out[c] = (channel - float(channel.mean())) / std
        return out

    def _apply_light_aug_gpu(self, slice_2d: Any) -> Any:
        """Apply light augmentation on GPU (flips + 90deg rot + jitter).

        Returns
        -------
        torch.Tensor
            The augmented (C, H, W) tensor on GPU, contiguous.
        """
        out = slice_2d.clone()
        k = int(self._rng.integers(0, 4))
        if k:
            out = torch.rot90(out, k=k, dims=(-2, -1))
        if self._rng.random() < 0.5:
            out = torch.flip(out, dims=[-1])
        if self._rng.random() < 0.5:
            out = torch.flip(out, dims=[-2])
        if self.intensity_jitter > 0.0:
            jitter = torch.as_tensor(
                self._rng.normal(loc=0.0, scale=self.intensity_jitter, size=out.shape),
                dtype=out.dtype,
                device=out.device,
            )
            out = out + jitter
        return out.contiguous()

    def _apply_light_aug(self, slice_2d: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply light augmentation: random flips + 90deg rotations + mild jitter (D-03e).

        Augmentation is deliberately light — random flips + 90deg rotations
        (anatomy is roughly symmetric for vessel topology) + mild intensity
        jitter. No spatial deformation, no heavy intensity remap (those conflict
        with the reconstruction target; the pretext task itself is the
        primary regularizer). The rotation reuses the ``np.rot90(k, axes=
        (-2, -1))`` pattern from :func:`OmeZarrDataset.load_patch` so the
        channel dim is never rotated (only the spatial HxW plane).

        Parameters
        ----------
        slice_2d : NDArray
            A ``(C, H, W)`` z-scored multi-channel slice.

        Returns
        -------
        NDArray
            The augmented slice, same shape as the input, finite values.
        """
        out = np.array(slice_2d, copy=True)
        # Random 90deg rotation on the spatial plane (axes=(-2, -1) keeps
        # the channel dim axis 0 untouched — only HxW rotates).
        k = int(self._rng.integers(0, 4))
        if k:
            out = np.rot90(out, k=k, axes=(-2, -1))
        # Random horizontal + vertical flips on the spatial plane.
        if self._rng.random() < 0.5:
            out = np.flip(out, axis=-1)  # horizontal flip (W)
        if self._rng.random() < 0.5:
            out = np.flip(out, axis=-2)  # vertical flip (H)
        # Mild per-channel intensity jitter (D-03e — mild only). The jitter
        # is additive Gaussian on the z-scored data, so a small std keeps
        # it within the intensity relationships the reconstruction loss
        # relies on (no heavy intensity remap).
        if self.intensity_jitter > 0.0:
            jitter = self._rng.normal(loc=0.0, scale=self.intensity_jitter, size=out.shape).astype(
                out.dtype, copy=False
            )
            out = out + jitter
        # np.rot90 / np.flip return views with negative strides; copy to a
        # contiguous array so downstream torch.from_numpy is safe.
        return np.ascontiguousarray(out)

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
