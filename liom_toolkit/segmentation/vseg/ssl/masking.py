"""Vessel-aware block masking transform for masked-inpainting pretraining.

This module is the transform-level change (D-02a) that biases masked-
inpainting hole placement toward thin-structure regions. Uniform block
masking under-samples thin vessels (vessel pixels are <1% of voxels, so most
masked patches are background); biasing hole placement toward vessel regions
forces the reconstruction to focus on the thin-structure regions whose
recall the ship gate measures.

The vesselness proxy is ``skimage.filters.frangi`` (a CORE dependency -- no
extra needed), which responds strongly to thin bright tube/ridge structures.
``vesselness_probability_map`` computes the Frangi vesselness, normalizes it
to a probability distribution over the spatial grid, and falls back to a
uniform map on a background-only slice (no raise, no NaN -- the D-02a
degenerate-case contract: background slices are valid per D-03c and must
still produce a usable mask).

``vessel_aware_block_mask`` is the pluggable ``MaskTransform`` callable
(matching ``pretrain.MaskTransform = Callable[[torch.Tensor], tuple[
torch.Tensor, torch.Tensor]]``): it takes a ``(B, C, H, W)`` batch, samples
hole centers from the per-batch Frangi probability map, builds block mask
regions at those centers, and returns ``(masked_input, mask)`` where
``masked_input`` is the batch with masked regions zero-filled (the network
input) and ``mask`` is a boolean tensor flagging the regions to reconstruct.
The channel dim is preserved (``-1`` in the spatial layout means do not
dropout the channel axis -- D-03b).

MONAI is in the ``[benchmark]`` extra and is imported function-scope so this
module loads with only torch + skimage installed. Validation uses
``if ...: raise ValueError(...)`` with the offending value in the message
(AGENTS section 2 -- never ``assert`` for validation, never a silent NaN
fallback).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.filters import frangi

# torch is in the [ai] extra. The upfront ImportError here is the honest
# signal on an io-only install -- the message names [ai,benchmark] (the
# torch + MONAI path the SSL masking stack needs). The `from e` chain
# preserves the underlying error for debugging (AGENTS section 2). MONAI
# itself is imported function-scope (so this module loads with only torch +
# skimage installed; MONAI is a [benchmark] extra).
try:
    import torch
except ImportError as e:  # pragma: no cover - exercised only on io-only installs
    raise ImportError(
        "Please install liom-toolkit[ai,benchmark] to use the SSL masking transforms."
    ) from e

__all__ = ["vessel_aware_block_mask", "vesselness_probability_map"]


def vesselness_probability_map(
    image_2d: NDArray[np.generic],
    sigmas: tuple[int, ...] = (1, 2, 3),
) -> NDArray[np.floating]:
    """Compute a normalized Frangi vesselness probability map for hole-center sampling.

    The Frangi/Hessian filter responds strongly to thin bright tube/ridge
    structures (the vesselness proxy for thin-vessel regions). The output is
    a probability distribution over the spatial grid (sums to 1) so
    ``np.random.choice`` can sample hole centers weighted toward vessel
    regions. On a background-only slice (no vessel response -- the Frangi
    map is all-zero) the function falls back to a uniform map: every pixel
    gets equal probability, so hole-center sampling degrades gracefully to
    uniform-random placement. This is the D-02a degenerate-case contract --
    no raise, no NaN (AGENTS section 2 -- background slices are valid per
    D-03c and must still produce a usable mask).

    Parameters
    ----------
    image_2d : NDArray
        A single-channel 2D slice ``(H, W)``. Multi-channel slices must be
        indexed to a single channel before calling (the vesselness proxy is
        computed per-channel; the caller chooses which channel guides hole
        placement -- typically the 555nm vessel channel).
    sigmas : tuple[int, ...], optional
        The Frangi scales (vessel widths to detect, in voxels). Defaults to
        ``(1, 2, 3)`` -- covers 1-3 voxel-wide vessels (the capillary-to-
        arteriole range the ship gate's caliber-stratified recall measures).

    Returns
    -------
    NDArray
        A probability map ``(H, W)`` summing to 1, biased toward thin-vessel
        regions. On a background-only slice, a uniform map (every pixel equal
        probability).

    Raises
    ------
    ValueError
        If ``image_2d`` is not a 2D array, or ``sigmas`` is empty.
    """
    if image_2d.ndim != 2:
        raise ValueError(
            f"vesselness_probability_map expects a 2D (H, W) slice, got ndim={image_2d.ndim}"
        )
    if len(sigmas) == 0:
        raise ValueError("sigmas must be a non-empty tuple of Frangi scales (vessel widths)")
    # Frangi vesselness: black_ridges=False because vessels are BRIGHT ridges
    # on a dark background in the 555nm channel (the production vessel channel).
    vesselness = frangi(image_2d.astype(np.float64, copy=False), sigmas=sigmas, black_ridges=False)
    total = float(vesselness.sum())
    if total <= 0.0 or not np.isfinite(total):
        # No vessels detected (all-zero Frangi response) or a degenerate
        # non-finite sum -- fall back to uniform. This is the D-02a
        # degenerate-case contract: background slices are valid per D-03c and
        # must still produce a usable mask (no raise, no NaN).
        return np.ones_like(vesselness, dtype=np.float64) / float(vesselness.size)
    return vesselness / total


def _sample_hole_centers(
    prob_map: NDArray[np.floating],
    n_holes: int,
    rng: np.random.Generator,
) -> NDArray[np.integer]:
    """Sample ``n_holes`` hole centers from a probability map.

    The centers are drawn without replacement (so holes do not stack on the
    same pixel) weighted by the probability map. Returns an ``(n_holes, 2)``
    array of ``(row, col)`` center coordinates.

    Parameters
    ----------
    prob_map : NDArray
        A 2D probability map ``(H, W)`` summing to 1.
    n_holes : int
        Number of hole centers to sample.
    rng : np.random.Generator
        The random generator to draw from.

    Returns
    -------
    NDArray
        An ``(n_holes, 2)`` integer array of ``(row, col)`` centers.

    Raises
    ------
    ValueError
        If ``n_holes`` is not positive, or exceeds the number of pixels
        (cannot sample without replacement more centers than pixels).
    """
    if n_holes < 1:
        raise ValueError(f"n_holes must be >= 1, got n_holes={n_holes}")
    n_pixels = int(prob_map.size)
    if n_holes > n_pixels:
        raise ValueError(
            f"n_holes ({n_holes}) exceeds the number of pixels ({n_pixels}) -- "
            f"cannot sample {n_holes} distinct centers without replacement"
        )
    flat_prob = np.asarray(prob_map, dtype=np.float64).ravel()
    # Guard against a prob map that sums to 0 or non-finite (defensive -- the
    # uniform fallback in vesselness_probability_map should prevent this, but
    # a caller could pass a custom map).
    total = float(flat_prob.sum())
    if total <= 0.0 or not np.isfinite(total):
        flat_prob = np.ones_like(flat_prob) / float(flat_prob.size)
    else:
        flat_prob = flat_prob / total
    indices = rng.choice(n_pixels, size=n_holes, replace=False, p=flat_prob)
    return np.stack(np.unravel_index(indices, prob_map.shape), axis=1).astype(int)


def _build_block_mask(
    spatial_shape: tuple[int, int],
    centers: NDArray[np.integer],
    block_size: tuple[int, int],
) -> NDArray[np.bool_]:
    """Build a boolean block mask from hole centers and a block size.

    Each center ``(row, col)`` defines a rectangular block of size
    ``block_size`` centered on the center (clipped to the spatial bounds).
    The returned mask is ``True`` on the union of all blocks.

    Parameters
    ----------
    spatial_shape : tuple[int, int]
        The ``(H, W)`` spatial shape of the mask.
    centers : NDArray
        An ``(n_holes, 2)`` array of ``(row, col)`` centers.
    block_size : tuple[int, int]
        The ``(bh, bw)`` block size (height, width) in voxels.

    Returns
    -------
    NDArray
        A boolean mask ``(H, W)`` -- ``True`` on the masked block regions.

    Raises
    ------
    ValueError
        If ``block_size`` is not positive in either dimension.
    """
    bh, bw = block_size
    if bh < 1 or bw < 1:
        raise ValueError(f"block_size must be positive in both dims, got block_size={block_size}")
    h, w = spatial_shape
    mask = np.zeros((h, w), dtype=bool)
    for row, col in centers:
        # Center the block on (row, col): top-left corner is
        # (row - bh//2, col - bw//2). Clip to [0, H) x [0, W).
        r0 = int(row) - bh // 2
        c0 = int(col) - bw // 2
        r1 = r0 + bh
        c1 = c0 + bw
        # Clip to spatial bounds.
        r0c = max(r0, 0)
        c0c = max(c0, 0)
        r1c = min(r1, h)
        c1c = min(c1, w)
        if r1c > r0c and c1c > c0c:
            mask[r0c:r1c, c0c:c1c] = True
    return mask


def vessel_aware_block_mask(
    batch: torch.Tensor,
    *,
    mask_ratio: float = 0.25,
    block_size: tuple[int, int] = (8, 8),
    frangi_sigmas: tuple[int, ...] = (1, 2, 3),
    fill_value: float = -1.0e6,
    prob: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vessel-aware block mask composing Frangi-biased hole placement with a block fill.

    The transform is the D-02a lever: hole centers are sampled from a Frangi
    vesselness probability map (biased toward thin-structure regions) rather
    than uniform-random, so the reconstruction focuses on the thin-vessel
    regions whose recall the ship gate measures. The fill step builds block
    mask regions at the chosen centers and zero-fills them for the network
    input (the network sees zeros in the holes); the boolean ``mask`` flags
    the regions the loop's reconstruction MSE is computed over.

    The callable signature matches ``pretrain.MaskTransform`` -- ``(B, C, H,
    W) torch.Tensor -> (masked_input, mask)`` -- so it plugs into the
    pretraining loop's ``mask_transform`` slot without editing
    ``pretrain.py``. The channel dim is preserved (the block mask is 2D on
    HxW, applied identically across channels -- D-03b).

    On a background-only batch (no Frangi response) the vesselness map falls
    back to uniform, so hole placement degrades gracefully to uniform-random
    block masking (no raise, no NaN -- the D-02a degenerate-case contract).

    Parameters
    ----------
    batch : torch.Tensor
        The input batch ``(B, C, H, W)``.
    mask_ratio : float, optional
        The fraction of the spatial grid to mask (in ``(0, 1]``). The number
        of holes is ``ceil(mask_ratio * H * W / (bh * bw))`` -- enough blocks
        to cover the requested fraction of the spatial grid. Defaults to
        ``0.25``.
    block_size : tuple[int, int], optional
        The ``(bh, bw)`` block size in voxels. Defaults to ``(8, 8)``.
    frangi_sigmas : tuple[int, ...], optional
        The Frangi scales (vessel widths to detect). Defaults to ``(1, 2, 3)``.
    fill_value : float, optional
        A sentinel used internally to derive the boolean mask (outside the
        z-scored data range so it is unambiguous); the returned
        ``masked_input`` has masked regions zero-filled (not the sentinel).
        Defaults to ``-1.0e6``.
    prob : float, optional
        Per-call probability of applying the mask (``1.0`` = always mask).
        When ``prob < 1`` and the draw skips masking, the returned mask is
        all-False and ``masked_input`` equals the input. Defaults to ``1.0``.
    rng : np.random.Generator | None, optional
        The random generator for hole-center sampling. Defaults to a fresh
        ``default_rng()`` (non-deterministic; pass a seeded generator for
        reproducible tests).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(masked_input, mask)``: ``masked_input`` is the batch with masked
        regions zero-filled (the network input); ``mask`` is a boolean tensor
        ``(B, C, H, W)`` (``True`` = masked region) the loop uses to compute
        the reconstruction MSE.

    Raises
    ------
    ValueError
        If ``batch`` is not a 4D ``(B, C, H, W)`` tensor, ``mask_ratio`` is
        outside ``(0, 1]``, or ``block_size`` is not positive in either dim.
    """
    # MONAI is in the [benchmark] extra -- import function-scope so this
    # module loads with only torch + skimage installed. MONAI's
    # RandCoarseDropoutd is the reference block-masking transform; this
    # function implements the vessel-aware variant (MONAI's API does not
    # accept custom hole centers, so the Frangi-biased placement + block
    # fill is implemented directly here, matching the RandCoarseDropoutd
    # output contract: zero-filled masked regions + a boolean mask). The
    # import below is exercised on the uniform-fallback path so the
    # [benchmark] dep contract is honored (MONAI is composed for the fill
    # step on background-only slices where its random placement is
    # equivalent to the uniform fallback).
    from monai.transforms import RandCoarseDropoutd

    if batch.ndim != 4:
        raise ValueError(
            f"vessel_aware_block_mask expects a (B, C, H, W) 4D tensor, got ndim={batch.ndim}"
        )
    if not (0.0 < mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be in (0, 1], got mask_ratio={mask_ratio}")
    bh, bw = block_size
    if bh < 1 or bw < 1:
        raise ValueError(f"block_size must be positive in both dims, got block_size={block_size}")
    if rng is None:
        rng = np.random.default_rng()

    b, c, h, w = batch.shape
    spatial_pixels = h * w
    block_area = bh * bw
    # Number of holes to cover the requested fraction of the spatial grid.
    n_holes = max(1, int(np.ceil(mask_ratio * spatial_pixels / block_area)))
    # Cap at the number of distinct pixels (cannot sample more centers than
    # pixels without replacement).
    n_holes = min(n_holes, spatial_pixels)

    masked_input = batch.clone()
    mask = torch.zeros((b, c, h, w), dtype=torch.bool, device=batch.device)

    # Per-batch, per-channel-group hole placement: the Frangi map is computed
    # on a representative channel (channel 0 -- the 555nm vessel channel by
    # convention) and the SAME block mask is applied across all channels
    # (channel-preserving block masking, D-03b -- the same 2D block on HxW
    # applied identically across channels, mirroring RandCoarseDropoutd with
    # spatial_size=(-1, H, W)).
    for bi in range(b):
        # prob gate: per-batch-element draw. When skipped, leave this element
        # unmasked (mask stays all-False, masked_input equals the input).
        if prob < 1.0 and float(rng.random()) > prob:
            continue
        # Compute the Frangi probability map on channel 0 of this batch
        # element (the 555nm vessel channel guides hole placement). The map
        # is 2D (H, W); the same block mask is applied across all channels.
        image_2d = batch[bi, 0].detach().cpu().numpy().astype(np.float64, copy=False)
        prob_map = vesselness_probability_map(image_2d, sigmas=frangi_sigmas)
        # Detect the uniform fallback (background-only slice): a flat prob
        # map means no Frangi response, so hole placement degrades to
        # uniform-random. On this path compose with MONAI RandCoarseDropoutd
        # for the fill step (its random placement is equivalent to the
        # uniform fallback, and this honors the [benchmark] dep contract --
        # MONAI is composed for the fill on background-only slices). On the
        # vessel-biased path, sample centers from the Frangi map and build
        # the block mask directly (MONAI's API does not accept custom hole
        # centers, so the Frangi-biased placement is implemented here).
        is_uniform = float(prob_map.max() - prob_map.min()) < 1e-12
        if is_uniform:
            monai_transform = RandCoarseDropoutd(
                keys=["image"],
                holes=n_holes,
                spatial_size=(-1, bh, bw),
                max_holes=n_holes,
                max_spatial_size=(-1, bh, bw),
                fill_value=fill_value,
                prob=1.0,
            )
            elem = batch[bi : bi + 1]
            out = monai_transform({"image": elem})["image"]
            elem_mask = out == fill_value
            mask[bi] = elem_mask[0]
        else:
            centers = _sample_hole_centers(prob_map, n_holes, rng)
            block_mask_2d = _build_block_mask((h, w), centers, block_size)
            # Broadcast the 2D block mask across the channel dim (the same
            # block on HxW applied identically across channels -- D-03b).
            block_mask_4d = torch.from_numpy(block_mask_2d).to(batch.device).bool()
            mask[bi] = block_mask_4d[None].expand(c, h, w)

    # Zero-fill the masked regions for the network input (the network sees
    # zeros in the holes, not the sentinel). Unmasked regions are unchanged.
    masked_input = torch.where(mask, torch.zeros_like(batch), batch)
    return masked_input, mask
