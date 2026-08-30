"""Unit tests for the vessel-aware block masking transform (``ssl/masking.py``).

These tests verify the transform-level change that biases masked-inpainting
hole placement toward thin-structure regions via a Frangi/Hessian vesselness
probability map:

* ``vesselness_probability_map`` -- a normalized [0, 1] Frangi vesselness map
  that peaks on a thin bright vessel line and is near-zero on flat background.
  A background-only slice falls back to a uniform map (no raise, no NaN -- the
  D-02a fallback so background slices still produce a valid mask).
* ``vessel_aware_block_mask`` -- the vessel-aware block mask that samples hole
  centers from the Frangi probability map (biased toward vessel regions)
  rather than uniform-random. Over many samples, hole centers cluster toward
  the vessel region vs uniform. It composes with MONAI ``RandCoarseDropoutd``
  for the fill step (``spatial_size=(-1, H, W)`` preserves the channel dim);
  masked regions are filled with the fill value, unmasked regions unchanged.
  The callable signature matches ``pretrain.MaskTransform`` -- ``(B, C, H, W)
  torch.Tensor -> (masked_input, mask)`` -- so it plugs into the pretraining
  loop's ``mask_transform`` slot without editing ``pretrain.py``.

Every test body gates on ``pytest.importorskip`` as its FIRST line because
``masking.py`` lazy-imports torch (the ``[ai]`` extra) and MONAI (the
``[benchmark]`` extra). The importorskip lives in the body, never at module
top (pytest #9542 would skip the whole module).
"""

import numpy as np
import pytest


def test_vesselness_probability_map_peaks_on_vessel(synthetic_vessel_slice):
    """vesselness_probability_map peaks (argmax) on the thin bright vessel line.

    The synthetic_vessel_slice fixture is a (1, 16, 16) slice with a 1px-tall
    horizontal bright line at row 7. Frangi vesselness responds strongly to
    thin bright tube/ridge structures, so the probability map's argmax should
    land on (or adjacent to) the vessel row, and the map should be near-zero
    on the flat background corners.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.masking import vesselness_probability_map

    image_2d = synthetic_vessel_slice[0]  # (16, 16) single-channel 2D
    prob = vesselness_probability_map(image_2d, sigmas=(1, 2, 3))
    assert prob.shape == image_2d.shape, f"probability map shape {prob.shape} != input"
    assert prob.ndim == 2, "vesselness_probability_map must return a 2D map"
    # The map is a probability distribution (sums to 1) over the spatial grid.
    assert abs(float(prob.sum()) - 1.0) < 1e-6, f"prob map must sum to 1, got {float(prob.sum())}"
    # The peak (argmax) should be on the vessel row (row 7) -- Frangi responds
    # to the thin bright ridge. Allow row 6/7/8 (Frangi has a 1-2px response
    # halo around a 1px ridge).
    peak_row = int(np.unravel_index(np.argmax(prob), prob.shape)[0])
    assert peak_row in (6, 7, 8), f"vesselness peak row {peak_row} not on vessel row 7 (±1)"
    # Flat background corners should be near-zero (the vessel line carries the
    # probability mass). Pick the four corners, well away from row 7.
    for (r, c) in [(0, 0), (0, 15), (15, 0), (15, 15)]:
        assert float(prob[r, c]) < 1e-3, (
            f"background corner ({r},{c}) prob {float(prob[r, c])} not near-zero"
        )


def test_vesselness_probability_map_background_only_falls_back_to_uniform():
    """A background-only (constant) slice falls back to a uniform probability map.

    D-02a -- a slice with no vessel response must NOT raise and must NOT
    produce NaN. It returns a uniform map (every pixel equal probability) so
    hole-center sampling degrades gracefully to uniform-random placement.
    AGENTS section 2 -- no silent NaN fallback; the uniform map is the
    explicit, documented degenerate-case behavior.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.masking import vesselness_probability_map

    bg = np.zeros((16, 16), dtype=np.float32)
    prob = vesselness_probability_map(bg, sigmas=(1, 2, 3))
    assert prob.shape == bg.shape
    assert not np.any(np.isnan(prob)), "background-only map must not contain NaN"
    # Uniform: every pixel equal, sums to 1.
    assert abs(float(prob.sum()) - 1.0) < 1e-6
    expected = 1.0 / float(bg.size)
    assert float(prob.max() - prob.min()) < 1e-9, (
        f"uniform fallback must be flat, got max-min={float(prob.max() - prob.min())}"
    )
    assert abs(float(prob[0, 0]) - expected) < 1e-9


def test_vessel_aware_block_mask_hole_centers_cluster_toward_vessel(synthetic_vessel_slice):
    """The vessel-aware block mask biases hole centers toward the vessel region.

    Over many samples, hole centers drawn from the Frangi probability map
    cluster closer to the vessel line than uniform-random hole centers. This
    is the D-02a lever -- uniform masking under-samples thin vessels (vessel
    pixels are <1% of voxels); biasing hole placement toward vessel regions
    forces the reconstruction to focus on thin-structure regions.
    """
    pytest.importorskip("monai")
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.ssl.masking import (
        vessel_aware_block_mask,
        vesselness_probability_map,
    )

    image_2d = synthetic_vessel_slice[0]  # (16, 16), vessel at row 7
    vessel_row = 7
    # Vessel mask: the set of pixels on the vessel line (for distance scoring).
    vessel_pixels = np.argwhere(image_2d >= 0.5)  # (N, 2) (row, col) on the line

    prob = vesselness_probability_map(image_2d, sigmas=(1, 2, 3))
    rng = np.random.default_rng(0)

    # Sample hole centers from the Frangi-biased map (N samples) and from a
    # uniform map (N samples). Compare mean distance from the sampled center
    # to the nearest vessel pixel. The Frangi-biased sample should be closer.
    n_samples = 200
    flat_prob = prob.ravel()
    flat_prob = flat_prob / flat_prob.sum()  # renormalize for np.random.choice
    uniform = np.ones_like(flat_prob) / flat_prob.size

    biased_idx = rng.choice(flat_prob.size, size=n_samples, p=flat_prob)
    uniform_idx = rng.choice(flat_prob.size, size=n_samples, p=uniform)

    def mean_dist_to_vessel(indices):
        coords = np.unravel_index(indices, image_2d.shape)  # (rows, cols)
        rows = np.asarray(coords[0])[:, None]
        cols = np.asarray(coords[1])[:, None]
        vp = vessel_pixels.astype(float)  # (Nv, 2)
        # Min Euclidean distance from each sampled center to any vessel pixel.
        d = np.sqrt((rows - vp[:, 0]) ** 2 + (cols - vp[:, 1]) ** 2)
        return float(d.min(axis=1).mean())

    biased_mean = mean_dist_to_vessel(biased_idx)
    uniform_mean = mean_dist_to_vessel(uniform_idx)
    assert biased_mean < uniform_mean, (
        f"vessel-aware hole centers should cluster closer to the vessel than "
        f"uniform (biased={biased_mean:.3f} vs uniform={uniform_mean:.3f})"
    )

    # Also exercise the full transform end-to-end on a batched tensor to
    # confirm the pluggable-callable contract holds (B, C, H, W) -> tuple.
    batch = torch.from_numpy(synthetic_vessel_slice)[None]  # (1, 1, 16, 16)
    masked_input, mask = vessel_aware_block_mask(
        batch, mask_ratio=0.25, block_size=(2, 2), fill_value=-1.0e6, prob=1.0
    )
    assert masked_input.shape == batch.shape, "masked input shape must equal input shape"
    assert mask.shape == batch.shape, "mask shape must equal input shape"
    assert mask.dtype == torch.bool, f"mask must be boolean, got {mask.dtype}"


def test_vessel_aware_block_mask_preserves_channel_dim_and_fills(synthetic_vessel_slice):
    """The composed MONAI RandCoarseDropoutd fill preserves the channel dim.

    ``spatial_size=(-1, H, W)`` keeps the channel axis (the ``-1`` means do
    not dropout the channel dim), so a (B, C, H, W) input yields a
    (B, C, H, W) masked output. Masked regions are filled with ``fill_value``
    (then zero-filled for the network input); unmasked regions are unchanged.
    This is the channel-preserving block-masking contract the pretraining
    loop's reconstruction MSE relies on.
    """
    pytest.importorskip("monai")
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.ssl.masking import vessel_aware_block_mask

    # 2-channel slice (D-03b -- both 555nm + 647nm channels preserved).
    sl = np.repeat(synthetic_vessel_slice, 2, axis=0)  # (2, 16, 16)
    batch = torch.from_numpy(sl)[None]  # (1, 2, 16, 16)
    fill_value = -1.0e6
    masked_input, mask = vessel_aware_block_mask(
        batch, mask_ratio=0.5, block_size=(4, 4), fill_value=fill_value, prob=1.0
    )
    # Channel dim preserved (no data[channel] drop).
    assert masked_input.ndim == 4, f"masked input ndim {masked_input.ndim} != 4"
    assert masked_input.shape[1] == 2, (
        f"channel dim must be preserved (D-03b), got C={masked_input.shape[1]}"
    )
    assert masked_input.shape == batch.shape, "shape must be unchanged"
    # At least some elements must be masked (prob=1.0, mask_ratio=0.5).
    assert int(mask.sum()) > 0, "no elements masked -- fill step dropped nothing"
    # Unmasked regions are unchanged (the network input equals the original
    # where the mask is False).
    unmasked_equal = torch.where(~mask, masked_input == batch, torch.ones_like(mask))
    assert bool(unmasked_equal.all()), "unmasked regions must equal the original input"
    # Masked regions in the network input are zero-filled (not the sentinel).
    masked_zero = torch.where(mask, masked_input == 0, torch.ones_like(mask))
    assert bool(masked_zero.all()), "masked regions in the network input must be zero-filled"


def test_vessel_aware_block_mask_background_only_no_raise():
    """A background-only batch does not raise and produces a valid mask.

    The Frangi map falls back to uniform on a background-only slice, so the
    vessel-aware transform degrades gracefully to uniform block masking -- no
    raise, no NaN. This is the D-02a degenerate-case contract.
    """
    pytest.importorskip("monai")
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.ssl.masking import vessel_aware_block_mask

    bg = torch.zeros((1, 2, 16, 16), dtype=torch.float32)
    masked_input, mask = vessel_aware_block_mask(
        bg, mask_ratio=0.25, block_size=(2, 2), fill_value=-1.0e6, prob=1.0
    )
    assert masked_input.shape == bg.shape
    assert not torch.any(torch.isnan(masked_input)), "background-only must not produce NaN"
    assert mask.dtype == torch.bool


def test_vessel_aware_block_mask_rejects_non_4d():
    """vessel_aware_block_mask raises ValueError on a non-4D tensor.

    AGENTS section 2 -- explicit failure with the offending value, never a
    silent wrong-shape return or an IndexError/AttributeError leak.
    """
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.ssl.masking import vessel_aware_block_mask

    bad = torch.zeros((2, 16, 16), dtype=torch.float32)  # 3D, not (B, C, H, W)
    with pytest.raises(ValueError):
        vessel_aware_block_mask(bad, mask_ratio=0.25, block_size=(2, 2))
