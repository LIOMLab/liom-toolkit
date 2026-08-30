"""Unit tests for the SSL corpus builder (``liom_toolkit/segmentation/vseg/ssl/corpus.py``).

These tests verify the tracer-tiny SSL corpus layer that Plan 02's
masked-inpainting pretraining loop consumes:

* ``extract_plane_slice`` — multi-plane slice extraction from a (C, Z, Y, X)
  dask volume along axis 1 (coronal/Z), 2 (sagittal/Y), 3 (axial/X). The
  channel dim is preserved (D-03b — both 555nm + 647nm channels are kept;
  the analog's ``self.data[channel]`` channel-drop is wrong here).
* ``z_score_per_channel`` — per-channel z-score normalization (D-03d —
  z-score, NOT CLAHE; CLAHE's non-linear remapping destroys the intensity
  relationships the inpainting reconstruction loss relies on). A zero-std
  channel raises ``ValueError`` naming the channel (no silent NaN escape —
  AGENTS section 2).
* ``mip_qc`` — max-intensity-projection quality check that flags
  catastrophic-signal-loss brains for skip.
* ``SSLCorpus`` — the corpus builder takes ``volume_paths`` / ``axis`` /
  ``plane_mix`` as PARAMETERS (AGENTS section 1 — no hardcoded ``/data/LSFM``
  default).

Every test body gates on ``pytest.importorskip("torch")`` as its FIRST line
because ``corpus.py`` module-top imports torch (the lazy-import guard raises
``ImportError("install liom-toolkit[ai,benchmark]")`` on a core-only
install). The importorskip lives in the body, never at module top (pytest
#9542 would skip the whole module including the non-torch tests).
"""

import dask.array as da
import numpy as np
import pytest


def test_extract_plane_slice_coronal(synthetic_ome_zarr):
    """extract_plane_slice along axis=1 (coronal/Z) returns a (C, Y, X) numpy slice.

    The synthetic_ome_zarr fixture is a (2, 8, 16, 16) volume; slicing along
    axis 1 (Z) at any valid index yields a (2, 16, 16) array with the channel
    dim preserved as axis 0 (D-03b — do NOT index away the channel dim).
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.dataset import _level0_component
    from liom_toolkit.segmentation.vseg.ssl.corpus import extract_plane_slice

    comp = _level0_component(synthetic_ome_zarr)
    vol = da.from_zarr(synthetic_ome_zarr, component=comp)
    sl = extract_plane_slice(vol, axis=1, index=4)
    assert sl.shape == (2, 16, 16), f"coronal slice shape {sl.shape} != (2, 16, 16)"
    assert sl.ndim == 3, f"slice ndim {sl.ndim} != 3 (channel dim must be preserved)"
    assert isinstance(sl, np.ndarray), "extract_plane_slice must return a numpy array"


def test_extract_plane_slice_sagittal(synthetic_ome_zarr):
    """extract_plane_slice along axis=2 (sagittal/Y) returns a (C, Z, X) slice."""
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.dataset import _level0_component
    from liom_toolkit.segmentation.vseg.ssl.corpus import extract_plane_slice

    comp = _level0_component(synthetic_ome_zarr)
    vol = da.from_zarr(synthetic_ome_zarr, component=comp)
    sl = extract_plane_slice(vol, axis=2, index=5)
    assert sl.shape == (2, 8, 16), f"sagittal slice shape {sl.shape} != (2, 8, 16)"
    assert sl.ndim == 3


def test_extract_plane_slice_axial(synthetic_ome_zarr):
    """extract_plane_slice along axis=3 (axial/X) returns a (C, Z, Y) slice."""
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.dataset import _level0_component
    from liom_toolkit.segmentation.vseg.ssl.corpus import extract_plane_slice

    comp = _level0_component(synthetic_ome_zarr)
    vol = da.from_zarr(synthetic_ome_zarr, component=comp)
    sl = extract_plane_slice(vol, axis=3, index=7)
    assert sl.shape == (2, 8, 16), f"axial slice shape {sl.shape} != (2, 8, 16)"
    assert sl.ndim == 3


def test_extract_plane_slice_invalid_axis(synthetic_ome_zarr):
    """extract_plane_slice raises ValueError on an out-of-range axis.

    AGENTS section 2 — explicit failure with the offending value, never
    a silent wrong-shape return or an IndexError leak.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.dataset import _level0_component
    from liom_toolkit.segmentation.vseg.ssl.corpus import extract_plane_slice

    comp = _level0_component(synthetic_ome_zarr)
    vol = da.from_zarr(synthetic_ome_zarr, component=comp)
    with pytest.raises(ValueError):
        extract_plane_slice(vol, axis=0, index=0)
    with pytest.raises(ValueError):
        extract_plane_slice(vol, axis=4, index=0)


def test_z_score_per_channel_normalizes():
    """z_score_per_channel yields per-channel mean ~0 and std ~1.

    D-03d — z-score per-channel normalization (NOT CLAHE). Each channel is
    normalized independently so the (555nm, 647nm) intensity scales do not
    bleed into each other.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.corpus import z_score_per_channel

    rng = np.random.default_rng(0)
    sl = rng.normal(loc=100.0, scale=50.0, size=(2, 16, 16)).astype(np.float32)
    # Give the two channels different scales to confirm per-channel (not
    # global) normalization.
    sl[1] = sl[1] * 3.0 + 200.0
    out = z_score_per_channel(sl)
    assert out.shape == sl.shape
    for c in range(2):
        assert abs(float(out[c].mean())) < 1e-5, f"channel {c} mean not ~0"
        assert abs(float(out[c].std()) - 1.0) < 1e-5, f"channel {c} std not ~1"


def test_z_score_per_channel_zero_std_raises():
    """z_score_per_channel raises ValueError on a zero-std (constant) channel.

    AGENTS section 2 — no silent NaN/zero-filled fallback. The error must
    name the offending channel so the caller can act on it.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.corpus import z_score_per_channel

    sl = np.zeros((2, 16, 16), dtype=np.float32)
    sl[0] = 5.0  # channel 0 is constant -> zero std
    sl[1] = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
    with pytest.raises(ValueError) as excinfo:
        z_score_per_channel(sl)
    assert "0" in str(excinfo.value), (
        f"ValueError must name the zero-std channel, got: {excinfo.value!r}"
    )


def test_mip_qc_passes_normal_volume(synthetic_ome_zarr):
    """mip_qc returns True for a volume with a non-trivial max-intensity projection.

    The synthetic_ome_zarr fixture has a bright sphere on channel 0, so its
    MIP is well above any reasonable threshold — a normal brain.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.dataset import _level0_component
    from liom_toolkit.segmentation.vseg.ssl.corpus import mip_qc

    comp = _level0_component(synthetic_ome_zarr)
    vol = da.from_zarr(synthetic_ome_zarr, component=comp)
    assert mip_qc(vol) is True


def test_mip_qc_flags_near_zero_volume():
    """mip_qc returns False for a near-zero-projection volume (catastrophic signal loss).

    A volume of all zeros (or near-zero) represents the catastrophic-signal-
    loss case the corpus builder must skip — including it would feed the
    pretraining loop empty slices that teach the network nothing.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.corpus import mip_qc

    vol = da.from_array(np.zeros((2, 8, 16, 16), dtype=np.uint16), chunks=(1, 8, 16, 16))
    assert mip_qc(vol) is False


def test_ssl_corpus_constructor_takes_path_parameters(tmp_path):
    """SSLCorpus accepts volume_paths + axis + plane_mix as parameters (no hardcoded default).

    AGENTS section 1 — all corpus paths are parameters; there is no
    hardcoded ``/data/LSFM`` default. The constructor takes a list of
    zarr paths and the plane-mix tuple as explicit arguments.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.corpus import SSLCorpus

    # A single-element path list into tmp_path (no lab default anywhere).
    paths = [str(tmp_path / "brain0.zarr")]
    corpus = SSLCorpus(volume_paths=paths, axis=1, plane_mix=(0.5, 0.25, 0.25))
    assert corpus.volume_paths == paths
    assert corpus.axis == 1
    assert corpus.plane_mix == (0.5, 0.25, 0.25)


def test_ssl_corpus_rejects_invalid_plane_mix(tmp_path):
    """SSLCorpus raises ValueError on a plane_mix that does not sum to 1.0.

    The 50/25/25 coronal/sagittal/axial mix (D-03) must sum to 1.0; a
    mis-normalized mix would silently bias the sampler. Explicit failure
    with the offending value (AGENTS section 2).
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.corpus import SSLCorpus

    paths = [str(tmp_path / "brain0.zarr")]
    with pytest.raises(ValueError):
        SSLCorpus(volume_paths=paths, axis=1, plane_mix=(0.5, 0.5, 0.5))


# ---------------------------------------------------------------------------
# Plan 03 expansion: multi-plane 50/25/25 + 2-channel + brain-centered+
# periphery + light aug + z-score (NOT CLAHE). These tests exercise the
# expanded SSLCorpus.__getitem__ sampler that the tracer-tiny constructor
# (Plan 01) left as a stub.
# ---------------------------------------------------------------------------


def test_ssl_corpus_multi_plane_mix_within_tolerance(synthetic_ome_zarr):
    """The multi-plane sampler yields the 50/25/25 coronal/sagittal/axial mix (D-03).

    Over a large sample (N=1000), the per-axis proportions drawn by the
    sampler must be within +/-0.05 of (0.5, 0.25, 0.25). The sampler draws
    each sample's plane axis from a categorical distribution weighted by
    ``plane_mix``; a fresh seeded RNG makes the draw reproducible.
    """
    pytest.importorskip("torch")
    import numpy as np

    from liom_toolkit.segmentation.vseg.ssl.corpus import SSLCorpus

    corpus = SSLCorpus(
        volume_paths=[synthetic_ome_zarr],
        axis=1,
        plane_mix=(0.5, 0.25, 0.25),
    )
    rng = np.random.default_rng(42)
    counts = {1: 0, 2: 0, 3: 0}
    n = 1000
    for _ in range(n):
        axis = corpus._sample_axis(rng)
        counts[axis] += 1
    props = {ax: counts[ax] / n for ax in (1, 2, 3)}
    assert abs(props[1] - 0.5) < 0.05, f"coronal prop {props[1]:.3f} not within 0.05 of 0.5"
    assert abs(props[2] - 0.25) < 0.05, f"sagittal prop {props[2]:.3f} not within 0.05 of 0.25"
    assert abs(props[3] - 0.25) < 0.05, f"axial prop {props[3]:.3f} not within 0.05 of 0.25"


def test_ssl_corpus_getitem_preserves_both_channels(synthetic_ome_zarr):
    """The sampler output is (2, H, W) -- both channels preserved (D-03b).

    The channel dim is NOT indexed away (the analog's ``self.data[channel]``
    channel-drop is wrong here). Both 555nm + 647nm channels are kept so the
    encoder learns cross-channel vessel structure.
    """
    pytest.importorskip("torch")
    import numpy as np

    from liom_toolkit.segmentation.vseg.ssl.corpus import SSLCorpus

    corpus = SSLCorpus(
        volume_paths=[synthetic_ome_zarr],
        axis=1,
        plane_mix=(1.0, 0.0, 0.0),  # coronal-only for a deterministic shape
        augment=False,
        rng=np.random.default_rng(0),
    )
    sl = corpus[0]
    assert sl.ndim == 3, f"sampler output ndim {sl.ndim} != 3 (channel dim must be preserved)"
    assert sl.shape[0] == 2, f"channel dim must be 2 (D-03b), got C={sl.shape[0]}"
    # Coronal slice of a (2, 8, 16, 16) volume is (2, 16, 16).
    assert sl.shape[1] == 16 and sl.shape[2] == 16, f"coronal spatial shape {sl.shape[1:]}"


def test_ssl_corpus_getitem_includes_background_periphery(synthetic_ome_zarr):
    """Brain-centered sampling includes background voxels (periphery NOT cropped, D-03c).

    The sampler picks a slice index near the brain center but with a periphery
    margin so batches contain both vessel-bearing tissue AND truly empty
    background. The ship gate's FPR-on-empty metric rewards learning what
    non-vessel looks like; cropping to the brain mask would suppress exactly
    that signal. The synthetic_ome_zarr fixture has a centered sphere with
    empty corners, so a brain-centered+periphery sample must contain both
    bright (tissue) and dim (background) regions -- verified by the presence
    of both high and low z-scored values (the periphery is NOT cropped away).
    """
    pytest.importorskip("torch")
    import numpy as np

    from liom_toolkit.segmentation.vseg.ssl.corpus import SSLCorpus

    corpus = SSLCorpus(
        volume_paths=[synthetic_ome_zarr],
        axis=1,
        plane_mix=(1.0, 0.0, 0.0),  # coronal-only for determinism
        augment=False,  # disable aug so the tissue/bg contrast is clean
    )
    # Sample several slices and confirm at least one has both tissue + bg.
    # After z-scoring, "background" voxels are not literally zero -- they are
    # the low tail of the z-score distribution. The check is for a bimodal
    # spread: both clearly-bright (tissue) and clearly-dim (background)
    # voxels exist, which a crop-to-tissue slice would NOT have.
    found_both = False
    for idx in range(8):
        sl = corpus[idx]
        # Full spatial extent (no crop): coronal slice of (2,8,16,16) is (2,16,16).
        assert sl.shape == (2, 16, 16), f"slice shape {sl.shape} -- periphery must not be cropped"
        # Both tissue (bright) and background (dim) voxels: the slice has a
        # meaningful spread (std > 0) AND both high and low quantiles are
        # present (not a uniform tissue-only slab).
        for c in range(2):
            std = float(sl[c].std())
            if std < 1e-6:
                continue  # constant channel -- skip (z-score would have raised)
            q_low = float(np.quantile(sl[c], 0.05))
            q_high = float(np.quantile(sl[c], 0.95))
            if q_high - q_low > 1.0:
                # Both bright and dim regions exist -- tissue + background.
                found_both = True
                break
        if found_both:
            break
    assert found_both, (
        "brain-centered+periphery sampling must include both tissue (bright) "
        "AND background (dim) voxels -- the periphery is NOT cropped (D-03c)"
    )


def test_ssl_corpus_get_patch_returns_correct_shape_and_is_finite(synthetic_ome_zarr):
    """get_patch returns a (C, PH, PW) z-scored patch -- the real-run disk-efficient path.

    get_patch crops the patch region from the dask array BEFORE .compute(),
    so only the patch is read from disk (not the full slice). The output is
    (C, PH, PW), z-scored, and finite -- the same contract as __getitem__
    but patch-sized.
    """
    pytest.importorskip("torch")
    import numpy as np

    from liom_toolkit.segmentation.vseg.ssl.corpus import SSLCorpus

    corpus = SSLCorpus(
        volume_paths=[synthetic_ome_zarr],
        axis=1,
        plane_mix=(1.0, 0.0, 0.0),  # coronal-only for determinism
        augment=False,
        rng=np.random.default_rng(0),
    )
    patch = corpus.get_patch((8, 8))  # 8x8 patch from a 16x16 slice
    assert patch.ndim == 3, f"get_patch output ndim {patch.ndim} != 3"
    assert patch.shape[0] == 2, f"channel dim must be 2, got C={patch.shape[0]}"
    assert patch.shape[1:] == (8, 8), f"patch spatial shape {patch.shape[1:]} != (8, 8)"
    assert bool(np.isfinite(patch).all()), "get_patch output must be finite (no NaN)"


def test_ssl_corpus_get_patch_smaller_than_slice(synthetic_ome_zarr):
    """get_patch with a patch smaller than the slice spatial extent succeeds.

    The synthetic volume is (2, 8, 16, 16); a coronal slice is (2, 16, 16)
    so an 8x8 patch is valid. A patch larger than 16x16 raises ValueError
    (the patch cannot exceed the slice extent).
    """
    pytest.importorskip("torch")
    import numpy as np

    from liom_toolkit.segmentation.vseg.ssl.corpus import SSLCorpus

    corpus = SSLCorpus(
        volume_paths=[synthetic_ome_zarr],
        axis=1,
        plane_mix=(1.0, 0.0, 0.0),
        augment=False,
        rng=np.random.default_rng(1),
    )
    # Valid patch.
    p = corpus.get_patch((4, 4))
    assert p.shape[1:] == (4, 4)
    # Patch larger than the 16x16 slice extent raises.
    with pytest.raises(ValueError, match="smaller than"):
        corpus.get_patch((32, 32))


def test_ssl_corpus_getitem_light_aug_shape_and_finite(synthetic_ome_zarr):
    """Light aug (flips + 90deg rotations + mild jitter) preserves shape and stays finite (D-03e).

    Augmentation is limited to random flips + 90deg rotations + mild intensity
    jitter -- NO elastic warp, NO heavy intensity remap (those conflict with
    the reconstruction target). The augmented sample has the same shape as
    the input and finite values (no NaN/inf from the jitter).
    """
    pytest.importorskip("torch")
    import numpy as np

    from liom_toolkit.segmentation.vseg.ssl.corpus import SSLCorpus

    corpus = SSLCorpus(
        volume_paths=[synthetic_ome_zarr],
        axis=1,
        plane_mix=(1.0, 0.0, 0.0),
        augment=True,
        rng=np.random.default_rng(2),
    )
    sl = corpus[0]
    assert sl.ndim == 3 and sl.shape[0] == 2, "aug must preserve (C, H, W) shape"
    assert np.all(np.isfinite(sl)), "augmented sample must have finite values (no NaN/inf)"


def test_ssl_corpus_getitem_z_scores_per_channel(synthetic_ome_zarr):
    """The expanded sampler z-scores per-channel (D-03d, NOT CLAHE).

    Each channel is normalized independently (mean ~0, std ~1). CLAHE's
    non-linear remapping would destroy the intensity relationships the
    inpainting reconstruction loss relies on; the sampler must NOT apply
    CLAHE.
    """
    pytest.importorskip("torch")
    import numpy as np

    from liom_toolkit.segmentation.vseg.ssl.corpus import SSLCorpus

    corpus = SSLCorpus(
        volume_paths=[synthetic_ome_zarr],
        axis=1,
        plane_mix=(1.0, 0.0, 0.0),
        augment=False,  # disable aug so the z-score stats are clean
        rng=np.random.default_rng(3),
    )
    sl = corpus[0]
    for c in range(2):
        mean = float(sl[c].mean())
        std = float(sl[c].std())
        assert abs(mean) < 0.1, f"channel {c} mean {mean:.3f} not ~0 (z-scored)"
        assert 0.5 < std < 2.0, f"channel {c} std {std:.3f} not ~1 (z-scored)"


def test_ssl_corpus_getitem_rejects_out_of_range_index(synthetic_ome_zarr):
    """SSLCorpus.__getitem__ raises ValueError on an out-of-range index.

    AGENTS section 2 -- explicit failure with the offending value, never a
    silent IndexError leak.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.corpus import SSLCorpus

    corpus = SSLCorpus(
        volume_paths=[synthetic_ome_zarr],
        axis=1,
        plane_mix=(1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError):
        corpus[99]  # only 8 coronal slices in the synthetic volume
