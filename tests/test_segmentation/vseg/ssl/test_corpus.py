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
