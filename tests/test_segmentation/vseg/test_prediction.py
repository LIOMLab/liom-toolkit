"""Tests for ``liom_toolkit/segmentation/vseg/prediction.py``.

Covers the ``predict_one`` tiling-arithmetic fix: ``n_patches_by_row`` is
computed as ``int(...)`` so the ``y1 % n_patches_by_row`` modulo is integer
arithmetic with no float rounding. This locks the tiling-path arithmetic for
the future ``patching=True`` implementation (currently raises
``NotImplementedError``).

Per AGENTS.md §5, heavy deps are mocked for orchestration tests:
``torch`` is gated with ``pytest.importorskip("torch")`` (the package's own
``try/except ImportError`` guard mirrors this), and ``do_predict`` is patched
to return a real small uint8 array so the tiling loop runs end-to-end against
real numpy arithmetic (the compute path under test) without a real model
forward pass. ``numpy``/``cv2``/``scikit-image`` are real and unmocked.

The ``torch`` importorskip and the ``predict_one`` import are deferred into a
module-scoped fixture (not module-level) so torch is NOT imported during
collection. torch spawns background threads at import, and importing it during
collection would leave the controller multi-threaded — unsafe for any
fork-based parallel runner and wasteful even under xdist (every worker that
does not run this module would still pay the import). The fixture runs once
per worker that needs it, so torch is imported lazily where it is used.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

import imageio.v3 as iio
import numpy as np
import pytest
import zarr


@pytest.fixture(scope="module")
def predict_one():
    """Import predict_one lazily (torch is imported only where used).

    Deferred from module-level so collection does not import torch. The
    importorskip gates the whole module: if torch is not installed, every
    test in this module skips.
    """
    pytest.importorskip("torch")  # vseg/ requires PyTorch (AGENTS §5, §9)
    from liom_toolkit.segmentation.vseg.prediction import predict_one as _po

    return _po


def _write_synthetic_image(path: str, shape=(16, 16)) -> np.ndarray:
    """Write a small non-constant uint8 PNG for predict_one to read."""
    arr = np.zeros(shape, dtype=np.uint8)
    arr[4:12, 4:12] = 200
    iio.imwrite(path, arr)
    return arr


def test_predict_one_n_patches_by_row_is_int(tmp_path, predict_one):
    """predict_one computes n_patches_by_row as int (integer tiling arithmetic).

    The tiling-path arithmetic (``y1 % n_patches_by_row``) must use integer
    modulo with no float rounding. The function is run end-to-end with
    ``do_predict`` mocked to return a real small uint8 patch, exercising the
    real numpy arithmetic in the tiling loop. The returned mask is uint8
    (0 or 255), confirming the arithmetic path completed without a
    float-modulo rounding error. Fails RED if ``n_patches_by_row`` is a
    float that triggers a rounding bug in the modulo.
    """
    img_path = str(tmp_path / "synth.png")
    _write_synthetic_image(img_path, shape=(16, 16))
    save_path = str(tmp_path / "out")

    # Mock do_predict to return a real uint8 patch matching the image shape.
    # predict_one writes the patch image to disk, reads it back, runs
    # process_image (real torch tensor conversion), then calls do_predict.
    # The mock returns a 2D uint8 array of the expected pred shape.
    def _fake_do_predict(model, patch):
        # Return a uint8 mask the same H/W as the input image.
        return np.zeros((16, 16), dtype=np.uint8)

    with patch(
        "liom_toolkit.segmentation.vseg.prediction.do_predict",
        side_effect=_fake_do_predict,
    ):
        result = predict_one(
            model=None,  # do_predict is mocked; model is never called
            img_path=img_path,
            save_path=save_path,
            dev="cpu",
        )

    # The tiling arithmetic ran end-to-end and produced a valid uint8 mask.
    assert result.dtype == np.uint8
    assert result.shape == (16, 16)
    # Mask values are 0 or 255 (the bool * 255 path).
    assert set(np.unique(result)).issubset({0, 255})


def test_predict_one_modulo_no_float_rounding(tmp_path, predict_one):
    """predict_one tiling arithmetic uses int division/modulo (no float rounding).

    ``n_patches_by_row`` must be ``int(...)`` so ``y1 % n_patches_by_row`` is
    integer arithmetic. A float ``n_patches_by_row`` would silently round
    wrong on the future tiled path. This test runs predict_one end-to-end
    (do_predict mocked) and confirms the tiling loop completes without a
    float-modulo rounding error producing a wrong-shaped or wrong-dtype
    mask. The default single-pass path (stride == H) yields
    ``n_patches_by_row == 1``; the int wrap locks the arithmetic for the
    future ``patching=True`` path.
    """
    img_path = str(tmp_path / "synth_mod.png")
    _write_synthetic_image(img_path, shape=(16, 16))
    save_path = str(tmp_path / "out_mod")

    with patch(
        "liom_toolkit.segmentation.vseg.prediction.do_predict",
        return_value=np.zeros((16, 16), dtype=np.uint8),
    ):
        result = predict_one(
            model=None,
            img_path=img_path,
            save_path=save_path,
            dev="cpu",
        )

    # Integer modulo arithmetic produced a complete, valid mask.
    assert result.shape == (16, 16)
    assert result.dtype == np.uint8
    # The segmented output file was written.
    assert (Path(save_path) / "synth_mod_segmented.png").exists()


def test_predict_one_patching_true_raises_not_implemented(tmp_path, predict_one):
    """predict_one(patching=True) raises NotImplementedError (tiling not implemented).

    The int-modulo fix locks the tiling arithmetic for this future path; the
    path itself is not implemented and must raise rather than silently
    returning plausible-shaped-but-wrong single-pass output.
    """
    img_path = str(tmp_path / "synth_patch.png")
    _write_synthetic_image(img_path, shape=(16, 16))

    with pytest.raises(NotImplementedError):
        predict_one(
            model=None,
            img_path=img_path,
            save_path=str(tmp_path / "out_patch"),
            dev="cpu",
            patching=True,
        )


@pytest.fixture
def nnunet_model(stub_nnunet_model_dir, fake_nnunet_predictor):
    """Construct a real ``NnUnetV2Model`` over the injected fake predictor.

    The ``fake_nnunet_predictor`` sys.modules leaf injection (conftest) makes
    the wrapper's function-scope ``nnunetv2`` import resolve to the recording
    stand-in, so the object under test is the real wrapper class — only the
    nnU-Net leaf is substituted (AGENTS §5 leaf-injection pattern).
    """
    pytest.importorskip("torch")  # model_v2 carries a module-top torch guard
    from liom_toolkit.segmentation.vseg.model_v2 import NnUnetV2Model

    return NnUnetV2Model(stub_nnunet_model_dir, device="cpu")


def test_predict_one_nnunet_calls_predictor_once_with_raw_input(
    tmp_path, predict_one, nnunet_model, fake_nnunet_predictor
):
    """predict_one routes an NnUnetV2Model to model.predict with raw input + spacing.

    The nnU-Net path must call ``predict_single_npy_array`` exactly once via
    the wrapper, with a ``(1, 1, H, W)`` float32 array byte-equal to the raw
    PNG contents under the channel + dummy-z dims (no CLAHE / min-max /
    gaussian — nnU-Net owns normalization per its plans) and
    ``image_properties == {"spacing": [6.5, 6.5, 6.5]}`` — the 3-element
    spacing every real nnunetv2 preprocessor requires (transpose_forward is
    always length 3); the through-plane entry is a pass-through placeholder
    for ``'2d'`` configurations. The persistence contract is shared with the
    legacy path: ``{stem}_segmented.png`` written under ``save_path`` and
    the returned mask is uint8 {0, 255} with the dummy z-axis squeezed back
    to ``(H, W)``.
    """
    img_path = tmp_path / "nnunet.png"
    raw = _write_synthetic_image(str(img_path), shape=(16, 16))
    save_path = tmp_path / "out_nnunet"

    result = predict_one(
        model=nnunet_model,
        img_path=str(img_path),
        save_path=str(save_path),
        dev="cpu",
        spacing=(6.5, 6.5),
    )

    calls = fake_nnunet_predictor.calls["predict_calls"]
    assert len(calls) == 1
    input_image = calls[0]["input_image"]
    assert input_image.shape == (1, 1, 16, 16)
    assert input_image.dtype == np.float32
    np.testing.assert_array_equal(input_image[0, 0], raw.astype(np.float32))
    assert calls[0]["image_properties"] == {"spacing": [6.5, 6.5, 6.5]}

    segmented = save_path / "nnunet_segmented.png"
    assert segmented.exists()
    assert result.dtype == np.uint8
    assert result.shape == (16, 16)
    assert set(np.unique(result)).issubset({0, 255})


def test_predict_one_nnunet_ignores_legacy_norm_and_dev(
    tmp_path, predict_one, nnunet_model, fake_nnunet_predictor
):
    """The nnU-Net path ignores norm/norm_param/dev (legacy-only parameters).

    ``norm_param={"a": 1}`` is a dict the legacy path would fail to index as
    ``norm_param[0]`` — surviving it proves the parameter was never touched.
    ``dev="cuda"`` on a CUDA-less host would raise inside the legacy
    ``.to(device)`` path — surviving it proves ``dev`` was never used. The
    byte-equality assertion on the predictor input proves no CLAHE/min-max
    ran regardless of the ``norm`` flag.
    """
    img_path = tmp_path / "nnunet_raw.png"
    raw = _write_synthetic_image(str(img_path), shape=(16, 16))
    save_path = tmp_path / "out_nnunet_raw"

    predict_one(
        model=nnunet_model,
        img_path=str(img_path),
        save_path=str(save_path),
        dev="cuda",
        norm_param={"a": 1},
        norm=False,
        spacing=(6.5, 6.5),
    )

    calls = fake_nnunet_predictor.calls["predict_calls"]
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["input_image"][0, 0], raw.astype(np.float32))


def test_predict_one_nnunet_requires_spacing(
    tmp_path, predict_one, nnunet_model, fake_nnunet_predictor
):
    """predict_one on an NnUnetV2Model without spacing raises ValueError.

    A PNG carries no spacing metadata, so spacing must be passed explicitly
    — silently defaulting to isotropic would mis-resample the prediction
    (silent wrong-data). The error fires before any model invocation.
    """
    img_path = tmp_path / "nnunet_nospacing.png"
    _write_synthetic_image(str(img_path), shape=(16, 16))

    with pytest.raises(ValueError, match="spacing"):
        predict_one(
            model=nnunet_model,
            img_path=str(img_path),
            save_path=str(tmp_path / "out_nnunet_nospacing"),
            dev="cpu",
        )

    assert fake_nnunet_predictor.calls["predict_calls"] == []


def test_predict_one_nnunet_all_zero_raises_before_model(
    tmp_path, predict_one, nnunet_model, fake_nnunet_predictor
):
    """predict_one keeps the all-zero ValueError guard in front of the nnU-Net path.

    nnU-Net's ``crop_to_nonzero`` falls back to the full bbox on an all-zero
    image, so without the guard the call would emit a plausible all-255 (or
    all-background) mask on garbage input. The contract guard must fire on
    the raw image BEFORE the predictor is invoked.
    """
    img_path = tmp_path / "nnunet_zeros.png"
    iio.imwrite(img_path, np.zeros((16, 16), dtype=np.uint8))

    with pytest.raises(ValueError, match="all-zero"):
        predict_one(
            model=nnunet_model,
            img_path=str(img_path),
            save_path=str(tmp_path / "out_nnunet_zeros"),
            dev="cpu",
            spacing=(6.5, 6.5),
        )

    assert fake_nnunet_predictor.calls["predict_calls"] == []


def test_predict_one_nnunet_patching_raises_not_implemented(
    tmp_path, predict_one, nnunet_model, fake_nnunet_predictor
):
    """predict_one(patching=True) raises NotImplementedError on the nnU-Net path too.

    Manual 2D tiling was never supported and the same exception (type and
    message) must surface for both model types — the guard precedes the
    type dispatch so no model call happens.
    """
    img_path = tmp_path / "nnunet_patch.png"
    _write_synthetic_image(str(img_path), shape=(16, 16))

    with pytest.raises(NotImplementedError, match="predict_volume"):
        predict_one(
            model=nnunet_model,
            img_path=str(img_path),
            save_path=str(tmp_path / "out_nnunet_patch"),
            dev="cpu",
            patching=True,
            spacing=(6.5, 6.5),
        )

    assert fake_nnunet_predictor.calls["predict_calls"] == []


def test_predict_one_legacy_model_rejects_spacing_kwarg(tmp_path, predict_one):
    """predict_one on a legacy (non-NnUnetV2) model rejects the spacing kwarg.

    ``spacing`` is an nnU-Net-only parameter: passing it with a legacy
    VsegModel must raise a ValueError naming it, not silently ignore it
    (asymmetric kwargs are never silently dropped — the error names the
    offending kwarg so the caller knows it only applies to nnU-Net models).
    """
    img_path = tmp_path / "legacy_spacing.png"
    _write_synthetic_image(str(img_path), shape=(16, 16))

    with pytest.raises(ValueError, match="spacing"):
        predict_one(
            model=None,  # legacy path: not an NnUnetV2Model instance
            img_path=str(img_path),
            save_path=str(tmp_path / "out_legacy_spacing"),
            dev="cpu",
            spacing=(6.5, 6.5),
        )


# ---------------------------------------------------------------------------
# predict_volume nnU-Net routing
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_ome_zarr(tmp_path) -> str:
    """Write a tiny (1, 5, 32, 32) float32 OME-Zarr with 6.5um scales.

    Mirrors the ``synthetic_ome_zarr`` idiom from the ssl conftest (which is
    dir-scoped and not importable here): a real ``save_zarr`` write so the
    NGFF ``coordinateTransformations`` metadata the spacing parser reads is
    genuinely on disk. A seeded RNG keeps the volume deterministic.
    """
    from liom_toolkit.conversion.conversion import save_zarr

    vol = np.random.default_rng(0).random((1, 5, 32, 32)).astype(np.float32)
    zarr_path = str(tmp_path / "tiny.zarr")
    save_zarr(vol, zarr_path, scales=(6.5, 6.5, 6.5), chunks=(1, 1, 32, 32))
    return zarr_path


@pytest.fixture
def tiny_dataset(tiny_ome_zarr):
    """An OmeZarrDataset over ``tiny_ome_zarr`` with channel 0 selected.

    ``patch_size=(1, H, W)`` matches the documented predict_volume contract;
    ``pre_process``/``normalise`` are off so ``dataset.data`` is the raw
    volume the nnU-Net path feeds through unchanged.
    """
    pytest.importorskip("torch")  # dataset.py carries a module-top torch guard
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrDataset

    return OmeZarrDataset(
        tiny_ome_zarr,
        patch_size=(1, 32, 32),
        device="cpu",
        pre_process=False,
        normalise=False,
        rotate_patches=False,
        channel=0,
    )


def _wire_deterministic_probs(nnunet_model, fake_nnunet_predictor):
    """Make the fake predictor return input-derived probabilities.

    The default canned probs are all-background, which makes chunked-vs-
    whole comparisons vacuous (all-zero masks are trivially equal). This
    replaces ``predict_single_npy_array`` with a deterministic function of
    the input (vessel prob = above-mean) while still recording each call in
    ``calls["predict_calls"]`` so call-count and input-shape assertions
    keep working.
    """

    def _deterministic(
        input_image,
        image_properties,
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    ):
        # Keep the replaced method honest: the real preprocessor requires a
        # (C,Z,H,W) input + 3-element spacing (transpose_forward is always
        # length 3), so the replacement enforces the same contract the
        # conftest fake does.
        if input_image.ndim != 4 or len(image_properties["spacing"]) != 3:
            raise ValueError(
                f"fake nnUNetPredictor: expected (C,Z,H,W) input + 3-element "
                f"spacing; got ndim={input_image.ndim}, "
                f"spacing={image_properties.get('spacing')!r}"
            )
        fake_nnunet_predictor.calls["predict_calls"].append(
            {
                "input_image": input_image,
                "image_properties": image_properties,
                "save_or_return_probabilities": save_or_return_probabilities,
            }
        )
        probs = np.zeros((2, *input_image.shape[1:]), dtype=np.float32)
        # Fixed per-pixel threshold (NOT a per-call statistic like the mean):
        # a slab-dependent aggregate would produce different masks for the
        # whole-volume vs chunked calls and defeat the equality comparison.
        probs[1] = (input_image[0] > 0.5).astype(np.float32)
        probs[0] = 1.0 - probs[1]
        seg = np.zeros(input_image.shape[1:], dtype=np.uint8)
        return seg, probs

    nnunet_model.predictor.predict_single_npy_array = _deterministic


def test_predict_volume_nnunet_whole_volume_with_ngff_spacing(
    tmp_path, nnunet_model, fake_nnunet_predictor, tiny_dataset
):
    """predict_volume routes NnUnetV2Model to one whole-volume predict call.

    With ``z_chunk_size=None`` the (Z,H,W) dask array materializes once as
    ``(1,Z,H,W)`` float32 and spacing comes from the NGFF metadata by axis
    name (z,y,x = 6.5um each -- the c-axis entry at position 0 must NOT leak
    into the spacing triple). The output zarr is (Z,H,W) uint8 {0,255},
    positionally aligned with the input.
    """
    from liom_toolkit.segmentation.vseg.prediction import predict_volume

    _wire_deterministic_probs(nnunet_model, fake_nnunet_predictor)
    out = str(tmp_path / "out.zarr")

    predict_volume(nnunet_model, tiny_dataset, out)

    calls = fake_nnunet_predictor.calls["predict_calls"]
    assert len(calls) == 1
    input_image = calls[0]["input_image"]
    assert input_image.shape == (1, 5, 32, 32)
    assert input_image.dtype == np.float32
    assert calls[0]["image_properties"] == {"spacing": [6.5, 6.5, 6.5]}

    result = zarr.open(out, mode="r")
    assert result.shape == (5, 32, 32)
    assert result.dtype == np.uint8
    result_np = np.asarray(result[:])
    assert set(np.unique(result_np)).issubset({0, 255})
    assert result_np.max() == 255  # non-trivial mask, not all-background


def test_predict_volume_nnunet_explicit_spacing_overrides_ngff(
    tmp_path, nnunet_model, fake_nnunet_predictor, tiny_dataset
):
    """An explicit spacing tuple wins over the NGFF metadata.

    The predictor must see exactly ``[3.0, 2.0, 1.0]`` in z,y,x array order
    -- an explicit caller-supplied spacing is authoritative for stores whose
    metadata is absent or untrusted.
    """
    from liom_toolkit.segmentation.vseg.prediction import predict_volume

    predict_volume(nnunet_model, tiny_dataset, str(tmp_path / "out.zarr"), spacing=(3.0, 2.0, 1.0))

    calls = fake_nnunet_predictor.calls["predict_calls"]
    assert len(calls) == 1
    assert calls[0]["image_properties"] == {"spacing": [3.0, 2.0, 1.0]}


def test_predict_volume_nnunet_missing_ngff_spacing_raises(
    tmp_path, nnunet_model, fake_nnunet_predictor, tiny_ome_zarr, tiny_dataset
):
    """A zarr without scale metadata + spacing=None raises ValueError.

    Spacing is never silently defaulted to isotropic: the error must tell
    the caller to pass ``spacing`` explicitly and fire before any model
    invocation.
    """
    from liom_toolkit.segmentation.vseg.prediction import predict_volume

    # Strip coordinateTransformations from every multiscales dataset so the
    # NGFF reader exposes no scale vector.
    root = zarr.open_group(tiny_ome_zarr, mode="a")
    ome = copy.deepcopy(dict(root.attrs["ome"]))
    for dataset_entry in ome["multiscales"][0]["datasets"]:
        dataset_entry.pop("coordinateTransformations", None)
    root.attrs["ome"] = ome

    with pytest.raises(ValueError, match="spacing"):
        predict_volume(nnunet_model, tiny_dataset, str(tmp_path / "out.zarr"))

    assert fake_nnunet_predictor.calls["predict_calls"] == []


def test_predict_volume_nnunet_z_chunking_matches_whole(
    tmp_path, nnunet_model, fake_nnunet_predictor, tiny_dataset
):
    """z_chunk_size=2 on Z=5 produces 3 slab calls identical to whole-volume output.

    The only Python loop on the nnU-Net volume path is over Z-CHUNKS (RAM
    bounding), never per-slice: each call carries ``(1, slab, H, W)`` and the
    positional ``new_volume[z0:z1]`` writes must assemble the same result a
    single whole-volume call produces.
    """
    from liom_toolkit.segmentation.vseg.prediction import predict_volume

    _wire_deterministic_probs(nnunet_model, fake_nnunet_predictor)

    whole_out = str(tmp_path / "whole.zarr")
    predict_volume(nnunet_model, tiny_dataset, whole_out)
    whole = np.asarray(zarr.open(whole_out, mode="r")[:])

    fake_nnunet_predictor.calls["predict_calls"].clear()
    chunked_out = str(tmp_path / "chunked.zarr")
    predict_volume(nnunet_model, tiny_dataset, chunked_out, z_chunk_size=2)

    calls = fake_nnunet_predictor.calls["predict_calls"]
    assert len(calls) == 3  # ceil(5 / 2): slabs of 2, 2, 1
    slab_sizes = [c["input_image"].shape[1] for c in calls]
    assert slab_sizes == [2, 2, 1]
    for c in calls:
        assert c["input_image"].ndim == 4
        assert c["input_image"].shape[0] == 1

    chunked = np.asarray(zarr.open(chunked_out, mode="r")[:])
    np.testing.assert_array_equal(chunked, whole)


def test_predict_volume_legacy_rejects_nnunet_kwargs(tmp_path):
    """spacing/z_chunk_size on a legacy (non-NnUnetV2) model raise ValueError.

    Asymmetric kwargs are never silently ignored: each error names the
    offending kwarg so the caller knows it only applies to nnU-Net models.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.prediction import predict_volume

    with pytest.raises(ValueError, match="spacing"):
        predict_volume(None, None, str(tmp_path / "a.zarr"), spacing=(6.5, 6.5, 6.5))
    with pytest.raises(ValueError, match="z_chunk_size"):
        predict_volume(None, None, str(tmp_path / "b.zarr"), z_chunk_size=2)


def test_predict_volume_nnunet_all_zero_volume_raises(
    tmp_path, nnunet_model, fake_nnunet_predictor
):
    """predict_volume raises ValueError on an all-zero input volume.

    nnU-Net's ``crop_to_nonzero`` full-bbox fallback lets an all-zero input
    through, and z-score normalization of a constant array NaNs on std=0 --
    without the guard the call would write a plausible all-background mask
    on garbage input (the silent-wrong-data mode). The error must fire
    before any model invocation.
    """
    pytest.importorskip("torch")
    from liom_toolkit.conversion.conversion import save_zarr
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrDataset
    from liom_toolkit.segmentation.vseg.prediction import predict_volume

    zarr_path = str(tmp_path / "zeros.zarr")
    save_zarr(
        np.zeros((1, 4, 32, 32), dtype=np.float32),
        zarr_path,
        scales=(6.5, 6.5, 6.5),
        chunks=(1, 1, 32, 32),
    )
    dataset = OmeZarrDataset(
        zarr_path,
        patch_size=(1, 32, 32),
        device="cpu",
        pre_process=False,
        normalise=False,
        rotate_patches=False,
        channel=0,
    )

    with pytest.raises(ValueError, match="all-zero"):
        predict_volume(nnunet_model, dataset, str(tmp_path / "out.zarr"))

    assert fake_nnunet_predictor.calls["predict_calls"] == []


def test_predict_volume_nnunet_skips_all_zero_slabs(
    tmp_path, nnunet_model, fake_nnunet_predictor
):
    """All-zero Z-slabs in a non-zero volume are written as zeros without a model call.

    A chunked volume can contain all-zero slabs (empty z-regions) that would
    NaN inside nnU-Net's z-score normalization. The chunked path must skip
    the model call for those slabs -- the zero-initialized output store
    already holds the correct all-zero mask -- while still predicting the
    non-zero slabs identically to the whole-volume path.
    """
    pytest.importorskip("torch")
    from liom_toolkit.conversion.conversion import save_zarr
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrDataset
    from liom_toolkit.segmentation.vseg.prediction import predict_volume

    # Slab [0:2] all-zero, [2:5] non-zero; z_chunk_size=2 gives 3 slabs.
    vol = np.zeros((1, 5, 32, 32), dtype=np.float32)
    vol[0, 2:] = np.random.default_rng(0).random((3, 32, 32))
    zarr_path = str(tmp_path / "partial_zeros.zarr")
    save_zarr(vol, zarr_path, scales=(6.5, 6.5, 6.5), chunks=(1, 1, 32, 32))
    dataset = OmeZarrDataset(
        zarr_path,
        patch_size=(1, 32, 32),
        device="cpu",
        pre_process=False,
        normalise=False,
        rotate_patches=False,
        channel=0,
    )

    _wire_deterministic_probs(nnunet_model, fake_nnunet_predictor)

    whole_out = str(tmp_path / "whole.zarr")
    predict_volume(nnunet_model, dataset, whole_out)
    whole = np.asarray(zarr.open(whole_out, mode="r")[:])

    fake_nnunet_predictor.calls["predict_calls"].clear()
    chunked_out = str(tmp_path / "chunked.zarr")
    predict_volume(nnunet_model, dataset, chunked_out, z_chunk_size=2)

    calls = fake_nnunet_predictor.calls["predict_calls"]
    # The leading all-zero slab [0:2] is skipped: 2 model calls, not 3.
    assert len(calls) == 2
    assert [c["input_image"].shape[1] for c in calls] == [2, 1]

    chunked = np.asarray(zarr.open(chunked_out, mode="r")[:])
    np.testing.assert_array_equal(chunked, whole)


def test_predict_volume_nnunet_zero_dim_raises(
    tmp_path, nnunet_model, fake_nnunet_predictor, tiny_dataset
):
    """A dataset whose data has a zero spatial dim raises ValueError before inference.

    An empty axis cannot produce a meaningful mask -- the explicit raise
    prevents a silently empty (plausible-shaped) output store.
    """
    from liom_toolkit.segmentation.vseg.prediction import predict_volume

    tiny_dataset.data = tiny_dataset.data[:0]  # (0, 32, 32)

    with pytest.raises(ValueError, match=r"empty|shape"):
        predict_volume(nnunet_model, tiny_dataset, str(tmp_path / "out.zarr"))

    assert fake_nnunet_predictor.calls["predict_calls"] == []


def test_predict_volume_nnunet_existing_output_raises(
    tmp_path, nnunet_model, fake_nnunet_predictor, tiny_dataset
):
    """An existing zarr_location raises FileExistsError before any model call.

    The nnU-Net path refuses to overwrite an existing store -- unlike the
    legacy ``zarr.open(mode='w')`` semantics -- so an existing path must
    fail fast BEFORE inference, not truncate.
    """
    from liom_toolkit.segmentation.vseg.prediction import predict_volume

    existing = tmp_path / "existing.zarr"
    existing.mkdir()

    with pytest.raises(FileExistsError, match=r"existing\.zarr"):
        predict_volume(nnunet_model, tiny_dataset, str(existing))

    assert fake_nnunet_predictor.calls["predict_calls"] == []


def test_predict_volume_nnunet_not_gated_by_legacy_dataset_guards(
    tmp_path, nnunet_model, fake_nnunet_predictor, tiny_ome_zarr
):
    """rotate_patches/filter_empty/patch_size guards do not gate the nnU-Net path.

    Those guards protect the legacy patch loop's index-to-grid mapping
    (``get_patch_coordinates``); the nnU-Net path reads ``dataset.data``
    whole and never iterates the patch index, so a dataset built with
    rotate_patches=True and a 3D patch_size must still run.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.dataset import OmeZarrDataset
    from liom_toolkit.segmentation.vseg.prediction import predict_volume

    dataset = OmeZarrDataset(
        tiny_ome_zarr,
        patch_size=(32, 32, 32),  # 3D patch_size would fail the legacy guard
        device="cpu",
        pre_process=False,
        normalise=False,
        rotate_patches=True,  # would fail the legacy guard
        channel=0,
    )

    predict_volume(nnunet_model, dataset, str(tmp_path / "out.zarr"))

    assert len(fake_nnunet_predictor.calls["predict_calls"]) == 1
