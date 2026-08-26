"""Import-smoke + weights_only round-trip tests for
``liom_toolkit/segmentation/vseg/model.py``.

These tests close the Wave-0 test gap identified for DEP-01: before Phase 3,
``test_model.py`` did not exist, so the ``torch.load`` ``weights_only=True``
audit landed in the dependency-adaptation work had no automated regression
coverage.

``model.py`` imports ``torch`` and ``wandb`` at module top. Per the
established optional-dep gating pattern (AGENTS section 5), each test calls
``pytest.importorskip`` BEFORE importing from ``vseg.model`` so the tests run
for real on the 3.12-full CI leg (where the ``ai`` extra is installed) and
cleanly skip on the 3.14-core leg (where it is not).

Coverage:

* ``test_vsegmodel_imports`` -- ``VsegModel`` is callable and is a subclass
  of ``torch.nn.Module``.
* ``test_vsegmodel_state_dict_round_trip_weights_only_true`` -- a plain
  ``state_dict`` saved and reloaded with ``weights_only=True`` loads back
  via ``load_state_dict`` without raising, proving the
  ``weights_only=True``-first branch works for the same object shape the
  production checkpoint load path expects.
"""

import pytest


def test_vsegmodel_imports():
    """VsegModel is callable and subclasses torch.nn.Module."""
    pytest.importorskip("torch")
    pytest.importorskip("wandb")

    import torch
    from liom_toolkit.segmentation.vseg.model import VsegModel

    assert callable(VsegModel)
    assert issubclass(VsegModel, torch.nn.Module)


def test_vsegmodel_state_dict_round_trip_weights_only_true(tmp_path):
    """A plain state_dict round-trips through torch.save/torch.load with
    weights_only=True and reloads via load_state_dict without raising --
    proves the weights_only=True-first branch works for the object shape the
    production checkpoint load path expects."""
    pytest.importorskip("torch")

    import torch
    from liom_toolkit.segmentation.vseg.model import VsegModel

    model = VsegModel(pretrained=False)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(model.state_dict(), str(checkpoint))

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    # load_state_dict succeeds without raising -- the state_dict shape
    # matches the model exactly.
    model.load_state_dict(state)


# --- predict_one norm/patching regression (BUG-01) ---------------------------
#
# predict_one is called by validate_model (vseg/validation.py) as
# predict_one(..., norm=True, patching=False), but predict_one historically
# did not accept ``norm`` or ``patching`` parameters, so the spec caller
# raised TypeError. The fix adds the two named parameters and defines their
# semantics: ``norm`` gates CLAHE (default True preserves the shipped
# always-CLAHE behavior; False skips CLAHE and uses the min-max uint8 image);
# ``patching=False`` runs the existing single full-image pass, while
# ``patching=True`` raises NotImplementedError pointing to predict_volume
# (no silent single-pass fallback when tiled inference was requested).
#
# All three tests are torch-gated per AGENTS section 5: each calls
# pytest.importorskip("torch") as the first line so the tests run on the
# ai-extra leg and skip cleanly on the core leg. torch/numpy are NOT mocked
# (a real torch.nn.Module stub model is used); cv2.createCLAHE is spied on
# via monkeypatch for the norm=False test, which is permitted (cv2 is not
# torch/numpy).


def _write_tiny_png(tmp_path, name: str = "tiny.png", size: int = 16):
    """Write a tiny seeded-random uint8 PNG and return its path."""
    import imageio.v3 as iio
    import numpy as np

    rng = np.random.default_rng(0)
    arr = rng.integers(1, 256, size=(size, size), dtype=np.uint8)
    path = tmp_path / name
    iio.imwrite(str(path), arr)
    return path


def _make_stub_model():
    """Build a minimal torch.nn.Module returning segmentation-shaped logits.

    Must be called AFTER pytest.importorskip("torch") so torch is available.
    Returns a tensor shaped (1, 1, H, W) matching the input spatial shape so
    that do_predict's squeeze/index arithmetic produces a valid 2D array.
    Output is filled with zeros (below the 0.5 threshold -> background), which
    is a valid segmentation result for a synthetic image.
    """
    import torch

    class _StubSegModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            n, _, h, w = x.shape
            return torch.zeros((n, 1, h, w), dtype=torch.float32, device=x.device)

    return _StubSegModel()


def test_predict_one_norm_true_patching_false(tmp_path):
    """validate_model's call shape (norm=True, patching=False) works end-to-end
    on a tiny synthetic PNG with a stub torch.nn.Module model. This is the
    BUG-01 D-05 spec-caller proof: the call returns a uint8 ndarray instead of
    raising TypeError on the missing norm/patching parameters."""
    pytest.importorskip("torch")

    from liom_toolkit.segmentation.vseg.prediction import predict_one

    img_path = _write_tiny_png(tmp_path)
    model = _make_stub_model()
    out_dir = tmp_path / "out_norm_true"

    result = predict_one(
        model=model,
        img_path=str(img_path),
        save_path=str(out_dir),
        norm=True,
        dev="cpu",
        patching=False,
    )

    import numpy as np

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8


def test_predict_one_patching_true_raises(tmp_path):
    """predict_one(..., patching=True) raises NotImplementedError whose message
    mentions predict_volume (the tiled-prediction alternative). This is the
    BUG-01 D-06 explicit-failure proof: patching=True must NOT silently fall
    back to single-pass (AGENTS section 2 -- no silent wrong-data fallback)."""
    pytest.importorskip("torch")

    from liom_toolkit.segmentation.vseg.prediction import predict_one

    img_path = _write_tiny_png(tmp_path)
    model = _make_stub_model()
    out_dir = tmp_path / "out_patch"

    with pytest.raises(NotImplementedError) as excinfo:
        predict_one(
            model=model,
            img_path=str(img_path),
            save_path=str(out_dir),
            dev="cpu",
            patching=True,
        )

    assert "predict_volume" in str(excinfo.value)


def test_predict_one_norm_false_skips_clahe(tmp_path, monkeypatch):
    """predict_one(..., norm=False, patching=False) produces output without
    applying CLAHE. cv2.createCLAHE is spied on via monkeypatch (a cv2
    function, not torch/numpy -- permitted by AGENTS section 5) and asserted
    not to be called. The output must still be a valid uint8 ndarray (the
    min-max-only path). This is the BUG-01 D-06 norm-gate proof: norm=False
    must skip CLAHE, not silently apply it."""
    pytest.importorskip("torch")

    import liom_toolkit.segmentation.vseg.prediction as prediction_mod
    from liom_toolkit.segmentation.vseg.prediction import predict_one

    img_path = _write_tiny_png(tmp_path)
    model = _make_stub_model()
    out_dir = tmp_path / "out_norm_false"

    calls: list = []

    def _fake_create_clahe(*args, **kwargs):
        calls.append((args, kwargs))

        class _AHE:
            def apply(self, image):
                return image

        return _AHE()

    monkeypatch.setattr(prediction_mod.cv2, "createCLAHE", _fake_create_clahe)

    result = predict_one(
        model=model,
        img_path=str(img_path),
        save_path=str(out_dir),
        norm=False,
        dev="cpu",
        patching=False,
    )

    import numpy as np

    assert calls == [], "CLAHE must not be applied when norm=False"
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
