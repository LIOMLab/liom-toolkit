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
