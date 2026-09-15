"""Pytest configuration for the vseg test subpackage.

Shared fixtures for the nnU-Net v2 predictor-wrapper tests:

* ``fake_nnunet_predictor`` injects a fake
  ``nnunetv2.inference.predict_from_raw_data`` leaf module into
  ``sys.modules`` (restored in ``finally``) so the wrapper can be exercised
  without the real nnU-Net package. The fake ``nnUNetPredictor`` records its
  constructor kwargs, its ``initialize_from_trained_model_folder`` arguments,
  and every ``predict_single_npy_array`` call, and returns a canned
  ``(segmentation, probabilities)`` pair the test can stage. The two parent
  package entries (``nnunetv2``, ``nnunetv2.inference``) are stubbed too, so
  the fixture also works on a torch-only host where nnunetv2 is not installed.
* ``stub_nnunet_model_dir`` writes a minimal nnU-Net trained-model directory
  (``dataset.json`` + ``plans.json`` + ``fold_0/checkpoint_final.pth``) into
  ``tmp_path`` -- exactly the layout the wrapper validates up front, without
  real nnU-Net artifacts.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def fake_nnunet_predictor() -> Any:
    """Inject a fake ``nnunetv2.inference.predict_from_raw_data`` module into sys.modules.

    The fake leaf module exposes ``nnUNetPredictor`` -- a recording stand-in
    for the real class. It captures constructor kwargs (device,
    ``perform_everything_on_device``, ...) in ``calls["ctor_kwargs"]``,
    ``initialize_from_trained_model_folder`` arguments in
    ``calls["init_calls"]``, and ``predict_single_npy_array`` calls in
    ``calls["predict_calls"]``. The predict method returns
    ``(state["seg"], state["probs"])``; when the test has not staged
    probabilities it returns a 2-class softmax whose vessel channel sits at
    0.1 (an all-background prediction) over the input's spatial shape.

    Yields a ``SimpleNamespace`` with ``predictor_cls``, ``calls``, and
    ``state`` so tests assert the wrapper's contract directly. All injected
    ``sys.modules`` entries are restored in ``finally`` -- the fake never
    leaks into other tests, and on a host WITH real nnunetv2 installed the
    real modules are put back.
    """
    leaf_name = "nnunetv2.inference.predict_from_raw_data"
    injected_names = ("nnunetv2", "nnunetv2.inference", leaf_name)
    saved = {name: sys.modules.get(name) for name in injected_names}

    state: dict[str, Any] = {"seg": None, "probs": None}
    calls: dict[str, list] = {"ctor_kwargs": [], "init_calls": [], "predict_calls": []}

    class FakePredictor:
        """Recording stand-in for ``nnunetv2...nnUNetPredictor``.

        Mirrors the real constructor contract: predictor kwargs arrive via
        ``**kwargs`` and are stored verbatim on the instance, so tests can
        read ``predictor.device`` / ``predictor.perform_everything_on_device``
        the way they would on the real object.
        """

        def __init__(self, **kwargs: Any) -> None:
            calls["ctor_kwargs"].append(kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

        def initialize_from_trained_model_folder(
            self,
            model_training_output_dir: str,
            use_folds: Any = None,
            checkpoint_name: str = "checkpoint_final.pth",
        ) -> None:
            calls["init_calls"].append(
                {
                    "model_dir": model_training_output_dir,
                    "use_folds": use_folds,
                    "checkpoint_name": checkpoint_name,
                }
            )

        def predict_single_npy_array(
            self,
            input_image: np.ndarray,
            image_properties: dict,
            segmentation_previous_stage: Any = None,
            output_file_truncated: Any = None,
            save_or_return_probabilities: bool = False,
        ) -> tuple[np.ndarray, np.ndarray]:
            # Enforce the real nnunetv2 preprocessor contract: every
            # configuration's transpose_forward is length 3, so
            # run_case_npy requires a (C,Z,H,W) input and a 3-element
            # spacing -- a (1,H,W) input + 2-element spacing crashes the
            # real DefaultPreprocessor with 'axes don't match array'. The
            # fake must reject the same shapes, or it green-locks whatever
            # wrong contract a caller sends.
            if input_image.ndim != 4:
                raise ValueError(
                    f"fake nnUNetPredictor: input_image.ndim must be 4 "
                    f"(C,Z,H,W); got ndim={input_image.ndim} "
                    f"shape={input_image.shape}"
                )
            spacing = image_properties["spacing"]
            if len(spacing) != 3:
                raise ValueError(
                    f"fake nnUNetPredictor: image_properties['spacing'] must "
                    f"have 3 elements; got {spacing!r}"
                )
            calls["predict_calls"].append(
                {
                    "input_image": input_image,
                    "image_properties": image_properties,
                    "save_or_return_probabilities": save_or_return_probabilities,
                }
            )
            probs = state["probs"]
            if probs is None:
                # Default canned output: a 2-class softmax where the vessel
                # channel sits below threshold (an all-background call).
                probs = np.zeros((2, *input_image.shape[1:]), dtype=np.float32)
                probs[0] = 0.9
                probs[1] = 0.1
            seg = state["seg"]
            if seg is None:
                seg = np.zeros(input_image.shape[1:], dtype=np.uint8)
            return seg, probs

    nnunetv2_pkg = types.ModuleType("nnunetv2")
    nnunetv2_pkg.__path__ = []  # mark as a package without touching the real one
    inference_pkg = types.ModuleType("nnunetv2.inference")
    inference_pkg.__path__ = []
    leaf = types.ModuleType(leaf_name)
    leaf.nnUNetPredictor = FakePredictor

    sys.modules["nnunetv2"] = nnunetv2_pkg
    sys.modules["nnunetv2.inference"] = inference_pkg
    sys.modules[leaf_name] = leaf
    try:
        yield SimpleNamespace(predictor_cls=FakePredictor, calls=calls, state=state)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.fixture
def stub_nnunet_model_dir(tmp_path: Path) -> Path:
    """Write a minimal nnU-Net trained-model dir that passes the wrapper's dir contract.

    Layout mirrors a real ``nnUNetv2_train`` output:
    ``<tmp>/Dataset999_Tiny/nnUNetTrainer__nnUNetPlans__2d/`` containing
    ``dataset.json``, ``plans.json``, and ``fold_0/checkpoint_final.pth`` --
    the three components ``NnUnetV2Model.__init__`` validates before calling
    nnU-Net. The JSON payloads are intentionally minimal: the fake-predictor
    seam never reads them; this fixture only proves the directory contract.
    """
    model_dir = tmp_path / "Dataset999_Tiny" / "nnUNetTrainer__nnUNetPlans__2d"
    fold_dir = model_dir / "fold_0"
    fold_dir.mkdir(parents=True)
    (model_dir / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"0": "LSFM_6p5um"},
                "labels": {"background": 0, "vessel": 1},
                "numTraining": 4,
                "file_ending": ".png",
                "dataset_name": "Dataset999_Tiny",
            }
        )
    )
    (model_dir / "plans.json").write_text(
        json.dumps({"plans_name": "nnUNetPlans", "configurations": {}})
    )
    (fold_dir / "checkpoint_final.pth").write_bytes(b"stub checkpoint bytes")
    return model_dir
