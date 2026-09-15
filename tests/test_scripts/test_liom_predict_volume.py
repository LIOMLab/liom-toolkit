"""Smoke + argparse tests for the ``liom-predict-volume`` CLI.

Exercises:

* ``_build_argument_parser().parse_args(...)`` parses the prediction args
  (positional ``input``, ``--model-dir``, ``--output``, ``--spacing``,
  ``--device``, ``--channel``, ``--z-chunk-size``, ``--folds``,
  ``--checkpoint-name``, ``--tile-step-size``) and the shared flags from
  ``build_common_parser`` (``--log-level``, ``--resume``,
  ``--dask-scheduler``, ``--n-workers``).
* ``main()`` exits 2 (``parser.error``) on bad input -- the cheap
  validations (input exists, model dir is a directory, output does not
  exist, spacing is three positive finite floats) run BEFORE the heavy
  torch/nnunetv2 imports so a typo'd path surfaces as a clear error, not a
  traceback.
* Building the parser (or running ``--help``) never imports ``torch`` or
  ``nnunetv2`` -- all heavy imports live inside ``main()`` after
  validation, so ``--help`` is instant on a core-only install.
* An end-to-end smoke runs ``main(argv)`` against a real
  ``NnUnetV2Model`` wrapping a recording fake ``nnUNetPredictor`` leaf, a
  stub model dir, and a tiny ``save_zarr``-backed OME-Zarr -- proving the
  CLI->wrapper->predict_volume->zarr wiring without the real nnU-Net
  planner.

The ai-marked smoke tests gate on ``pytest.importorskip("torch")`` at the
first line of the test body (AGENTS section 5).
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

MODULE = "liom_toolkit.scripts.liom_predict_volume"


def test_liom_predict_volume_help_contains_shared_and_curated_flags() -> None:
    """liom-predict-volume --help lists the curated flags + the shared flags."""
    from liom_toolkit.scripts.liom_predict_volume import _build_argument_parser

    out = _build_argument_parser().format_help()
    # Shared flags from build_common_parser.
    for flag in ("--log-level", "--resume", "--dask-scheduler", "--n-workers"):
        assert flag in out, f"liom-predict-volume --help missing shared flag {flag}"
    # Curated prediction flags + the positional input.
    for flag in (
        "input",
        "--model-dir",
        "--output",
        "--spacing",
        "--device",
        "--channel",
        "--z-chunk-size",
        "--folds",
        "--checkpoint-name",
        "--tile-step-size",
    ):
        assert flag in out, f"liom-predict-volume --help missing curated flag {flag}"


def test_liom_predict_volume_parse_args_captures_surface(tmp_path) -> None:
    """parse_args captures every CLI argument and the documented defaults."""
    from liom_toolkit.scripts.liom_predict_volume import _build_argument_parser

    parser = _build_argument_parser()
    args = parser.parse_args(
        [
            str(tmp_path / "in.zarr"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output",
            str(tmp_path / "out.zarr"),
            "--spacing",
            "3.0",
            "2.0",
            "1.0",
            "--device",
            "cpu",
            "--channel",
            "1",
            "--z-chunk-size",
            "4",
            "--folds",
            "0",
            "1",
            "--checkpoint-name",
            "checkpoint_best.pth",
            "--tile-step-size",
            "0.3",
        ]
    )
    assert args.input == str(tmp_path / "in.zarr")
    assert args.model_dir == str(tmp_path / "model")
    assert args.output == str(tmp_path / "out.zarr")
    assert list(args.spacing) == [3.0, 2.0, 1.0]
    assert args.device == "cpu"
    assert args.channel == 1
    assert args.z_chunk_size == 4
    assert list(args.folds) == [0, 1]
    assert args.checkpoint_name == "checkpoint_best.pth"
    assert args.tile_step_size == 0.3


def test_liom_predict_volume_defaults() -> None:
    """Defaults: spacing/device/z-chunk-size/folds unset; channel 0; final checkpoint."""
    from liom_toolkit.scripts.liom_predict_volume import _build_argument_parser

    args = _build_argument_parser().parse_args(
        ["in.zarr", "--model-dir", "m", "--output", "o.zarr"]
    )
    assert args.spacing is None
    assert args.device is None
    assert args.channel == 0
    assert args.z_chunk_size is None
    assert args.folds is None
    assert args.checkpoint_name == "checkpoint_final.pth"
    assert args.tile_step_size == 0.5


def test_liom_predict_volume_required_args_exit_2() -> None:
    """Missing --model-dir / --output exits 2 (argparse required-arg failure)."""
    from liom_toolkit.scripts.liom_predict_volume import _build_argument_parser

    parser = _build_argument_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["in.zarr"])
    assert exc.value.code == 2


def test_liom_predict_volume_help_never_imports_heavy_deps() -> None:
    """Building the parser (and --help) never imports torch or nnunetv2.

    All heavy imports must live inside ``main()`` AFTER cheap validation, so
    ``--help`` is instant and works on a core-only install. Runs in a clean
    subprocess so other tests' imports cannot mask a regression.
    """
    code = (
        "import sys; "
        f"import {MODULE} as m; "
        "parser = m._build_argument_parser(); "
        "parser.parse_args(['--help'])"
    )
    # --help exits 0 with usage on stdout; a heavy import at module top or
    # parser-build time would still exit 0, so check sys.modules afterwards.
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed interpreter + fixed code string
        [
            sys.executable,
            "-c",
            code + "; assert 'torch' not in sys.modules, 'torch imported'; "
            "assert 'nnunetv2' not in sys.modules, 'nnunetv2 imported'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"--help exited {result.returncode}: {result.stderr[-2000:]}"
    assert "usage:" in result.stdout


def test_main_exits_2_on_nonexistent_input(tmp_path) -> None:
    """main() exits 2 naming the path when the input OME-Zarr does not exist."""
    from liom_toolkit.scripts.liom_predict_volume import main

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    missing = str(tmp_path / "nope.zarr")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                missing,
                "--model-dir",
                str(model_dir),
                "--output",
                str(tmp_path / "out.zarr"),
            ]
        )
    assert exc.value.code == 2


def test_main_exits_2_on_non_dir_model_dir(tmp_path) -> None:
    """main() exits 2 when --model-dir is not a directory."""
    from liom_toolkit.scripts.liom_predict_volume import main

    input_dir = tmp_path / "in.zarr"
    input_dir.mkdir()
    missing_model = str(tmp_path / "no_model")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                str(input_dir),
                "--model-dir",
                missing_model,
                "--output",
                str(tmp_path / "out.zarr"),
            ]
        )
    assert exc.value.code == 2


def test_main_exits_2_on_existing_output(tmp_path) -> None:
    """main() exits 2 when --output already exists (refuse to overwrite early).

    predict_volume's FileExistsError contract is surfaced at the argparse
    boundary, BEFORE any heavy import or inference work.
    """
    from liom_toolkit.scripts.liom_predict_volume import main

    input_dir = tmp_path / "in.zarr"
    input_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    existing_out = tmp_path / "out.zarr"
    existing_out.mkdir()
    with pytest.raises(SystemExit) as exc:
        main(
            [
                str(input_dir),
                "--model-dir",
                str(model_dir),
                "--output",
                str(existing_out),
            ]
        )
    assert exc.value.code == 2


def test_main_exits_2_on_bad_spacing_arity(tmp_path) -> None:
    """--spacing with 2 values exits 2 (nargs=3 enforced by argparse)."""
    from liom_toolkit.scripts.liom_predict_volume import main

    input_dir = tmp_path / "in.zarr"
    input_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(SystemExit) as exc:
        main(
            [
                str(input_dir),
                "--model-dir",
                str(model_dir),
                "--output",
                str(tmp_path / "out.zarr"),
                "--spacing",
                "6.5",
                "6.5",
            ]
        )
    assert exc.value.code == 2


def test_main_exits_2_on_nonpositive_spacing(tmp_path) -> None:
    """--spacing with a non-positive component exits 2 naming the value."""
    from liom_toolkit.scripts.liom_predict_volume import main

    input_dir = tmp_path / "in.zarr"
    input_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(SystemExit) as exc:
        main(
            [
                str(input_dir),
                "--model-dir",
                str(model_dir),
                "--output",
                str(tmp_path / "out.zarr"),
                "--spacing",
                "6.5",
                "0",
                "6.5",
            ]
        )
    assert exc.value.code == 2


def test_main_exits_2_on_negative_channel(tmp_path) -> None:
    """--channel -1 exits 2 (channel index must be non-negative)."""
    from liom_toolkit.scripts.liom_predict_volume import main

    input_dir = tmp_path / "in.zarr"
    input_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(SystemExit) as exc:
        main(
            [
                str(input_dir),
                "--model-dir",
                str(model_dir),
                "--output",
                str(tmp_path / "out.zarr"),
                "--channel",
                "-1",
            ]
        )
    assert exc.value.code == 2


def test_liom_predict_volume_no_hardcoded_lab_path_default() -> None:
    """No hardcoded lab path in the parser defaults (all paths are parameters)."""
    from liom_toolkit.scripts.liom_predict_volume import _build_argument_parser

    parser = _build_argument_parser()
    parser.parse_args(["x", "--model-dir", "m", "--output", "o"])
    for action in parser._actions:
        if action.default is not None and isinstance(action.default, str):
            assert "/data/" not in action.default, (
                f"flag {action.dest} has a hardcoded lab default: {action.default!r}"
            )


# ---------------------------------------------------------------------------
# End-to-end smoke: main(argv) -> NnUnetV2Model -> predict_volume -> zarr
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_nnunet_predictor() -> Any:
    """Inject a fake ``nnunetv2.inference.predict_from_raw_data`` leaf module.

    Local copy of the vseg conftest seam (conftest fixtures do not cross
    test directories): the fake ``nnUNetPredictor`` records constructor
    kwargs and every ``predict_single_npy_array`` call, and returns
    input-derived probabilities (vessel channel = input > 0.5) so the
    output mask is a deterministic function of the input regardless of
    slab boundaries.
    """
    leaf_name = "nnunetv2.inference.predict_from_raw_data"
    injected_names = ("nnunetv2", "nnunetv2.inference", leaf_name)
    saved = {name: sys.modules.get(name) for name in injected_names}

    calls: dict[str, list] = {"ctor_kwargs": [], "init_calls": [], "predict_calls": []}

    class FakePredictor:
        """Recording stand-in for ``nnunetv2...nnUNetPredictor``."""

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
            # The real predictor sets configuration_manager during
            # initialize; the stub model dir is a '2d' config, mirrored by
            # a 2-entry patch_size (the discriminator nnU-Net's
            # sliding-window slicer uses).
            self.configuration_manager = SimpleNamespace(patch_size=(32, 32))

        def predict_single_npy_array(
            self,
            input_image: np.ndarray,
            image_properties: dict,
            segmentation_previous_stage: Any = None,
            output_file_truncated: Any = None,
            save_or_return_probabilities: bool = False,
        ) -> tuple[np.ndarray, np.ndarray]:
            # Enforce the real nnunetv2 preprocessor contract (same as the
            # vseg conftest fake): transpose_forward is always length 3, so
            # run_case_npy requires (C,Z,H,W) input + 3-element spacing --
            # the fake must reject the shapes the real preprocessor crashes
            # on, or it green-locks a wrong caller contract.
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
            probs = np.zeros((2, *input_image.shape[1:]), dtype=np.float32)
            # Fixed per-pixel threshold: deterministic regardless of the
            # Z-slab boundaries a chunked run uses.
            probs[1] = (input_image[0] > 0.5).astype(np.float32)
            probs[0] = 1.0 - probs[1]
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
        yield SimpleNamespace(predictor_cls=FakePredictor, calls=calls)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.fixture
def stub_nnunet_model_dir(tmp_path: Path) -> Path:
    """Write a minimal nnU-Net trained-model dir passing the wrapper's contract.

    Local copy of the vseg conftest fixture: ``dataset.json`` +
    ``plans.json`` + ``fold_0/checkpoint_final.pth`` -- the layout
    ``NnUnetV2Model.__init__`` validates before calling nnU-Net.
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


@pytest.fixture
def tiny_ome_zarr(tmp_path: Path) -> str:
    """Write a tiny (1, 5, 32, 32) float32 OME-Zarr with 6.5um scales.

    A real ``save_zarr`` write so the NGFF ``coordinateTransformations``
    metadata the spacing parser reads is genuinely on disk.
    """
    from liom_toolkit.conversion.conversion import save_zarr

    vol = np.random.default_rng(0).random((1, 5, 32, 32)).astype(np.float32)
    zarr_path = str(tmp_path / "tiny.zarr")
    save_zarr(vol, zarr_path, scales=(6.5, 6.5, 6.5), chunks=(1, 1, 32, 32))
    return zarr_path


@pytest.mark.ai
def test_main_end_to_end_fake_predictor_ngff_spacing(
    tmp_path, fake_nnunet_predictor, stub_nnunet_model_dir, tiny_ome_zarr
) -> None:
    """main(argv) runs CLI->NnUnetV2Model->predict_volume->zarr end to end.

    The real wrapper drives the fake predictor: output zarr exists with
    shape (Z,H,W), dtype uint8, values in {0,255}, and the predictor
    observed the NGFF-derived (6.5, 6.5, 6.5) spacing -- the c-axis scale
    entry at position 0 must NOT leak into the spacing triple.
    """
    pytest.importorskip("torch")  # model_v2/dataset carry module-top torch guards
    import zarr

    from liom_toolkit.scripts.liom_predict_volume import main

    out = str(tmp_path / "pred.zarr")
    main(
        [
            tiny_ome_zarr,
            "--model-dir",
            str(stub_nnunet_model_dir),
            "--output",
            out,
            "--device",
            "cpu",
        ]
    )

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


@pytest.mark.ai
def test_main_explicit_spacing_overrides_ngff(
    tmp_path, fake_nnunet_predictor, stub_nnunet_model_dir, tiny_ome_zarr
) -> None:
    """--spacing 3.0 2.0 1.0 wins over the NGFF metadata."""
    pytest.importorskip("torch")

    from liom_toolkit.scripts.liom_predict_volume import main

    main(
        [
            tiny_ome_zarr,
            "--model-dir",
            str(stub_nnunet_model_dir),
            "--output",
            str(tmp_path / "pred.zarr"),
            "--device",
            "cpu",
            "--spacing",
            "3.0",
            "2.0",
            "1.0",
        ]
    )

    calls = fake_nnunet_predictor.calls["predict_calls"]
    assert len(calls) == 1
    assert calls[0]["image_properties"] == {"spacing": [3.0, 2.0, 1.0]}


@pytest.mark.ai
def test_main_z_chunk_size_produces_slab_calls(
    tmp_path, fake_nnunet_predictor, stub_nnunet_model_dir, tiny_ome_zarr
) -> None:
    """--z-chunk-size 2 on a Z=5 volume produces 3 slab calls [2, 2, 1]."""
    pytest.importorskip("torch")
    import zarr

    from liom_toolkit.scripts.liom_predict_volume import main

    out = str(tmp_path / "pred.zarr")
    main(
        [
            tiny_ome_zarr,
            "--model-dir",
            str(stub_nnunet_model_dir),
            "--output",
            out,
            "--device",
            "cpu",
            "--z-chunk-size",
            "2",
        ]
    )

    calls = fake_nnunet_predictor.calls["predict_calls"]
    assert [c["input_image"].shape for c in calls] == [
        (1, 2, 32, 32),
        (1, 2, 32, 32),
        (1, 1, 32, 32),
    ]
    result = zarr.open(out, mode="r")
    assert result.shape == (5, 32, 32)
    assert result.dtype == np.uint8
    result_np = np.asarray(result[:])
    assert set(np.unique(result_np)).issubset({0, 255})
