"""Tests for the OME-Zarr/PNG → nnU-Net v2 raw-format converter + CLI.

Covers four concerns:

* **Round-trip** — ``prepare_nnunet_2d`` writes ``imagesTr/<case>_0000.png`` +
  ``labelsTr/<case>.png`` + ``dataset.json`` in the nnU-Net v2 raw-format
  layout. The written images round-trip (data equality with the input), the
  file counts match ``len(image_paths)``, and ``dataset.json`` has the
  nnU-Net v2 schema (``channel_names`` / ``labels`` / ``numTraining`` /
  ``file_ending`` / ``dataset_name``).
* **Input-path validation** — a nonexistent input path raises ``ValueError``
  with the offending path (no silent wrong-data fallback — AGENTS §2).
* **CLI** — ``liom-prepare-nnunet-dataset`` is registered in
  ``[project.scripts]`` and ``main(["<input>", "<output>", "--dataset-id",
  "101"])`` produces the expected output directory.
* **All paths parameterized** — no hardcoded lab paths (AGENTS §1); the
  converter takes ``image_paths`` / ``label_paths`` / ``output_dir`` as
  function params, the CLI takes them as positionals.

These tests do NOT need torch (the converter is pure imageio + json IO) — no
``importorskip``.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest


def _write_synthetic_slices(
    dir_path: Path, n: int = 3, shape: tuple[int, int] = (32, 32)
) -> tuple[list[str], list[str], list[np.ndarray], list[np.ndarray]]:
    """Write ``n`` synthetic image + label PNG pairs into ``dir_path``.

    Returns ``(image_paths, label_paths, images, labels)`` so the caller can
    assert round-trip data equality against the written files.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    label_paths: list[str] = []
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for i in range(n):
        img = np.zeros(shape, dtype=np.uint8)
        img[8:24, 8:24] = 200  # a bright square "vessel-ish" region
        lbl = np.zeros(shape, dtype=np.uint8)
        lbl[8:24, 8:24] = 1  # binary vessel mask
        img_p = dir_path / f"img_{i:02d}.png"
        lbl_p = dir_path / f"img_{i:02d}_mask.png"
        iio.imwrite(img_p, img)
        iio.imwrite(lbl_p, lbl)
        image_paths.append(str(img_p))
        label_paths.append(str(lbl_p))
        images.append(img)
        labels.append(lbl)
    return image_paths, label_paths, images, labels


def test_prepare_nnunet_2d_round_trip(tmp_path) -> None:
    """prepare_nnunet_2d writes the nnU-Net v2 raw layout and round-trips.

    Asserts: (a) imagesTr has N files matching len(image_paths), (b) labelsTr
    has N files, (c) dataset.json has the nnU-Net v2 schema keys with correct
    values, (d) a written image round-trips (data equality with the input).
    """
    from liom_toolkit.scripts.prepare_nnunet_dataset import prepare_nnunet_2d

    src = tmp_path / "src"
    image_paths, label_paths, images, _labels = _write_synthetic_slices(src, n=3)
    out_dir = tmp_path / "Dataset101_LIOM6p5"

    prepare_nnunet_2d(
        image_paths=image_paths,
        label_paths=label_paths,
        output_dir=str(out_dir),
        dataset_id=101,
    )

    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    assert images_tr.is_dir(), "imagesTr directory must be created"
    assert labels_tr.is_dir(), "labelsTr directory must be created"

    img_files = sorted(images_tr.glob("case_*_0000.png"))
    lbl_files = sorted(labels_tr.glob("case_*.png"))
    assert len(img_files) == len(image_paths), (
        f"imagesTr must have {len(image_paths)} files, got {len(img_files)}"
    )
    assert len(lbl_files) == len(label_paths), (
        f"labelsTr must have {len(label_paths)} files, got {len(lbl_files)}"
    )

    # dataset.json schema (nnU-Net v2).
    dataset_json_path = out_dir / "dataset.json"
    assert dataset_json_path.is_file(), "dataset.json must be written"
    dataset_json = json.loads(dataset_json_path.read_text())
    assert "channel_names" in dataset_json, "dataset.json must have channel_names"
    assert "labels" in dataset_json, "dataset.json must have labels"
    assert "numTraining" in dataset_json, "dataset.json must have numTraining"
    assert "file_ending" in dataset_json, "dataset.json must have file_ending"
    assert "dataset_name" in dataset_json, "dataset.json must have dataset_name"
    assert dataset_json["numTraining"] == len(image_paths)
    assert dataset_json["file_ending"] == ".png"
    assert dataset_json["labels"] == {"background": 0, "vessel": 1}
    assert dataset_json["dataset_name"] == "Dataset101_LIOM6p5"

    # Round-trip: read back the first image and compare to the input.
    written = iio.imread(img_files[0])
    np.testing.assert_array_equal(written, images[0])


def test_prepare_nnunet_2d_raises_on_nonexistent_input(tmp_path) -> None:
    """prepare_nnunet_2d raises ValueError when an input path does not exist.

    No silent wrong-data fallback (AGENTS §2): the offending path is in the
    message so the failure is actionable.
    """
    from liom_toolkit.scripts.prepare_nnunet_dataset import prepare_nnunet_2d

    nonexistent = str(tmp_path / "ghost.png")
    with pytest.raises(ValueError, match="input image does not exist"):
        prepare_nnunet_2d(
            image_paths=[nonexistent],
            label_paths=[str(tmp_path / "lbl.png")],
            output_dir=str(tmp_path / "out"),
            dataset_id=999,
        )


def test_prepare_nnunet_2d_raises_on_nonexistent_label(tmp_path) -> None:
    """prepare_nnunet_2d raises ValueError when a label path does not exist."""
    from liom_toolkit.scripts.prepare_nnunet_dataset import prepare_nnunet_2d

    src = tmp_path / "src"
    src.mkdir()
    img_p = src / "img.png"
    iio.imwrite(img_p, np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="label does not exist"):
        prepare_nnunet_2d(
            image_paths=[str(img_p)],
            label_paths=[str(tmp_path / "ghost_mask.png")],
            output_dir=str(tmp_path / "out"),
            dataset_id=999,
        )


def test_prepare_nnunet_2d_raises_on_length_mismatch(tmp_path) -> None:
    """prepare_nnunet_2d raises ValueError when image/label counts differ."""
    from liom_toolkit.scripts.prepare_nnunet_dataset import prepare_nnunet_2d

    src = tmp_path / "src"
    image_paths, label_paths, _imgs, _lbls = _write_synthetic_slices(src, n=2)
    # Pass only one label for two images.
    with pytest.raises(ValueError, match="length mismatch"):
        prepare_nnunet_2d(
            image_paths=image_paths,
            label_paths=[label_paths[0]],
            output_dir=str(tmp_path / "out"),
            dataset_id=999,
        )


def test_prepare_nnunet_cli_creates_output(tmp_path, monkeypatch) -> None:
    """The liom-prepare-nnunet-dataset CLI creates the nnU-Net raw dataset.

    Sets ``sys.argv`` to ``["liom-prepare-nnunet-dataset", "<input>",
    "<output>", "--dataset-id", "101"]`` and calls ``main()``, then asserts
    the output directory with imagesTr/labelsTr/dataset.json is created. The
    CLI takes a single input directory of PNG slices (with matching
    ``*_mask.png`` labels) and writes the nnU-Net raw layout.
    """
    import sys

    from liom_toolkit.scripts.prepare_nnunet_dataset import main

    src = tmp_path / "src"
    _write_synthetic_slices(src, n=2)
    out_dir = tmp_path / "Dataset101_LIOM6p5"

    monkeypatch.setattr(
        sys,
        "argv",
        ["liom-prepare-nnunet-dataset", str(src), str(out_dir), "--dataset-id", "101"],
    )
    main()

    assert (out_dir / "imagesTr").is_dir()
    assert (out_dir / "labelsTr").is_dir()
    assert (out_dir / "dataset.json").is_file()
    dataset_json = json.loads((out_dir / "dataset.json").read_text())
    assert dataset_json["numTraining"] == 2


def test_prepare_nnunet_cli_errors_on_nonexistent_input(tmp_path, monkeypatch, capsys) -> None:
    """The CLI exits 2 with a clear message when the input path does not exist.

    Uses ``parser.error`` (the established boundary-validation pattern in
    ``liom_train_model.py``) so a bad path surfaces as exit 2 with the
    offending value, not a raw traceback from inside the converter.
    """
    import sys

    from liom_toolkit.scripts.prepare_nnunet_dataset import main

    nonexistent = str(tmp_path / "ghost_dir")
    monkeypatch.setattr(
        sys,
        "argv",
        ["liom-prepare-nnunet-dataset", nonexistent, str(tmp_path / "out"), "--dataset-id", "999"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert nonexistent in captured.err or nonexistent in captured.out


def test_liom_prepare_nnunet_dataset_console_script_registered() -> None:
    """liom-prepare-nnunet-dataset is registered in [project.scripts]."""
    import tomllib

    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    scripts = data["project"]["scripts"]
    assert "liom-prepare-nnunet-dataset" in scripts, (
        "liom-prepare-nnunet-dataset must be registered in [project.scripts]"
    )
    assert scripts["liom-prepare-nnunet-dataset"] == (
        "liom_toolkit.scripts.prepare_nnunet_dataset:main"
    )
