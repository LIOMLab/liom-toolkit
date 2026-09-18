"""Tests for the OME-Zarr/PNG → nnU-Net v2 raw-format converter + CLI.

Covers five concerns:

* **Round-trip** — ``prepare_nnunet_2d`` writes ``imagesTr/<case>_0000.png`` +
  ``labelsTr/<case>.png`` + ``dataset.json`` + ``case_brain_manifest.json``
  in the nnU-Net v2 raw-format layout. Case ids carry brain provenance:
  ``<input_dir_name>_<stem>`` by default (never anonymous ``case_NNNN`` —
  anonymous ids make a per-brain split unverifiable, the exact failure a
  random nnU-Net split over unnamed cases produced), or an explicit
  ``case_names`` override.
* **Manifest** — ``case_brain_manifest.json`` maps ``{case_id: brain}`` so
  the train/val split is auditable by other labs; ``brain_names`` defaults
  to the input dir name.
* **Validation** — nonexistent input paths, image/label length mismatch,
  ``case_names``/``brain_names`` length mismatch, duplicate case ids, and
  case ids ending in a ``_NNNN`` modality suffix (which would alias with
  nnU-Net's ``<case>_0000.png`` channel naming) all raise ``ValueError``
  naming the offender (no silent wrong-data fallback — AGENTS §2).
* **CLI** — ``liom-prepare-nnunet-dataset`` accepts N input dirs (one per
  brain) + an output dir; each input dir contributes its dir name as the
  case prefix. A nonexistent input dir exits 2 via ``parser.error``.
* **Rebuild contract** — a non-empty output dir raises ``FileExistsError``;
  re-running the converter never silently merges into a stale dataset.

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

    Asserts: (a) imagesTr has N files matching len(image_paths), named
    ``<input_dir_name>_<stem>_0000.png`` (brain-provenance ids, never
    anonymous ``case_NNNN``), (b) labelsTr has N files, (c) dataset.json has
    the nnU-Net v2 schema keys with correct values, (d) a written image
    round-trips (data equality with the input).
    """
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d

    src = tmp_path / "s23"
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

    img_files = sorted(images_tr.glob("s23_*_0000.png"))
    lbl_files = sorted(labels_tr.glob("s23_*.png"))
    assert len(img_files) == len(image_paths), (
        f"imagesTr must have {len(image_paths)} files, got {len(img_files)}"
    )
    assert len(lbl_files) == len(label_paths), (
        f"labelsTr must have {len(label_paths)} files, got {len(lbl_files)}"
    )
    # No anonymous case_NNNN emission anywhere in the output.
    assert not list(images_tr.glob("case_*")), (
        "anonymous case_* names must never be emitted — they lose brain provenance"
    )
    assert not list(labels_tr.glob("case_*")), (
        "anonymous case_* names must never be emitted — they lose brain provenance"
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


def test_prepare_nnunet_2d_writes_case_brain_manifest(tmp_path) -> None:
    """Every conversion writes case_brain_manifest.json mapping case → brain.

    The manifest is the audit record other labs use to verify the train/val
    split is per-brain: default brain = the input dir name, default case id =
    ``<dir_name>_<stem>``. Keys are sorted in the written JSON.
    """
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d

    src = tmp_path / "s24"
    image_paths, label_paths, _imgs, _lbls = _write_synthetic_slices(src, n=2)
    out_dir = tmp_path / "Dataset101_LIOM6p5"

    prepare_nnunet_2d(
        image_paths=image_paths,
        label_paths=label_paths,
        output_dir=str(out_dir),
        dataset_id=101,
    )

    manifest_path = out_dir / "case_brain_manifest.json"
    assert manifest_path.is_file(), "case_brain_manifest.json must be written"
    manifest = json.loads(manifest_path.read_text())
    expected = {"s24_img_00": "s24", "s24_img_01": "s24"}
    assert manifest == expected, f"manifest must map case → brain, got {manifest}"


def test_prepare_nnunet_2d_explicit_case_names_and_brain_names(tmp_path) -> None:
    """Explicit case_names/brain_names override the derived defaults.

    ``case_names[i]`` wins over ``<dir>_<stem>`` derivation; ``brain_names[i]``
    wins over the parent dir name. The manifest reflects the explicit values.
    """
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d

    src = tmp_path / "whatever"
    image_paths, label_paths, _imgs, _lbls = _write_synthetic_slices(src, n=2)
    out_dir = tmp_path / "Dataset101_LIOM6p5"

    prepare_nnunet_2d(
        image_paths=image_paths,
        label_paths=label_paths,
        output_dir=str(out_dir),
        dataset_id=101,
        case_names=["s23_575", "s23_700"],
        brain_names=["s23", "s23"],
    )

    assert (out_dir / "imagesTr" / "s23_575_0000.png").is_file()
    assert (out_dir / "imagesTr" / "s23_700_0000.png").is_file()
    assert (out_dir / "labelsTr" / "s23_575.png").is_file()
    assert (out_dir / "labelsTr" / "s23_700.png").is_file()
    manifest = json.loads((out_dir / "case_brain_manifest.json").read_text())
    assert manifest == {"s23_575": "s23", "s23_700": "s23"}


def test_prepare_nnunet_2d_rejects_duplicate_case_ids(tmp_path) -> None:
    """Duplicate derived or explicit case ids raise ValueError naming them.

    Two identically-named slices under identically-named parent dirs would
    collide on ``<dir>_<stem>``; a silent overwrite would drop a labeled slice
    from the dataset (silent data loss).
    """
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d

    src_a = tmp_path / "a" / "slices"
    src_b = tmp_path / "b" / "slices"
    imgs_a, lbls_a, _ia, _la = _write_synthetic_slices(src_a, n=1)
    imgs_b, lbls_b, _ib, _lb = _write_synthetic_slices(src_b, n=1)

    with pytest.raises(ValueError, match="slices_img_00"):
        prepare_nnunet_2d(
            image_paths=imgs_a + imgs_b,
            label_paths=lbls_a + lbls_b,
            output_dir=str(tmp_path / "out"),
            dataset_id=101,
        )

    src = tmp_path / "s23"
    image_paths, label_paths, _imgs, _lbls = _write_synthetic_slices(src, n=2)
    with pytest.raises(ValueError, match="dup_case"):
        prepare_nnunet_2d(
            image_paths=image_paths,
            label_paths=label_paths,
            output_dir=str(tmp_path / "out2"),
            dataset_id=101,
            case_names=["dup_case", "dup_case"],
        )


def test_prepare_nnunet_2d_rejects_modality_suffix_case_ids(tmp_path) -> None:
    """A case id ending ``_NNNN`` raises ValueError — it aliases nnU-Net naming.

    nnU-Net writes images as ``<case>_0000.png``; a case id like ``s23_0000``
    would produce ``s23_0000_0000.png`` whose case stem parses ambiguously.
    """
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d

    src = tmp_path / "s23"
    image_paths, label_paths, _imgs, _lbls = _write_synthetic_slices(src, n=1)

    with pytest.raises(ValueError, match="s23_0000"):
        prepare_nnunet_2d(
            image_paths=image_paths,
            label_paths=label_paths,
            output_dir=str(tmp_path / "out"),
            dataset_id=101,
            case_names=["s23_0000"],
        )


def test_prepare_nnunet_2d_escapes_numeric_stem_modality_suffix(tmp_path) -> None:
    """Derived ids from index-named slices escape the ``_NNNN`` modality suffix.

    Real labeled slices are routinely named by index (``1000.png`` +
    ``1000_mask.png``); the derived ``<dir>_<stem>`` id ``s23_1000`` would
    alias with nnU-Net's ``<case>_0000`` channel naming, so the digit tail
    is letter-prefixed (``s23_s1000``) instead of failing — the alias never
    reaches nnU-Net and the slice is not silently dropped from the dataset.
    """
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d

    src = tmp_path / "s23"
    src.mkdir(parents=True)
    img = np.zeros((16, 16), dtype=np.uint8)
    lbl = np.zeros((16, 16), dtype=np.uint8)
    lbl[4:12, 4:12] = 1
    for stem in ("575", "1000"):
        iio.imwrite(src / f"{stem}.png", img)
        iio.imwrite(src / f"{stem}_mask.png", lbl)

    out_dir = tmp_path / "Dataset101_LIOM6p5"
    prepare_nnunet_2d(
        image_paths=[str(src / "575.png"), str(src / "1000.png")],
        label_paths=[str(src / "575_mask.png"), str(src / "1000_mask.png")],
        output_dir=str(out_dir),
        dataset_id=101,
    )

    assert (out_dir / "imagesTr" / "s23_575_0000.png").is_file()
    assert (out_dir / "imagesTr" / "s23_s1000_0000.png").is_file()
    assert (out_dir / "labelsTr" / "s23_s1000.png").is_file()
    manifest = json.loads((out_dir / "case_brain_manifest.json").read_text())
    assert manifest == {"s23_575": "s23", "s23_s1000": "s23"}


def test_prepare_nnunet_2d_rejects_case_names_length_mismatch(tmp_path) -> None:
    """case_names/brain_names must be parallel to image_paths."""
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d

    src = tmp_path / "s23"
    image_paths, label_paths, _imgs, _lbls = _write_synthetic_slices(src, n=2)

    with pytest.raises(ValueError, match="case_names"):
        prepare_nnunet_2d(
            image_paths=image_paths,
            label_paths=label_paths,
            output_dir=str(tmp_path / "out"),
            dataset_id=101,
            case_names=["only_one"],
        )
    with pytest.raises(ValueError, match="brain_names"):
        prepare_nnunet_2d(
            image_paths=image_paths,
            label_paths=label_paths,
            output_dir=str(tmp_path / "out2"),
            dataset_id=101,
            brain_names=["s23"],
        )


def test_prepare_nnunet_2d_raises_on_nonexistent_input(tmp_path) -> None:
    """prepare_nnunet_2d raises ValueError when an input path does not exist.

    No silent wrong-data fallback (AGENTS §2): the offending path is in the
    message so the failure is actionable.
    """
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d

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
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d

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
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d

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


def test_prepare_nnunet_2d_raises_on_nonempty_output_dir(tmp_path) -> None:
    """prepare_nnunet_2d raises FileExistsError on a non-empty output dir.

    Reusing an existing dataset dir leaves stale case files from a previous
    run: ``dataset.json`` reflects only the new count while the leftover files
    silently contaminate fingerprint extraction and training (silent
    wrong-data). The rebuild path is an explicit remove — never a silent
    merge. An existing EMPTY directory is fine to reuse.
    """
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import prepare_nnunet_2d

    src = tmp_path / "src"
    image_paths, label_paths, _imgs, _lbls = _write_synthetic_slices(src, n=2)
    out_dir = tmp_path / "Dataset101_LIOM6p5"
    (out_dir / "imagesTr").mkdir(parents=True)
    (out_dir / "imagesTr" / "s23_0099_0000.png").write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="non-empty"):
        prepare_nnunet_2d(
            image_paths=image_paths,
            label_paths=label_paths,
            output_dir=str(out_dir),
            dataset_id=101,
        )

    # An existing but empty dir is accepted.
    empty_dir = tmp_path / "Dataset102_LIOM6p5"
    empty_dir.mkdir()
    prepare_nnunet_2d(
        image_paths=image_paths,
        label_paths=label_paths,
        output_dir=str(empty_dir),
        dataset_id=102,
    )
    assert (empty_dir / "dataset.json").is_file()


def test_prepare_nnunet_cli_creates_output(tmp_path, monkeypatch) -> None:
    """The liom-prepare-nnunet-dataset CLI creates the nnU-Net raw dataset.

    Sets ``sys.argv`` to ``["liom-prepare-nnunet-dataset", "<input>",
    "<output>", "--dataset-id", "101"]`` and calls ``main()``, then asserts
    the output directory with imagesTr/labelsTr/dataset.json is created. A
    single input dir still works under the ``nargs="+"`` input contract.
    """
    import sys

    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import main

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


def test_prepare_nnunet_cli_multiple_input_dirs(tmp_path, monkeypatch) -> None:
    """The CLI accepts N input dirs — one per brain — + an output dir.

    ``liom-prepare-nnunet-dataset <dir_s23> <dir_s24> <output>``: each input
    dir contributes its dir name as the case prefix, and the manifest maps
    every case to its brain so the LOO split is auditable.
    """
    import sys

    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import main

    dir_s23 = tmp_path / "s23"
    dir_s24 = tmp_path / "s24"
    _write_synthetic_slices(dir_s23, n=2)
    _write_synthetic_slices(dir_s24, n=1)
    out_dir = tmp_path / "Dataset101_LIOM6p5"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "liom-prepare-nnunet-dataset",
            str(dir_s23),
            str(dir_s24),
            str(out_dir),
            "--dataset-id",
            "101",
        ],
    )
    main()

    img_names = sorted(p.name for p in (out_dir / "imagesTr").glob("*.png"))
    assert img_names == ["s23_img_00_0000.png", "s23_img_01_0000.png", "s24_img_00_0000.png"]
    dataset_json = json.loads((out_dir / "dataset.json").read_text())
    assert dataset_json["numTraining"] == 3
    manifest = json.loads((out_dir / "case_brain_manifest.json").read_text())
    assert manifest == {
        "s23_img_00": "s23",
        "s23_img_01": "s23",
        "s24_img_00": "s24",
    }


def test_prepare_nnunet_cli_errors_on_nonexistent_input(tmp_path, monkeypatch, capsys) -> None:
    """The CLI exits 2 with a clear message when the input path does not exist.

    Uses ``parser.error`` (the established boundary-validation pattern in
    ``liom_train_model.py``) so a bad path surfaces as exit 2 with the
    offending value, not a raw traceback from inside the converter.
    """
    import sys

    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import main

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
        "liom_toolkit.scripts.liom_prepare_nnunet_dataset:main"
    )
