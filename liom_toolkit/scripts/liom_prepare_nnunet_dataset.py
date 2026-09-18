#!/usr/bin/env python3
"""CLI: convert labeled 2D coronal PNG slices to nnU-Net v2 raw format.

Reads one or more directories of ``<case>.png`` image slices paired with
``<case>_mask.png`` label slices (one directory per brain), and writes the
nnU-Net v2 raw-format layout::

    <output_dir>/
      imagesTr/<case_id>_0000.png   # one per image, channel 0
      ...
      labelsTr/<case_id>.png        # one per label
      ...
      dataset.json                  # channel_names, labels, numTraining, ...
      case_brain_manifest.json      # {case_id: brain} — split audit record

Case ids carry brain provenance: ``<input_dir_name>_<stem>`` by default (or
an explicit ``case_names`` override), never anonymous ``case_NNNN`` — an
nnU-Net random split over anonymous cases cannot be verified as per-brain.
The ``case_brain_manifest.json`` mapping makes the train/val split auditable
and feeds ``write_loo_splits`` when a hand-written split is needed.

All paths are parameterized (CLI args / function params) — no hardcoded lab
paths (AGENTS §1). Input paths are validated before writing (raise
``ValueError`` with the offending path — AGENTS §2).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from liom_toolkit.scripts._common import build_common_parser

logger = logging.getLogger(__name__)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the nnU-Net dataset-preparation CLI.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser (call ``parse_args()`` on it).
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[build_common_parser()],
    )
    p.add_argument(
        "input_paths",
        nargs="+",
        help="Path(s) to directories of PNG image slices with matching <name>_mask.png "
        "labels — one directory per brain; each dir name becomes the case-id prefix",
    )
    p.add_argument(
        "output_dir",
        help="Path to the nnU-Net raw dataset directory to write (e.g. Dataset101_LIOM6p5)",
    )
    p.add_argument(
        "--dataset-id",
        type=int,
        required=True,
        help="nnU-Net dataset id (e.g. 101 → Dataset101_LIOM6p5)",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default="LIOM6p5",
        help="Dataset short name (default=%(default)s)",
    )
    p.add_argument(
        "--file-ending",
        type=str,
        default=".png",
        help="File ending for the written images/labels (default=%(default)s)",
    )
    return p


def _sanitize_case_id(raw: str) -> str:
    """Sanitize a case id to nnU-Net-safe characters.

    Keeps ``[A-Za-z0-9_-]``; every other character becomes ``_``.

    Returns
    -------
    str
        The sanitized case id.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)


def _derive_case_ids(
    image_paths: list[str],
    case_names: list[str] | None,
) -> list[str]:
    """Derive and validate the case id for each image.

    Default: ``<parent_dir_name>_<stem>`` so every id carries brain
    provenance. An explicit ``case_names[i]`` wins and is sanitized +
    validated identically. Anonymous ``case_NNNN`` ids are never emitted —
    an nnU-Net random split over anonymous cases cannot be verified as
    per-brain.

    A derived id that would end in a ``_NNNN`` modality suffix (real slice
    files are routinely named by index — ``1000.png`` → ``s23_1000``) is
    escaped by letter-prefixing the digit tail (``s23_1000`` →
    ``s23_s1000``) so the alias can never reach nnU-Net. An explicit
    ``case_names`` id ending ``_NNNN`` still raises — a caller-supplied
    name is a typo/contract violation, not a naming convention to escape.

    Returns
    -------
    list[str]
        The validated case ids, parallel to ``image_paths``.

    Raises
    ------
    ValueError
        If a case id is empty after sanitization, an explicit
        ``case_names`` id ends with a ``_NNNN`` modality suffix (it would
        alias with nnU-Net's ``<case>_0000`` channel naming), or an id
        collides with another case id. The offending id is named in the
        message.
    """
    case_ids: list[str] = []
    for i, img_p in enumerate(image_paths):
        raw = (
            case_names[i]
            if case_names is not None
            else f"{Path(img_p).parent.name}_{Path(img_p).stem}"
        )
        case = _sanitize_case_id(raw)
        if not case:
            raise ValueError(
                f"prepare_nnunet_2d: empty case id derived from {raw!r} (image {img_p})"
            )
        if re.search(r"_\d{4}$", case):
            if case_names is not None:
                raise ValueError(
                    f"prepare_nnunet_2d: case id {case!r} ends with a _NNNN "
                    "modality suffix — it would alias with nnU-Net's "
                    "<case>_0000 channel naming; rename the slice or pass an "
                    "explicit case_names override"
                )
            case = re.sub(r"_(\d{4})$", r"_s\1", case)
        case_ids.append(case)
    duplicates = sorted({c for c in case_ids if case_ids.count(c) > 1})
    if duplicates:
        raise ValueError(
            f"prepare_nnunet_2d: duplicate case ids {duplicates} — each "
            "case id must be unique; rename the input slices or pass an "
            "explicit case_names override"
        )
    return case_ids


def _normalize_label(label: np.ndarray, source_path: str) -> np.ndarray:
    """Map a binary vessel mask to the declared ``{0, 1}`` label scheme.

    Lab masks arrive in both conventions — ``{0, 1}`` and ``{0, 255}`` —
    sometimes mixed within one brain's directory. ``dataset.json`` declares
    ``{background: 0, vessel: 1}``, so a ``{0, 255}`` mask is remapped
    (``255 → 1``) before writing; a ``{0, 1}`` mask passes through
    unchanged. Any other value set (e.g. ``{0, 1, 2}`` or ``{0, 1, 255}``)
    is ambiguous for a binary label and raises ``ValueError`` naming the
    values and file — silently binarizing a multi-class or corrupt mask
    would be silent wrong-data (AGENTS §2).

    Returns
    -------
    np.ndarray
        The label array with values in ``{0, 1}`` (uint8).

    Raises
    ------
    ValueError
        If the mask contains values outside the ``{0, 1}`` / ``{0, 255}``
        conventions.
    """
    values = set(np.unique(label).tolist())
    if values <= {0, 1}:
        return np.asarray(label, dtype=np.uint8)
    if values <= {0, 255}:
        return (np.asarray(label) == 255).astype(np.uint8)
    raise ValueError(
        f"prepare_nnunet_2d: label mask {source_path} has unexpected values "
        f"{sorted(values)} — expected a binary vessel mask ({0, 1} or "
        f"{{0, 255}}); dataset.json declares only background=0, vessel=1"
    )


def prepare_nnunet_2d(
    image_paths: list[str],
    label_paths: list[str],
    output_dir: str,
    dataset_id: int = 999,
    file_ending: str = ".png",
    dataset_name: str = "LIOM6p5",
    *,
    case_names: list[str] | None = None,
    brain_names: list[str] | None = None,
) -> None:
    """Convert labeled 2D slices to nnU-Net v2 raw format.

    Writes ``imagesTr/<case>_0000<file_ending>`` per image,
    ``labelsTr/<case><file_ending>`` per label, ``dataset.json`` with the
    nnU-Net v2 schema (``channel_names`` / ``labels`` / ``numTraining`` /
    ``file_ending`` / ``dataset_name``), and ``case_brain_manifest.json``
    mapping ``{case_id: brain}`` — the audit record that lets other labs
    verify the train/val split is per-brain.

    Parameters
    ----------
    image_paths : list[str]
        Paths to the input 2D image slices (any imageio-readable format).
    label_paths : list[str]
        Paths to the matching label masks (one per image, same order).
    output_dir : str
        Path to the nnU-Net raw dataset directory to write.
    dataset_id : int
        The nnU-Net dataset id (used in ``dataset_name``).
    file_ending : str
        File ending for the written images/labels.
    dataset_name : str
        Dataset short name (combined with ``dataset_id`` to form
        ``Dataset{id:03d}_{name}``).
    case_names : list[str] | None
        Optional explicit case ids, parallel to ``image_paths``. Default:
        ``<parent_dir_name>_<stem>`` per image.
    brain_names : list[str] | None
        Optional brain label per image, parallel to ``image_paths``.
        Default: the image's parent directory name.

    Raises
    ------
    FileExistsError
        If ``output_dir`` exists and is non-empty -- leftover
        case files from a previous run would silently contaminate
        the new dataset (``dataset.json``'s ``numTraining`` only reflects
        the new count, but nnU-Net's fingerprint extraction and integrity
        check see the stale files). Remove the directory or pick a fresh
        ``output_dir``.
    ValueError
        If ``image_paths`` and ``label_paths`` have different lengths, if
        ``case_names``/``brain_names`` lengths differ from
        ``len(image_paths)``, if a case id is invalid (empty, ``_NNNN``
        suffix, or duplicated), if a label mask contains values outside the
        binary ``{0, 1}`` / ``{0, 255}`` conventions, or if any input
        image/label path does not exist (the offending value is in the
        message).
    """
    if len(image_paths) != len(label_paths):
        raise ValueError(
            f"prepare_nnunet_2d: length mismatch — {len(image_paths)} images vs "
            f"{len(label_paths)} labels (one label per image required)"
        )
    if case_names is not None and len(case_names) != len(image_paths):
        raise ValueError(
            f"prepare_nnunet_2d: case_names length mismatch — "
            f"{len(case_names)} names vs {len(image_paths)} images"
        )
    if brain_names is not None and len(brain_names) != len(image_paths):
        raise ValueError(
            f"prepare_nnunet_2d: brain_names length mismatch — "
            f"{len(brain_names)} names vs {len(image_paths)} images"
        )

    for img_p in image_paths:
        if not Path(img_p).is_file():
            raise ValueError(f"prepare_nnunet_2d: input image does not exist: {img_p}")
    for lbl_p in label_paths:
        if not Path(lbl_p).is_file():
            raise ValueError(f"prepare_nnunet_2d: label does not exist: {lbl_p}")

    case_ids = _derive_case_ids(image_paths, case_names)

    root = Path(output_dir)
    # Refuse to write into a non-empty dataset dir: a re-run with fewer
    # cases would leave stale case files that dataset.json no longer
    # accounts for, contaminating fingerprint extraction and training with
    # data the caller did not pass. An existing EMPTY dir is fine to reuse.
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"prepare_nnunet_2d: output_dir exists and is non-empty: {root} -- "
            "remove it or choose a fresh directory so stale case files from a "
            "previous run cannot contaminate the dataset"
        )
    images_tr = root / "imagesTr"
    labels_tr = root / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}
    for case, img_p, lbl_p in zip(case_ids, image_paths, label_paths, strict=True):
        img = iio.imread(img_p)
        lbl = _normalize_label(iio.imread(lbl_p), lbl_p)
        iio.imwrite(images_tr / f"{case}_0000{file_ending}", img)
        iio.imwrite(labels_tr / f"{case}{file_ending}", lbl)
    for i, (case, img_p) in enumerate(zip(case_ids, image_paths, strict=True)):
        brain = brain_names[i] if brain_names is not None else Path(img_p).parent.name
        manifest[case] = brain

    dataset_json = {
        "channel_names": {"0": "LSFM_6p5um"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": len(image_paths),
        "file_ending": file_ending,
        "dataset_name": f"Dataset{dataset_id:03d}_{dataset_name}",
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    (root / "case_brain_manifest.json").write_text(
        json.dumps(dict(sorted(manifest.items())), indent=2)
    )
    logger.info("Wrote nnU-Net dataset %s (%d cases)", root, len(image_paths))


def _load_case_brain_manifest(manifest: Mapping[str, str] | str | Path) -> dict[str, str]:
    """Return the ``{case_id: brain}`` mapping from a Mapping or a JSON path.

    Returns
    -------
    dict[str, str]
        The case id → brain mapping.

    Raises
    ------
    ValueError
        If ``manifest`` is a path that does not exist, or the loaded JSON is
        not an object mapping strings to strings.
    """
    if isinstance(manifest, Mapping):
        # str() round-trip is identity here but pins dict[str, str] for the
        # type checker — dict(manifest) infers Unknown key/value types.
        return {str(k): str(v) for k, v in manifest.items()}
    path = Path(manifest)
    if not path.is_file():
        raise ValueError(f"case_brain_manifest not found: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in loaded.items()
    ):
        raise ValueError(
            f"case_brain_manifest at {path} must be a JSON object mapping "
            "case id (str) to brain name (str)"
        )
    return {str(k): str(v) for k, v in loaded.items()}


def write_loo_splits(
    manifest: Mapping[str, str] | str | Path,
    out_path: str,
    *,
    val_first: str | None = None,
) -> list[dict[str, list[str]]]:
    """Write a verified 2-fold leave-one-brain-out ``splits_final.json``.

    Groups the manifest's case ids by brain and builds two complementary
    folds: fold 0 trains on one brain and validates on ``val_first`` (or the
    lexicographically larger brain by default), fold 1 swaps. nnU-Net's
    ``do_split`` honors a pre-existing splits file verbatim but only *warns*
    on train/val overlap, so :func:`verify_loo_splits` runs on the built
    splits BEFORE the file is written — a malformed split can never reach
    training silently.

    Parameters
    ----------
    manifest : Mapping[str, str] | str | Path
        The ``{case_id: brain}`` mapping, or a path to
        ``case_brain_manifest.json``.
    out_path : str
        Destination for ``splits_final.json`` (e.g.
        ``$nnUNet_preprocessed/<Dataset>/splits_final.json`` — installed
        after ``plan_and_preprocess``, before training).
    val_first : str | None
        Brain validated in fold 0. Default: the lexicographically larger
        brain name (s24 > s23).

    Returns
    -------
    list[dict[str, list[str]]]
        The written splits: ``[{"train": [...], "val": [...]}, ...]``.

    Raises
    ------
    ValueError
        If the manifest does not contain exactly 2 distinct brains, or if
        ``val_first`` is not one of them (the offending names are in the
        message).
    """
    manifest_map = _load_case_brain_manifest(manifest)
    brains = sorted(set(manifest_map.values()))
    if len(brains) != 2:
        raise ValueError(
            f"write_loo_splits: leave-one-brain-out requires exactly 2 "
            f"distinct brains, found {len(brains)}: {brains}"
        )
    val0 = val_first if val_first is not None else brains[-1]
    if val0 not in brains:
        raise ValueError(
            f"write_loo_splits: val_first={val0!r} is not a brain in the manifest {brains}"
        )
    other = brains[0] if brains[1] == val0 else brains[1]

    def _cases_of(brain: str) -> list[str]:
        return sorted(c for c, b in manifest_map.items() if b == brain)

    splits = [
        {"train": _cases_of(other), "val": _cases_of(val0)},
        {"train": _cases_of(val0), "val": _cases_of(other)},
    ]
    verify_loo_splits(splits, manifest_map)
    Path(out_path).write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote 2-fold LOO splits %s (val brains: %s, %s)", out_path, val0, other)
    return splits


def verify_loo_splits(
    splits: list[dict[str, list[str]]],
    manifest: Mapping[str, str] | str | Path,
) -> None:
    """Verify a 2-fold leave-one-brain-out splits structure.

    Raises on every malformed variant — nnU-Net only warns on overlap, so
    this check is the only gate between a hand-written (or hand-corrupted)
    ``splits_final.json`` and a silently leaky training run.

    Parameters
    ----------
    splits : list[dict[str, list[str]]]
        The splits structure: ``[{"train": [...], "val": [...]}, ...]``.
    manifest : Mapping[str, str] | str | Path
        The ``{case_id: brain}`` mapping, or a path to
        ``case_brain_manifest.json``.

    Raises
    ------
    ValueError
        When ``len(splits) != 2``; when a fold lacks ``train``/``val``
        lists; when a case appears in both train and val of the same fold;
        when a brain's cases are split across train/val within a fold
        (brain must be atomic); when a fold's val set is not exactly one
        brain's full case set; when the two folds' val brains are not
        complementary; when a splits case is absent from the manifest; or
        when a manifest case appears in neither fold's val set. Offending
        case ids and brains are named in the message.
    """
    manifest_map = _load_case_brain_manifest(manifest)
    if len(splits) != 2:
        raise ValueError(f"verify_loo_splits: expected exactly 2 folds, got {len(splits)}")

    val_brains: list[str] = []
    for i, fold in enumerate(splits):
        if not isinstance(fold, dict) or "train" not in fold or "val" not in fold:
            raise ValueError(
                f"verify_loo_splits: fold {i} must be a dict with 'train' "
                f"and 'val' case lists, got {fold!r}"
            )
        train = set(fold["train"])
        val = set(fold["val"])

        missing_from_manifest = sorted((train | val) - set(manifest_map))
        if missing_from_manifest:
            raise ValueError(
                f"verify_loo_splits: fold {i} references cases absent from "
                f"the manifest: {missing_from_manifest}"
            )

        overlap = sorted(train & val)
        if overlap:
            raise ValueError(
                f"verify_loo_splits: fold {i} has cases in both train and val: {overlap}"
            )

        train_brains = {manifest_map[c] for c in train}
        val_brain_set = {manifest_map[c] for c in val}
        shared = train_brains & val_brain_set
        if shared:
            offenders = sorted(c for c in train | val if manifest_map[c] in shared)
            raise ValueError(
                f"verify_loo_splits: fold {i} splits brain(s) {sorted(shared)} "
                f"across train/val — a brain's cases must be wholly in train "
                f"or wholly in val (offending cases: {offenders})"
            )

        if len(val_brain_set) != 1:
            raise ValueError(
                f"verify_loo_splits: fold {i} val must contain exactly one "
                f"brain's cases, found brains {sorted(val_brain_set)}"
            )
        (val_brain,) = val_brain_set
        expected_val = {c for c, b in manifest_map.items() if b == val_brain}
        if val != expected_val:
            missing = sorted(expected_val - val)
            extra = sorted(val - expected_val)
            raise ValueError(
                f"verify_loo_splits: fold {i} val is not brain {val_brain}'s "
                f"full case set — missing {missing}, extra {extra}"
            )
        val_brains.append(val_brain)

    if val_brains[0] == val_brains[1]:
        raise ValueError(
            f"verify_loo_splits: both folds validate on brain "
            f"{val_brains[0]!r} — the two folds' val brains must be "
            "complementary (each brain validated exactly once)"
        )

    never_validated = sorted(set(manifest_map) - set(splits[0]["val"]) - set(splits[1]["val"]))
    if never_validated:
        raise ValueError(
            f"verify_loo_splits: manifest cases never appear in a val set: {never_validated}"
        )


def _discover_pairs(input_dir: Path) -> tuple[list[str], list[str]]:
    """Discover (image, label) PNG pairs in ``input_dir``.

    A pair is ``<name>.png`` (image) + ``<name>_mask.png`` (label). Returns
    parallel lists of image paths and label paths, sorted by image name.

    Returns
    -------
    tuple[list[str], list[str]]
        Parallel lists of image paths and label paths, sorted by image name.

    Raises
    ------
    ValueError
        If no image slices are found, or if an image has no matching
        ``<name>_mask.png`` label.
    """
    images = sorted(p for p in input_dir.glob("*.png") if not p.name.endswith("_mask.png"))
    if not images:
        raise ValueError(f"No image PNG slices found in {input_dir} (expected <name>.png files)")
    image_paths: list[str] = []
    label_paths: list[str] = []
    for img in images:
        lbl = img.with_name(f"{img.stem}_mask{img.suffix}")
        if not lbl.is_file():
            raise ValueError(
                f"No matching label for {img.name} — expected {lbl.name} in {input_dir}"
            )
        image_paths.append(str(img))
        label_paths.append(str(lbl))
    return image_paths, label_paths


def main() -> None:
    """Prepare an nnU-Net v2 raw dataset from per-brain directories of PNG slices.

    Parses CLI arguments, validates each input path exists (``parser.error``
    exits 2 with the offending value on a bad path), configures logging via
    ``basicConfig`` on the root logger, discovers the image/label pairs in
    every input directory, and delegates to :func:`prepare_nnunet_2d`. Each
    input dir's name becomes the case-id prefix, preserving brain
    provenance end-to-end.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    for raw in args.input_paths:
        if not Path(raw).is_dir():
            parser.error(f"input path does not exist or is not a directory: {raw}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    image_paths: list[str] = []
    label_paths: list[str] = []
    for raw in args.input_paths:
        imgs, lbls = _discover_pairs(Path(raw))
        image_paths.extend(imgs)
        label_paths.extend(lbls)

    prepare_nnunet_2d(
        image_paths=image_paths,
        label_paths=label_paths,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        file_ending=args.file_ending,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
