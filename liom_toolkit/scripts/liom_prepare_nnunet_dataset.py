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
from pathlib import Path

import imageio.v3 as iio

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

    Returns
    -------
    list[str]
        The validated case ids, parallel to ``image_paths``.

    Raises
    ------
    ValueError
        If a case id is empty after sanitization, ends with a ``_NNNN``
        modality suffix (it would alias with nnU-Net's ``<case>_0000``
        channel naming), or collides with another case id. The offending
        id is named in the message.
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
            raise ValueError(
                f"prepare_nnunet_2d: case id {case!r} ends with a _NNNN "
                "modality suffix — it would alias with nnU-Net's "
                "<case>_0000 channel naming; rename the slice or pass an "
                "explicit case_names override"
            )
        case_ids.append(case)
    duplicates = sorted({c for c in case_ids if case_ids.count(c) > 1})
    if duplicates:
        raise ValueError(
            f"prepare_nnunet_2d: duplicate case ids {duplicates} — each "
            "case id must be unique; rename the input slices or pass an "
            "explicit case_names override"
        )
    return case_ids


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
        suffix, or duplicated), or if any input image/label path does not
        exist (the offending value is in the message).
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
        lbl = iio.imread(lbl_p)
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
