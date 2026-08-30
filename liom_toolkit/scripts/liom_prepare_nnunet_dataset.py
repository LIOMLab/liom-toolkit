#!/usr/bin/env python3
"""CLI: convert labeled 2D coronal PNG slices to nnU-Net v2 raw format.

Reads a directory of ``<case>.png`` image slices paired with
``<case>_mask.png`` label slices, and writes the nnU-Net v2 raw-format
layout::

    <output_dir>/
      imagesTr/case_0000_0000.png   # one per image, channel 0
      imagesTr/case_0001_0000.png
      ...
      labelsTr/case_0000.png        # one per label
      labelsTr/case_0001.png
      ...
      dataset.json                  # channel_names, labels, numTraining, ...

All paths are parameterized (CLI args / function params) — no hardcoded lab
paths (AGENTS §1). Input paths are validated before writing (raise
``ValueError`` with the offending path — AGENTS §2).
"""

from __future__ import annotations

import argparse
import json
import logging
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
        "input_path",
        help="Path to a directory of PNG image slices with matching <name>_mask.png labels",
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


def prepare_nnunet_2d(
    image_paths: list[str],
    label_paths: list[str],
    output_dir: str,
    dataset_id: int = 999,
    file_ending: str = ".png",
    dataset_name: str = "LIOM6p5",
) -> None:
    """Convert labeled 2D slices to nnU-Net v2 raw format.

    Writes ``imagesTr/<case>_0000<file_ending>`` per image,
    ``labelsTr/<case><file_ending>`` per label, and ``dataset.json`` with the
    nnU-Net v2 schema (``channel_names`` / ``labels`` / ``numTraining`` /
    ``file_ending`` / ``dataset_name``).

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

    Raises
    ------
    ValueError
        If ``image_paths`` and ``label_paths`` have different lengths, or if
        any input image/label path does not exist (the offending path is in
        the message).
    """
    if len(image_paths) != len(label_paths):
        raise ValueError(
            f"prepare_nnunet_2d: length mismatch — {len(image_paths)} images vs "
            f"{len(label_paths)} labels (one label per image required)"
        )

    for img_p in image_paths:
        if not Path(img_p).is_file():
            raise ValueError(f"prepare_nnunet_2d: input image does not exist: {img_p}")
    for lbl_p in label_paths:
        if not Path(lbl_p).is_file():
            raise ValueError(f"prepare_nnunet_2d: label does not exist: {lbl_p}")

    root = Path(output_dir)
    images_tr = root / "imagesTr"
    labels_tr = root / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    for i, (img_p, lbl_p) in enumerate(zip(image_paths, label_paths, strict=True)):
        case = f"case_{i:04d}"
        img = iio.imread(img_p)
        lbl = iio.imread(lbl_p)
        iio.imwrite(images_tr / f"{case}_0000{file_ending}", img)
        iio.imwrite(labels_tr / f"{case}{file_ending}", lbl)

    dataset_json = {
        "channel_names": {"0": "LSFM_6p5um"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": len(image_paths),
        "file_ending": file_ending,
        "dataset_name": f"Dataset{dataset_id:03d}_{dataset_name}",
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
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
    """Prepare an nnU-Net v2 raw dataset from a directory of labeled PNG slices.

    Parses CLI arguments, validates the input path exists (``parser.error``
    exits 2 with the offending value on a bad path), configures logging via
    ``basicConfig`` on the root logger, discovers the image/label pairs in
    the input directory, and delegates to :func:`prepare_nnunet_2d`.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.is_dir():
        parser.error(f"input path does not exist or is not a directory: {args.input_path}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )

    image_paths, label_paths = _discover_pairs(input_path)

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
