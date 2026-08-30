"""Benchmark orchestrator — trains, predicts, and scores each contender.

``run_benchmark`` is the entry point: it calls each contender's
``train_and_predict`` to obtain predicted masks, binarizes them defensively
(so the eval-metric matrix always receives ``NDArray[np.bool_]``), and
scores them through the ship-gate eval-metric matrix from
:mod:`liom_toolkit.segmentation.vseg.eval_metrics`.

The orchestrator surfaces contender failures (no silent swallow — a
contender that raises propagates the exception). Per-slice
``ValueError`` from the eval metrics (vessel-free slices where a metric is
undefined) is caught and recorded as ``"vessel-free slice — metric
undefined"`` in the result row — never as a NaN (the silent-wrong-data
failure mode AGENTS §2 forbids).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from liom_toolkit.segmentation.vseg.benchmark.contenders import Contender

__all__ = ["run_benchmark"]

logger = logging.getLogger(__name__)


def run_benchmark(
    contenders: list[Contender],
    split_config: dict,
    eval_config: dict | None = None,
    output_dir: str = ".",
) -> dict[str, dict[str, float | str | dict[str, float]]]:
    """Train + predict + score each contender through the ship-gate eval matrix.

    Parameters
    ----------
    contenders : list[Contender]
        The contender instances to evaluate. Each is trained and scored
        independently.
    split_config : dict
        Must contain ``"train_slices"`` (list[str]), ``"test_slices"``
        (list[str]), and ``"gt_masks"`` (list[NDArray] — ground-truth boolean
        masks aligned with ``test_slices``). May also carry ``"patch_size"``
        and ``"ddp"`` which are threaded into ``train_and_predict``.
    eval_config : dict | None
        Optional overrides for the eval-metric kwargs: ``"voxel_size_um"``,
        ``"capillary_radius_um"``, ``"boundary_patch_size"``. ``None`` uses
        the metric defaults.
    output_dir : str
        The directory each contender trains into (checkpoint location).

    Returns
    -------
    dict[str, dict[str, float | str | dict[str, float]]]
        Per-contender metric table keyed by contender ``name``. Each row
        contains the 6 ship-gate metrics + ``reported_dice``. Scalar metrics
        are floats; dict-valued metrics (``caliber_stratified_recall``,
        ``boundary_artifact_regression``) are dicts. A contender whose every
        slice is vessel-free gets ``"vessel-free slice — metric undefined"``
        for that metric.

    Raises
    ------
    ImportError
        If torch (the ``[ai]`` extra) is not installed.
    ValueError
        If ``split_config`` is missing required keys, or the number of GT
        masks does not match the number of test slices.
    """
    try:
        import torch  # ruff: ignore[unused-import] — gates the orchestrator on the [ai] extra
    except ImportError as e:
        raise ImportError("Please install liom-toolkit[ai] to run the benchmark.") from e

    import numpy as np

    from liom_toolkit.segmentation.vseg.eval_metrics import (
        boundary_artifact_regression,
        caliber_stratified_recall,
        centerline_recall,
        cl_dice_metric,
        fpr_on_empty,
        reported_dice,
        spurious_thin_vessel_rate,
    )

    for key in ("train_slices", "test_slices", "gt_masks"):
        if key not in split_config:
            raise ValueError(
                f"split_config must contain {key!r} — got keys {sorted(split_config.keys())}"
            )

    train_slices = split_config["train_slices"]
    test_slices = split_config["test_slices"]
    gt_masks = split_config["gt_masks"]
    patch_size = split_config.get("patch_size", (1, 256, 256))
    ddp = split_config.get("ddp", False)

    if len(gt_masks) != len(test_slices):
        raise ValueError(
            f"gt_masks length ({len(gt_masks)}) must match test_slices length "
            f"({len(test_slices)}) — one GT mask per test slice"
        )

    eval_cfg = eval_config or {}
    voxel_size_um = eval_cfg.get("voxel_size_um", 6.5)
    capillary_radius_um = eval_cfg.get("capillary_radius_um", 5.0)
    boundary_patch_size = eval_cfg.get("boundary_patch_size", (256, 256))

    # (metric_name, callable) pairs — the ship-gate matrix + reported_dice.
    scalar_metrics: list[tuple[str, Callable[..., float]]] = [
        ("centerline_recall", centerline_recall),
        ("spurious_thin_vessel_rate", spurious_thin_vessel_rate),
        ("fpr_on_empty", fpr_on_empty),
        ("cl_dice_metric", cl_dice_metric),
        ("reported_dice", reported_dice),
    ]
    dict_metrics: list[tuple[str, Callable[..., dict[str, float]]]] = [
        ("caliber_stratified_recall", caliber_stratified_recall),
        ("boundary_artifact_regression", boundary_artifact_regression),
    ]

    results: dict[str, dict[str, float | str | dict[str, float]]] = {}
    for contender in contenders:
        logger.info("Running contender: %s", contender.name)
        # No silent swallow — a contender that raises propagates (AGENTS §2).
        predictions = contender.train_and_predict(
            train_slices, test_slices, output_dir, patch_size=patch_size, ddp=ddp
        )

        # Binarize defensively: contenders return bool, but a 0/255 uint8 array
        # passed to cl_score would scale tprec/tsens by 255 → wrong values.
        pred_masks = [np.asarray(pred, dtype=bool) for pred in predictions]

        per_slice_rows: list[dict[str, float | str | dict[str, float]]] = []
        for pred, gt in zip(pred_masks, gt_masks, strict=True):
            gt_bool = np.asarray(gt, dtype=bool)
            row: dict[str, float | str | dict[str, float]] = {}

            for name, fn in scalar_metrics:
                try:
                    row[name] = fn(pred, gt_bool)
                except ValueError:
                    # Vessel-free slice — metric undefined. Record as a row,
                    # NOT a NaN (no silent NaN escape into the result table).
                    row[name] = "vessel-free slice — metric undefined"

            for name, fn in dict_metrics:
                kwargs = (
                    {"voxel_size_um": voxel_size_um, "capillary_radius_um": capillary_radius_um}
                    if name == "caliber_stratified_recall"
                    else {"patch_size": boundary_patch_size}
                )
                try:
                    row[name] = fn(pred, gt_bool, **kwargs)
                except ValueError:
                    row[name] = "vessel-free slice — metric undefined"

            per_slice_rows.append(row)

        # Aggregate per-contender: mean over slices where the metric is defined.
        agg: dict[str, float | str | dict[str, float]] = {}
        for name, _ in scalar_metrics:
            values = [r[name] for r in per_slice_rows if isinstance(r[name], (int, float))]
            agg[name] = float(np.mean(values)) if values else "vessel-free slice — metric undefined"
        for name, _ in dict_metrics:
            dicts = [r[name] for r in per_slice_rows if isinstance(r[name], dict)]
            if dicts:
                keys = dicts[0].keys()
                agg[name] = {k: float(np.mean([d[k] for d in dicts])) for k in keys}
            else:
                agg[name] = "vessel-free slice — metric undefined"

        results[contender.name] = agg

    return results
