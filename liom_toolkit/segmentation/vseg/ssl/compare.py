"""Pretrained-vs-from-scratch comparison harness for the SSL warm-start evaluation.

``run_comparison`` is the orchestrator that judges whether self-supervised
pretraining improves vessel segmentation over the from-scratch baseline. It
mirrors :func:`liom_toolkit.segmentation.vseg.benchmark.run.run_benchmark`
but scores TWO contender result sets (pretrained-init + from-scratch) through
the EXISTING :mod:`liom_toolkit.segmentation.vseg.eval_metrics` matrix and
records the per-metric delta plus the capillary-recall effect-size.

Design contract (the locked comparison protocol):

* **REUSED eval code** -- the harness imports the ship-gate metrics from
  ``eval_metrics`` (centerline_recall, caliber_stratified_recall,
  boundary_artifact_regression, spurious_thin_vessel_rate, fpr_on_empty,
  cl_dice_metric, reported_dice) and NEVER reimplements them. The
  warm-started model is evaluated by THESE implementations against the SAME
  gate as the from-scratch baseline.
* **Per-volume split** -- the harness enforces
  :func:`liom_toolkit.segmentation.vseg.benchmark.split.per_volume_split`
  (S23 train / S24 held-out test). Patch-level i.i.d. splitting leaks
  vascular structure across train/test and inflates Dice 10-20+ points;
  ``per_volume_split`` raises ``ValueError`` on a patch-level config and on
  brain overlap.
* **Equalized compute** -- the same ``iterations_per_epoch`` is threaded
  into both contender records so the comparison is not confounded by a
  compute-budget mismatch (the Phase-14 lesson: MONAI was crippled by 4x
  fewer patches + 30 vs 250 iterations/epoch).
* **Effect-size, NOT p-values** -- with n=2 brains, p-values are
  uninterpretable (Varoquaux 2017). The harness records the per-metric
  delta and the capillary-recall effect-size (the targeted gate-iii
  requirement), never a p-value.
* **No silent NaN** -- a vessel-free slice where a metric is undefined is
  recorded as ``"vessel-free slice -- metric undefined"`` in the per-slice
  row, never as a NaN (AGENTS section 2 -- no silent wrong-data fallback).

Validation uses ``if ...: raise ValueError(...)`` with the offending value
in the message (AGENTS section 2 -- never ``assert`` for validation).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

__all__ = ["run_comparison"]

logger = logging.getLogger(__name__)


def run_comparison(
    brain_paths: dict[str, list[str]],
    train_brains: list[str],
    test_brains: list[str],
    pretrained_predictions: list[NDArray],
    from_scratch_predictions: list[NDArray],
    gt_masks: list[NDArray],
    *,
    iterations_per_epoch: int,
    patch_level: bool = False,
    eval_config: dict | None = None,
) -> dict[str, object]:
    """Score pretrained-init vs from-scratch through the REUSED eval_metrics gate.

    Parameters
    ----------
    brain_paths : dict[str, list[str]]
        Mapping from brain identifier to the list of slice paths belonging
        to that brain (the brain-keyed structure ``per_volume_split``
        requires). A flat patch list is the patch-level i.i.d. config and
        is rejected.
    train_brains : list[str]
        Brain identifiers whose slices form the train partition (e.g.
        ``["S23"]``).
    test_brains : list[str]
        Brain identifiers whose slices form the held-out test partition
        (e.g. ``["S24"]``).
    pretrained_predictions : list[NDArray]
        Predicted vessel masks from the pretrained-init model, one per test
        slice, aligned with ``gt_masks``.
    from_scratch_predictions : list[NDArray]
        Predicted vessel masks from the from-scratch baseline model, one
        per test slice, aligned with ``gt_masks``.
    gt_masks : list[NDArray]
        Ground-truth boolean masks for the test slices, aligned with both
        prediction lists.
    iterations_per_epoch : int
        The gradient-step budget threaded IDENTICALLY into both contender
        runs (equalized compute -- the same budget the from-scratch
        baseline used).
    patch_level : bool
        Must be ``False``. Passing ``True`` selects the patch-level i.i.d.
        config and is rejected by ``per_volume_split``.
    eval_config : dict | None
        Optional overrides for the eval-metric kwargs:
        ``"voxel_size_um"``, ``"capillary_radius_um"``,
        ``"boundary_patch_size"``. ``None`` uses the metric defaults.

    Returns
    -------
    dict[str, object]
        A result dict with keys:

        * ``"pretrained"`` -- per-metric aggregated matrix for the
          pretrained-init model.
        * ``"from_scratch"`` -- per-metric aggregated matrix for the
          from-scratch baseline.
        * ``"delta"`` -- per-metric delta (pretrained - from_scratch);
          includes ``"capillary_recall"`` (the gate-iii effect-size).
        * ``"capillary_recall_effect_size"`` -- the targeted capillary-recall
          effect-size (pretrained capillary_recall - from_scratch
          capillary_recall), surfaced for the ship decision.
        * ``"per_slice"`` -- per-slice rows for each contender (the
          ``"vessel-free slice -- metric undefined"`` rows live here).
        * ``"compute_budget"`` -- the equalized compute record
          (``iterations_per_epoch`` for both contenders).

    Raises
    ------
    ImportError
        If torch (the ``[ai]`` extra) is not installed.
    ValueError
        If ``per_volume_split`` rejects the split (patch-level config or
        brain overlap), if the prediction/GT lists are mismatched in
        length, or if ``iterations_per_epoch`` is not positive.
    """
    # Gate the orchestrator on the [ai] extra (mirrors run_benchmark). The
    # eval_metrics imports are REUSED UNCHANGED -- no new eval code.
    try:
        import torch  # ruff: ignore[unused-import] -- gates the orchestrator on the [ai] extra
    except ImportError as e:
        raise ImportError(
            "Please install liom-toolkit[ai] to run the SSL comparison harness."
        ) from e

    import numpy as np

    from liom_toolkit.segmentation.vseg.benchmark.split import per_volume_split
    from liom_toolkit.segmentation.vseg.eval_metrics import (
        boundary_artifact_regression,
        caliber_stratified_recall,
        centerline_recall,
        cl_dice_metric,
        fpr_on_empty,
        reported_dice,
        spurious_thin_vessel_rate,
    )

    # Equalized compute is a positive integer gradient-step budget. A
    # non-positive budget would silently run zero-step training and present
    # a randomly-initialized model as a trained contender (the silent-wrong-
    # data path AGENTS section 2 forbids).
    if iterations_per_epoch <= 0:
        raise ValueError(
            f"iterations_per_epoch must be positive (got {iterations_per_epoch}) "
            "-- a non-positive budget would run zero-step training"
        )

    # Enforce the per-volume split BEFORE scoring. per_volume_split raises
    # ValueError on patch_level=True, on a flat patch list, and on brain
    # overlap -- the locked Phase-14 fairness discipline.
    per_volume_split(
        brain_paths,
        train_brains=train_brains,
        test_brains=test_brains,
        patch_level=patch_level,
    )

    # Length alignment -- one GT mask + one prediction per test slice for
    # each contender. A mismatch would zip-silently and score the wrong
    # prediction against the wrong GT.
    n_test = len(gt_masks)
    if len(pretrained_predictions) != n_test:
        raise ValueError(
            f"pretrained_predictions length ({len(pretrained_predictions)}) must "
            f"match gt_masks length ({n_test}) -- one prediction per test slice"
        )
    if len(from_scratch_predictions) != n_test:
        raise ValueError(
            f"from_scratch_predictions length ({len(from_scratch_predictions)}) must "
            f"match gt_masks length ({n_test}) -- one prediction per test slice"
        )

    eval_cfg = eval_config or {}
    voxel_size_um = eval_cfg.get("voxel_size_um", 6.5)
    capillary_radius_um = eval_cfg.get("capillary_radius_um", 5.0)
    # Default the boundary patch size from a 256x256 grid (the Phase-14
    # patch size); an explicit override wins. A hardcoded default that does
    # not fit the image would silently produce ny=0/nx=0 and a misleading
    # "vessel-free slice -- metric undefined" row, masking the boundary
    # metric.
    boundary_patch_size = eval_cfg.get("boundary_patch_size", (256, 256))

    # (metric_name, callable) pairs -- the ship-gate matrix + reported_dice.
    # These are the REUSED eval_metrics functions; the harness does NOT
    # reimplement any of them.
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

    def _score_contender(
        predictions: list[NDArray],
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        """Score one contender's predictions through the eval_metrics matrix.

        Returns the aggregated per-metric matrix and the per-slice rows.
        Per-slice ``ValueError`` from the eval metrics (vessel-free slices
        where a metric is undefined) is caught and recorded as
        ``"vessel-free slice -- metric undefined"`` -- never a NaN.

        Returns
        -------
        tuple[dict[str, object], list[dict[str, object]]]
            ``(aggregated_matrix, per_slice_rows)``. The aggregated matrix
            is mean-over-defined-slices per metric; the per-slice rows
            carry the ``"vessel-free slice -- metric undefined"`` markers.
        """
        pred_masks = [np.asarray(pred, dtype=bool) for pred in predictions]
        per_slice_rows: list[dict[str, object]] = []
        for pred, gt in zip(pred_masks, gt_masks, strict=True):
            gt_bool = np.asarray(gt, dtype=bool)
            row: dict[str, object] = {}
            for name, fn in scalar_metrics:
                try:
                    row[name] = fn(pred, gt_bool)
                except ValueError:
                    # Vessel-free slice -- metric undefined. Record as a
                    # row, NOT a NaN (no silent NaN escape into the table).
                    row[name] = "vessel-free slice -- metric undefined"
            for name, fn in dict_metrics:
                kwargs = (
                    {
                        "voxel_size_um": voxel_size_um,
                        "capillary_radius_um": capillary_radius_um,
                    }
                    if name == "caliber_stratified_recall"
                    else {"patch_size": boundary_patch_size}
                )
                try:
                    row[name] = fn(pred, gt_bool, **kwargs)
                except ValueError:
                    row[name] = "vessel-free slice -- metric undefined"
            per_slice_rows.append(row)

        # Aggregate: mean over slices where the metric is defined.
        agg: dict[str, object] = {}
        for name, _ in scalar_metrics:
            values = [r[name] for r in per_slice_rows if isinstance(r[name], (int, float))]
            agg[name] = (
                float(np.mean(values)) if values else "vessel-free slice -- metric undefined"
            )
        for name, _ in dict_metrics:
            dicts = [r[name] for r in per_slice_rows if isinstance(r[name], dict)]
            if dicts:
                # Guard against a per-slice dict missing a key that
                # dicts[0] has -- mean over the slices where the key is
                # present (skip slices missing it), so a future metric
                # variant that conditionally omits a key does not raise
                # KeyError mid-aggregation.
                keys = dicts[0].keys()
                agg[name] = {k: float(np.mean([d[k] for d in dicts if k in d])) for k in keys}
            else:
                agg[name] = "vessel-free slice -- metric undefined"
        return agg, per_slice_rows

    pretrained_agg, pretrained_per_slice = _score_contender(pretrained_predictions)
    from_scratch_agg, from_scratch_per_slice = _score_contender(from_scratch_predictions)

    # Per-metric delta (pretrained - from_scratch). Scalar metrics subtract
    # directly; the capillary_recall sub-field of caliber_stratified_recall
    # is the gate-iii effect-size. Dict metrics (caliber_stratified_recall,
    # boundary_artifact_regression) produce a per-key delta dict.
    delta: dict[str, object] = {}
    for name, _ in scalar_metrics:
        p_val = pretrained_agg[name]
        f_val = from_scratch_agg[name]
        if isinstance(p_val, (int, float)) and isinstance(f_val, (int, float)):
            delta[name] = float(p_val - f_val)
        else:
            # One or both undefined on every slice -- the delta is undefined.
            delta[name] = "vessel-free slice -- metric undefined"
    for name, _ in dict_metrics:
        p_dict = pretrained_agg[name]
        f_dict = from_scratch_agg[name]
        if isinstance(p_dict, dict) and isinstance(f_dict, dict):
            delta[name] = {k: float(p_dict[k] - f_dict[k]) for k in p_dict if k in f_dict}
        else:
            delta[name] = "vessel-free slice -- metric undefined"

    # Surface the capillary-recall effect-size as a top-level delta key AND
    # a top-level result field -- this is the targeted gate-iii requirement
    # the ship decision judges. It is an effect-size (delta), NOT a p-value
    # (n=2 makes p-values uninterpretable; Varoquaux 2017). Extracting it
    # from the caliber_stratified_recall dict makes the ship-decision check
    # a single-key lookup rather than a nested-dict traversal.
    capillary_effect_size: object
    if isinstance(delta.get("caliber_stratified_recall"), dict):
        capillary_effect_size = delta["caliber_stratified_recall"].get(
            "capillary_recall", "vessel-free slice -- metric undefined"
        )
    else:
        capillary_effect_size = "vessel-free slice -- metric undefined"
    delta["capillary_recall"] = capillary_effect_size

    return {
        "pretrained": pretrained_agg,
        "from_scratch": from_scratch_agg,
        "delta": delta,
        "capillary_recall_effect_size": capillary_effect_size,
        "per_slice": {
            "pretrained": pretrained_per_slice,
            "from_scratch": from_scratch_per_slice,
        },
        "compute_budget": {
            "iterations_per_epoch": iterations_per_epoch,
            "pretrained": iterations_per_epoch,
            "from_scratch": iterations_per_epoch,
        },
    }
