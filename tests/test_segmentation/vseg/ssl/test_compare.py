"""TDD tests for the pretrained-vs-from-scratch comparison harness (ssl.compare).

The comparison harness (``run_comparison``) is the orchestrator that judges
whether self-supervised pretraining improves vessel segmentation over the
from-scratch baseline. It mirrors ``benchmark/run.py`` but scores TWO
contender result sets (pretrained-init + from-scratch) through the EXISTING
``eval_metrics.py`` matrix and records the per-metric delta plus the
capillary-recall effect-size -- NOT a p-value (n=2 brains makes p-values
uninterpretable; effect-size thresholding is the replicable bar).

Behaviors asserted:

* The per-metric matrix is computed for BOTH contenders via the REUSED
  ``eval_metrics`` functions, and the delta (pretrained - from_scratch) is
  recorded per metric.
* The capillary-recall effect-size is recorded (a ``capillary_recall`` delta
  field), NOT a p-value (no ``p_value`` / ``pvalue`` / ``scipy.stats`` in the
  output).
* The per-volume split is enforced via ``per_volume_split``: a
  ``patch_level=True`` config raises ``ValueError``; brain overlap (a brain
  in both train and test) raises ``ValueError``.
* A vessel-free slice records ``"vessel-free slice -- metric undefined"`` in
  the per-slice row, NOT a NaN (no silent NaN escape into the result table).
* Compute is equalized: ``iterations_per_epoch`` is threaded identically into
  both contender records (the same gradient-step budget for both runs).

All tests run on CPU and are gated on the ``[ai]`` extra (torch) via
``pytest.importorskip`` at the first line of each test body and the
``@pytest.mark.ai`` marker.
"""

from __future__ import annotations

import numpy as np
import pytest


def _vessel_slice() -> np.ndarray:
    """A 64x64 GT mask with a horizontal vessel (non-empty skeleton)."""
    gt = np.zeros((64, 64), dtype=bool)
    gt[28:36, 16:48] = True
    return gt


def _empty_slice() -> np.ndarray:
    """An all-zero 64x64 mask (vessel-free slice -- metric undefined)."""
    return np.zeros((64, 64), dtype=bool)


@pytest.mark.ai
def test_run_comparison_computes_per_metric_matrix_and_delta_for_both_contenders() -> None:
    """run_comparison scores both contenders through eval_metrics and records the delta.

    Two fake contender result sets (pretrained-init vs from-scratch) are
    scored through the REUSED eval_metrics matrix; the output contains the
    per-metric matrix for BOTH contenders and the per-metric delta
    (pretrained - from_scratch). The metrics are computed by the existing
    eval_metrics functions, NOT reimplemented in compare.py.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.compare import run_comparison

    gt = _vessel_slice()
    # Perfect prediction for pretrained; slightly worse for from-scratch.
    pretrained_pred = gt.copy()
    from_scratch_pred = np.zeros((64, 64), dtype=bool)
    from_scratch_pred[28:36, 16:40] = True  # misses the right end

    brain_paths = {"S23": ["s23_0.png"], "S24": ["s24_0.png"]}

    result = run_comparison(
        brain_paths=brain_paths,
        train_brains=["S23"],
        test_brains=["S24"],
        pretrained_predictions=[pretrained_pred],
        from_scratch_predictions=[from_scratch_pred],
        gt_masks=[gt],
        iterations_per_epoch=250,
    )

    # Both contender matrices are present.
    assert "pretrained" in result, "result must contain the pretrained-init matrix"
    assert "from_scratch" in result, "result must contain the from-scratch matrix"
    expected_metrics = {
        "centerline_recall",
        "caliber_stratified_recall",
        "boundary_artifact_regression",
        "spurious_thin_vessel_rate",
        "fpr_on_empty",
        "cl_dice_metric",
        "reported_dice",
    }
    assert expected_metrics.issubset(result["pretrained"].keys()), (
        f"pretrained matrix missing metrics: {expected_metrics - set(result['pretrained'].keys())}"
    )
    assert expected_metrics.issubset(result["from_scratch"].keys()), (
        "from_scratch matrix missing metrics: "
        f"{expected_metrics - set(result['from_scratch'].keys())}"
    )
    # The delta (pretrained - from_scratch) is recorded per metric.
    assert "delta" in result, "result must contain the per-metric delta table"
    assert expected_metrics.issubset(result["delta"].keys()), (
        f"delta table missing metrics: {expected_metrics - set(result['delta'].keys())}"
    )
    # Perfect prediction -> centerline_recall 1.0; from_scratch misses -> < 1.0.
    assert result["pretrained"]["centerline_recall"] == pytest.approx(1.0, abs=1e-6)
    assert result["from_scratch"]["centerline_recall"] < 1.0
    # Delta = pretrained - from_scratch > 0 for centerline_recall here.
    assert result["delta"]["centerline_recall"] > 0


@pytest.mark.ai
def test_run_comparison_records_capillary_recall_effect_size_not_p_value() -> None:
    """run_comparison records the capillary-recall effect-size, NOT a p-value.

    The capillary-recall delta is the targeted effect-size the ship decision
    judges (gate iii). With n=2 brains, p-values are uninterpretable
    (Varoquaux 2017); the harness records the delta + effect-size, never a
    p-value. The output contains a ``capillary_recall`` delta field and NO
    ``p_value`` / ``pvalue`` field.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.compare import run_comparison

    gt = _vessel_slice()
    pretrained_pred = gt.copy()
    from_scratch_pred = gt.copy()

    brain_paths = {"S23": ["s23_0.png"], "S24": ["s24_0.png"]}

    result = run_comparison(
        brain_paths=brain_paths,
        train_brains=["S23"],
        test_brains=["S24"],
        pretrained_predictions=[pretrained_pred],
        from_scratch_predictions=[from_scratch_pred],
        gt_masks=[gt],
        iterations_per_epoch=250,
    )

    # The capillary-recall effect-size is recorded as a delta.
    assert "capillary_recall" in result["delta"], (
        "delta must contain the capillary_recall effect-size (gate iii)"
    )
    # The capillary-recall effect-size is surfaced as a top-level field too
    # (the targeted effect-size the ship decision judges).
    assert "capillary_recall_effect_size" in result, (
        "result must surface the capillary_recall effect-size as a top-level field"
    )
    # NO p-value anywhere in the output (effect-size thresholding, not p-values).
    flat_keys = set(result.keys())
    assert "p_value" not in flat_keys, "result must NOT contain a p_value field"
    assert "pvalue" not in flat_keys, "result must NOT contain a pvalue field"


@pytest.mark.ai
def test_run_comparison_rejects_patch_level_split() -> None:
    """run_comparison raises ValueError when patch_level=True is passed.

    Patch-level i.i.d. splitting leaks vascular structure across train/test
    and inflates Dice 10-20+ points; the harness enforces per_volume_split
    which rejects the patch-level config explicitly.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.compare import run_comparison

    gt = _vessel_slice()
    brain_paths = {"S23": ["s23_0.png"], "S24": ["s24_0.png"]}

    with pytest.raises(ValueError, match=r"patch-level i\.i\.d\. split is rejected"):
        run_comparison(
            brain_paths=brain_paths,
            train_brains=["S23"],
            test_brains=["S24"],
            pretrained_predictions=[gt.copy()],
            from_scratch_predictions=[gt.copy()],
            gt_masks=[gt],
            iterations_per_epoch=250,
            patch_level=True,
        )


@pytest.mark.ai
def test_run_comparison_rejects_brain_overlap() -> None:
    """run_comparison raises ValueError when a brain is in both train and test.

    A brain in both splits leaks vascular structure across train/test -- the
    silent-wrong-data failure mode per_volume_split exists to prevent.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.compare import run_comparison

    gt = _vessel_slice()
    brain_paths = {"S23": ["s23_0.png"], "S24": ["s24_0.png"]}

    with pytest.raises(ValueError, match="appears in both train and test"):
        run_comparison(
            brain_paths=brain_paths,
            train_brains=["S23"],
            test_brains=["S23", "S24"],
            pretrained_predictions=[gt.copy()],
            from_scratch_predictions=[gt.copy()],
            gt_masks=[gt],
            iterations_per_epoch=250,
        )


@pytest.mark.ai
def test_run_comparison_records_vessel_free_slice_as_undefined_not_nan() -> None:
    """A vessel-free slice records 'vessel-free slice -- metric undefined', NOT a NaN.

    The no-silent-NaN contract (AGENTS section 2): a metric undefined on an
    empty slice is recorded as an explicit string row, never as a NaN that
    could escape into the result table and propagate as a plausible-looking
    but wrong value.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.compare import run_comparison

    empty = _empty_slice()
    # A prediction on a vessel-free slice (all-zero GT). The eval metrics
    # raise ValueError on empty GT; the harness records the undefined row.
    pred = np.zeros((64, 64), dtype=bool)

    brain_paths = {"S23": ["s23_0.png"], "S24": ["s24_0.png"]}

    result = run_comparison(
        brain_paths=brain_paths,
        train_brains=["S23"],
        test_brains=["S24"],
        pretrained_predictions=[pred],
        from_scratch_predictions=[pred],
        gt_masks=[empty],
        iterations_per_epoch=250,
    )

    # The per-slice rows record the undefined marker (NOT a NaN). The
    # aggregated matrix records the marker when every slice is vessel-free.
    assert "per_slice" in result, "result must expose the per-slice rows"
    assert "pretrained" in result["per_slice"], "per_slice must contain pretrained rows"
    pretrained_rows = result["per_slice"]["pretrained"]
    assert len(pretrained_rows) == 1
    row = pretrained_rows[0]
    # At least one metric is undefined on the empty slice; it must be the
    # explicit string marker, NOT a NaN.
    undefined_marker = "vessel-free slice -- metric undefined"
    undefined_metrics = [v for v in row.values() if isinstance(v, str)]
    assert undefined_metrics, (
        "expected at least one 'vessel-free slice -- metric undefined' row on the empty slice"
    )
    assert all(v == undefined_marker for v in undefined_metrics), (
        f"undefined rows must be the exact marker {undefined_marker!r}, got {undefined_metrics}"
    )
    # No NaN anywhere in the per-slice row.
    for v in row.values():
        if isinstance(v, float):
            assert not np.isnan(v), "no NaN may escape into the per-slice row"


@pytest.mark.ai
def test_run_comparison_equalizes_compute_between_contenders() -> None:
    """iterations_per_epoch is threaded identically into both contender records.

    The Phase-14 fairness lesson: MONAI was crippled by 4x fewer patches +
    30 vs 250 iterations/epoch. The harness threads the same
    iterations_per_epoch into both the pretrained-init and from-scratch runs
    so the comparison is not confounded by a compute-budget mismatch. The
    output records the iterations_per_epoch used for BOTH contenders.
    """
    pytest.importorskip("torch")
    from liom_toolkit.segmentation.vseg.ssl.compare import run_comparison

    gt = _vessel_slice()
    brain_paths = {"S23": ["s23_0.png"], "S24": ["s24_0.png"]}

    result = run_comparison(
        brain_paths=brain_paths,
        train_brains=["S23"],
        test_brains=["S24"],
        pretrained_predictions=[gt.copy()],
        from_scratch_predictions=[gt.copy()],
        gt_masks=[gt],
        iterations_per_epoch=250,
    )

    # The compute budget is recorded and equal for both contenders.
    assert "compute_budget" in result, "result must record the compute budget"
    budget = result["compute_budget"]
    assert budget["iterations_per_epoch"] == 250
    assert budget["pretrained"] == budget["from_scratch"], (
        "iterations_per_epoch must be identical for both contenders (equalized compute)"
    )
