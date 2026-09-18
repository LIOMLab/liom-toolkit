"""Known-answer tests for the direction-aware ship-gate adjudication harness.

The ship gate is a per-metric matrix whose rows have DIFFERENT comparison
directions (recall/clDice are >= floors, rate rows are <= ceilings, reported
rows never gate). If a direction is inverted -- e.g. an FPR ceiling evaluated
as a floor -- the gate passes failing models silently, which is
silent-wrong-data on the decision instrument itself. These tests pin the
pre-registered thresholds, the per-row directions, the both-folds rule, the
fail-closed undefined convention, and every ship_decision branch.

All tests are CPU-only and require NO torch -- the gate scores saved masks
through ``eval_metrics`` (scipy/skimage, transitively present via the [seg]
extra), so no ``ai`` marker and no ``importorskip`` is needed.

Coverage:

* Direction encoding -- the inverted-FPR canary (0.9 against a <= ceiling
  fails; an inverted gate would pass it), recall floor semantics, and the
  regression_delta sign convention (positive delta = seam artifact).
* Reported-only isolation -- capillary_recall and reported_dice values never
  influence ``FoldVerdict.passed``.
* Both-folds rule -- ``arm_passes`` requires EVERY fold verdict to pass.
* Fail-closed undefined -- a gating metric measured ``"undefined"`` fails the
  row with the reason named; a reported row ``"undefined"`` is unaffected.
* ``ship_decision`` -- baseline-pass+tie, baseline-pass+real-improvement,
  baseline-fail+alternative-pass, and all-fail NO-SHIP branches.
* ``score_prediction_set`` -- returns all seven metric keys with sub-keyed
  rows flattened, and yields ``"undefined"`` for the raising metrics when
  every slice of a fold is GT-empty.
"""

from __future__ import annotations

import numpy as np
import pytest


def _passing_metrics() -> dict[str, object]:
    """Return a metrics dict that passes every gating row of the gate.

    Values are chosen strictly inside every pre-registered bound so a single
    row can be perturbed per test without tripping a sibling row.
    """
    return {
        "centerline_recall": 0.95,
        "caliber_stratified_recall": {
            "large_vessel_recall": 0.95,
            "capillary_recall": 0.5,
        },
        "boundary_artifact_regression": {
            "boundary_quality": 0.9,
            "interior_quality": 0.91,
            "regression_delta": 0.0,
        },
        "spurious_thin_vessel_rate": 0.0,
        "fpr_on_empty": 0.005,
        "cl_dice_metric": 0.95,
        "reported_dice": 0.9,
    }


def _row(verdict, key):
    """Return the RowVerdict for ``key`` from a FoldVerdict."""
    matches = [r for r in verdict.rows if r.row.key == key]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one row for {key}, got {len(matches)}")
    return matches[0]


# ---------------------------------------------------------------------------
# GATE_ROWS — the pre-registered table encoded verbatim
# ---------------------------------------------------------------------------


def test_gate_rows_encodes_preregistered_table():
    """GATE_ROWS carries exactly the 8 pre-registered rows, 6 of them gating.

    The numeric thresholds must equal the pre-registered gate table verbatim
    — the commit order is the audit trail, so any drift between the document
    and the code is post-hoc tuning.
    """
    from liom_toolkit.segmentation.vseg.gate import GATE_ROWS

    assert len(GATE_ROWS) == 8
    by_key = {r.key: r for r in GATE_ROWS}

    centerline = by_key["centerline_recall"]
    assert (centerline.direction, centerline.gating) == ("ge", True)
    assert centerline.bound == 0.90
    assert centerline.legacy_ref == 0.898 and centerline.legacy_margin == 0.01
    assert centerline.nnunet_ref == 0.927 and centerline.nnunet_tolerance == 0.02

    large = by_key["caliber_stratified_recall.large_vessel_recall"]
    assert (large.direction, large.gating) == ("ge", True)
    assert large.bound == 0.90
    assert large.legacy_ref == 0.898 and large.legacy_margin == 0.01
    assert large.nnunet_ref == 0.927 and large.nnunet_tolerance == 0.02

    capillary = by_key["caliber_stratified_recall.capillary_recall"]
    assert (capillary.direction, capillary.gating) == ("reported", False)

    delta = by_key["boundary_artifact_regression.regression_delta"]
    assert (delta.direction, delta.gating) == ("le", True)
    assert delta.bound == 0.02

    spurious = by_key["spurious_thin_vessel_rate"]
    assert (spurious.direction, spurious.gating) == ("le", True)
    assert spurious.bound == 0.005
    assert spurious.nnunet_ref == 0.0 and spurious.nnunet_tolerance == 0.005

    fpr = by_key["fpr_on_empty"]
    assert (fpr.direction, fpr.gating) == ("le", True)
    assert fpr.bound == 0.02
    assert fpr.legacy_ref == 0.013 and fpr.legacy_margin == 0.01
    assert fpr.nnunet_ref == 0.009 and fpr.nnunet_tolerance == 0.01

    cldice = by_key["cl_dice_metric"]
    assert (cldice.direction, cldice.gating) == ("ge", True)
    assert cldice.bound == 0.90
    assert cldice.legacy_ref == 0.903 and cldice.legacy_margin == 0.01
    assert cldice.nnunet_ref == 0.924 and cldice.nnunet_tolerance == 0.02

    reported = by_key["reported_dice"]
    assert (reported.direction, reported.gating) == ("reported", False)

    assert sum(1 for r in GATE_ROWS if r.gating) == 6


# ---------------------------------------------------------------------------
# (a) Direction encoding — the inverted-failure canary
# ---------------------------------------------------------------------------


def test_fpr_ceiling_rejects_inverted_measurement():
    """fpr_on_empty=0.9 FAILS the <= ceiling — the inverted-failure canary.

    If the comparison direction were implemented as a floor (>=), a 0.9 FPR
    would pass. This test proves the ceiling direction is wired correctly.
    """
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate

    metrics = _passing_metrics()
    metrics["fpr_on_empty"] = 0.9
    verdict = evaluate_gate(metrics, fold=0)

    assert _row(verdict, "fpr_on_empty").passed is False
    assert verdict.passed is False


def test_fpr_within_effective_ceiling_passes():
    """fpr_on_empty=0.005 passes all three le conjuncts (0.02 / 0.023 / 0.019)."""
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate

    verdict = evaluate_gate(_passing_metrics(), fold=0)

    assert _row(verdict, "fpr_on_empty").passed is True
    assert verdict.passed is True


def test_fpr_above_nnunet_tolerance_fails_effective_ceiling():
    """fpr_on_empty=0.0195 fails — the effective ceiling is 0.019, not 0.02.

    The bound conjunct (0.02) and the legacy conjunct (0.023) would pass, but
    the nnUNet-anchored conjunct (0.009 + 0.01 = 0.019) is tighter — proving
    conjuncts are ANDed, not ORed.
    """
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate

    metrics = _passing_metrics()
    metrics["fpr_on_empty"] = 0.0195
    verdict = evaluate_gate(metrics, fold=0)

    assert _row(verdict, "fpr_on_empty").passed is False
    assert verdict.passed is False


def test_centerline_recall_floor_direction():
    """centerline_recall=0.85 fails the 0.90 F floor; 0.95 passes."""
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate

    metrics = _passing_metrics()
    metrics["centerline_recall"] = 0.85
    assert evaluate_gate(metrics, fold=0).passed is False

    metrics = _passing_metrics()
    metrics["centerline_recall"] = 0.95
    assert evaluate_gate(metrics, fold=0).passed is True


def test_regression_delta_sign_convention():
    """regression_delta uses a <= ceiling: +0.05 fails; -0.3 and +0.01 pass.

    regression_delta = interior - boundary, so a POSITIVE delta is the seam
    artifact — the gate is `measured <= 0.02`, not `>= -delta`.
    """
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate

    metrics = _passing_metrics()
    metrics["boundary_artifact_regression"]["regression_delta"] = 0.05
    assert (
        _row(
            evaluate_gate(metrics, fold=0),
            "boundary_artifact_regression.regression_delta",
        ).passed
        is False
    )

    for ok_delta in (-0.3, 0.01):
        metrics = _passing_metrics()
        metrics["boundary_artifact_regression"]["regression_delta"] = ok_delta
        verdict = evaluate_gate(metrics, fold=0)
        assert _row(verdict, "boundary_artifact_regression.regression_delta").passed is True
        assert verdict.passed is True


# ---------------------------------------------------------------------------
# (b) Reported-only isolation — capillary + reported_dice never gate
# ---------------------------------------------------------------------------


def test_reported_rows_never_gate():
    """capillary=0.0 and reported_dice=0.0 cannot flip FoldVerdict.passed.

    At 6.5 µm the capillary bin sits below the ~2-voxel Nyquist floor — every
    contender measures 0.0, so the row is reported-only. A model that wins
    every resolvable row must not be blocked by a physically impossible one.
    """
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate

    metrics = _passing_metrics()
    metrics["caliber_stratified_recall"]["capillary_recall"] = 0.0
    metrics["reported_dice"] = 0.0
    verdict = evaluate_gate(metrics, fold=0)

    assert verdict.passed is True
    assert _row(verdict, "caliber_stratified_recall.capillary_recall").passed is True
    assert _row(verdict, "reported_dice").passed is True


# ---------------------------------------------------------------------------
# (c) Both-folds rule
# ---------------------------------------------------------------------------


def test_arm_passes_requires_every_fold():
    """arm_passes is False when any single fold verdict fails."""
    from liom_toolkit.segmentation.vseg.gate import arm_passes, evaluate_gate

    passing = evaluate_gate(_passing_metrics(), fold=0)
    failing_metrics = _passing_metrics()
    failing_metrics["centerline_recall"] = 0.85
    failing = evaluate_gate(failing_metrics, fold=1)

    assert arm_passes([passing, failing]) is False
    assert arm_passes([passing, evaluate_gate(_passing_metrics(), fold=1)]) is True


# ---------------------------------------------------------------------------
# (d) Fail-closed undefined
# ---------------------------------------------------------------------------


def test_undefined_gating_row_fails_closed():
    """A gating metric measured "undefined" fails the row and names the cause.

    Cannot-measure is never a silent pass: if a metric raised on every slice
    of a fold the gate must record a failed verdict, not skip the row.
    """
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate

    metrics = _passing_metrics()
    metrics["centerline_recall"] = "undefined"
    verdict = evaluate_gate(metrics, fold=0)

    row = _row(verdict, "centerline_recall")
    assert row.passed is False
    assert "undefined" in row.detail
    assert verdict.passed is False


def test_undefined_reported_row_does_not_fail():
    """A reported row measured "undefined" is recorded but cannot fail the fold."""
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate

    metrics = _passing_metrics()
    metrics["caliber_stratified_recall"]["capillary_recall"] = "undefined"
    metrics["reported_dice"] = "undefined"
    verdict = evaluate_gate(metrics, fold=0)

    assert verdict.passed is True
    assert _row(verdict, "caliber_stratified_recall.capillary_recall").passed is True


def test_missing_gating_key_raises_valueerror():
    """A missing gating metric key raises ValueError naming the key.

    A misspelled or absent key is a defect, not a pass — the gate fails loudly
    instead of silently treating a missing row as satisfied.
    """
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate

    metrics = _passing_metrics()
    del metrics["fpr_on_empty"]
    with pytest.raises(ValueError, match="fpr_on_empty"):
        evaluate_gate(metrics, fold=0)


def test_flat_dotted_key_fallback():
    """Score output uses flat "a.b" keys; the gate resolves them too."""
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate

    metrics = _passing_metrics()
    metrics["caliber_stratified_recall.large_vessel_recall"] = 0.95
    metrics["caliber_stratified_recall.capillary_recall"] = 0.5
    metrics["boundary_artifact_regression.regression_delta"] = 0.0
    del metrics["caliber_stratified_recall"]
    del metrics["boundary_artifact_regression"]

    assert evaluate_gate(metrics, fold=0).passed is True


# ---------------------------------------------------------------------------
# (e) ship_decision — all four branches of the pre-registered rule
# ---------------------------------------------------------------------------


def _verdicts(metrics: dict[str, object], folds=(0, 1)):
    """Build one FoldVerdict per fold for the same metrics dict."""
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate

    return [evaluate_gate(metrics, fold=f) for f in folds]


def test_ship_decision_baseline_wins_on_tie():
    """Baseline passing + an equally-good alternative -> the baseline ships.

    Ties go to the baseline: an alternative must IMPROVE to displace the
    default, not merely match it.
    """
    from liom_toolkit.segmentation.vseg.gate import ship_decision

    per_arm = {
        "baseline": _verdicts(_passing_metrics()),
        "custom_loss": _verdicts(_passing_metrics()),
    }
    decision = ship_decision(per_arm)

    assert decision.ship is True
    assert decision.winner == "baseline"
    assert decision.gate_passed == {"baseline": True, "custom_loss": True}


def test_ship_decision_alternative_wins_on_real_improvement():
    """Alternative improving a gating row > eps with no regressions wins.

    centerline_recall improves by 0.03 (> the 0.02 improvement epsilon) on
    both folds while every other gating row stays equal — the pre-registered
    displacement condition.
    """
    from liom_toolkit.segmentation.vseg.gate import ship_decision

    better = _passing_metrics()
    better["centerline_recall"] = 0.98
    per_arm = {
        "baseline": _verdicts(_passing_metrics()),
        "custom_loss": _verdicts(better),
    }
    decision = ship_decision(per_arm)

    assert decision.ship is True
    assert decision.winner == "custom_loss"


def test_ship_decision_alternative_below_epsilon_loses():
    """An improvement inside the epsilon band does NOT displace the baseline.

    centerline_recall +0.01 is below the 0.02 improvement epsilon — effect
    sizes below epsilon are ties, and ties go to the baseline.
    """
    from liom_toolkit.segmentation.vseg.gate import ship_decision

    slightly_better = _passing_metrics()
    slightly_better["centerline_recall"] = 0.96
    per_arm = {
        "baseline": _verdicts(_passing_metrics()),
        "custom_loss": _verdicts(slightly_better),
    }
    decision = ship_decision(per_arm)

    assert decision.ship is True
    assert decision.winner == "baseline"


def test_ship_decision_regression_beyond_epsilon_blocks_alternative():
    """An alternative that improves one row but regresses another > eps loses."""
    from liom_toolkit.segmentation.vseg.gate import ship_decision

    mixed = _passing_metrics()
    mixed["centerline_recall"] = 0.98
    mixed["fpr_on_empty"] = 0.018  # +0.013 vs baseline 0.005 > the 0.01 eps
    per_arm = {
        "baseline": _verdicts(_passing_metrics()),
        "custom_loss": _verdicts(mixed),
    }
    decision = ship_decision(per_arm)

    assert decision.ship is True
    assert decision.winner == "baseline"


def test_ship_decision_baseline_fails_alternative_passes():
    """When the gate-of-record fails, a both-folds-passing alternative wins."""
    from liom_toolkit.segmentation.vseg.gate import ship_decision

    failing = _passing_metrics()
    failing["centerline_recall"] = 0.80
    per_arm = {
        "baseline": _verdicts(failing),
        "custom_loss": _verdicts(_passing_metrics()),
    }
    decision = ship_decision(per_arm)

    assert decision.ship is True
    assert decision.winner == "custom_loss"
    assert decision.gate_passed["baseline"] is False


def test_ship_decision_all_fail_is_no_ship():
    """If no arm passes the gate the verdict is NO-SHIP — the gate means it."""
    from liom_toolkit.segmentation.vseg.gate import ship_decision

    failing = _passing_metrics()
    failing["centerline_recall"] = 0.80
    per_arm = {
        "baseline": _verdicts(failing),
        "custom_loss": _verdicts(failing),
    }
    decision = ship_decision(per_arm)

    assert decision.ship is False
    assert decision.winner is None


# ---------------------------------------------------------------------------
# (f) score_prediction_set — the eval_metrics composition
# ---------------------------------------------------------------------------


def _vessel_pair(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """A (pred, gt) pair of 256x256 masks with a shared vessel block."""
    gt = np.zeros((size, size), dtype=bool)
    gt[100:120, 60:200] = True
    pred = gt.copy()
    return pred, gt


def test_score_prediction_set_returns_all_metric_keys():
    """score_prediction_set returns all 7 metrics, sub-dicts flattened to a.b.

    The two dict-returning metrics (caliber_stratified_recall,
    boundary_artifact_regression) flatten to `metric.subkey` so the gate's
    dotted-key lookup resolves them directly.
    """
    from liom_toolkit.segmentation.vseg.gate import score_prediction_set

    pred, gt = _vessel_pair()
    result = score_prediction_set([(pred, gt)])

    expected = {
        "centerline_recall",
        "caliber_stratified_recall.capillary_recall",
        "caliber_stratified_recall.large_vessel_recall",
        "boundary_artifact_regression.boundary_quality",
        "boundary_artifact_regression.interior_quality",
        "boundary_artifact_regression.regression_delta",
        "spurious_thin_vessel_rate",
        "fpr_on_empty",
        "cl_dice_metric",
        "reported_dice",
    }
    assert expected <= set(result)
    assert isinstance(result["centerline_recall"], float)
    assert result["centerline_recall"] == pytest.approx(1.0)


def test_score_prediction_set_aggregates_mean_over_defined_slices():
    """Metric values aggregate as the mean over slices where they are defined."""
    from liom_toolkit.segmentation.vseg.gate import score_prediction_set

    pred, gt = _vessel_pair()
    # Second slice: prediction misses half the vessel -> lower recall metrics.
    pred2 = pred.copy()
    pred2[:, 150:] = False
    result = score_prediction_set([(pred, gt), (pred2, gt)])

    assert 0.0 < result["centerline_recall"] < 1.0


def test_score_prediction_set_all_empty_gt_yields_undefined():
    """A fold where every slice is GT-empty yields "undefined" for raising metrics.

    centerline_recall, caliber_stratified_recall, fpr_on_empty, reported_dice,
    boundary_artifact_regression, and spurious_thin_vessel_rate raise on
    empty/undefined input — aggregated they report "undefined" (fail-closed at
    the gate), never a NaN or silent 0.0. cl_dice_metric returns 0.0 on
    both-empty skeletons by upstream convention and stays defined.
    """
    from liom_toolkit.segmentation.vseg.gate import score_prediction_set

    empty = np.zeros((256, 256), dtype=bool)
    result = score_prediction_set([(empty, empty), (empty, empty)])

    assert result["centerline_recall"] == "undefined"
    assert result["caliber_stratified_recall.capillary_recall"] == "undefined"
    assert result["fpr_on_empty"] == "undefined"
    assert result["reported_dice"] == "undefined"
    assert result["boundary_artifact_regression.regression_delta"] == "undefined"
    assert result["spurious_thin_vessel_rate"] == "undefined"
    # cl_dice_metric is defined (0.0) on both-empty skeletons by design.
    assert result["cl_dice_metric"] == pytest.approx(0.0)


def test_score_prediction_set_undefined_feeds_fail_closed_gate():
    """An all-undefined gating row fails the fold — cannot-measure != pass."""
    from liom_toolkit.segmentation.vseg.gate import evaluate_gate, score_prediction_set

    empty = np.zeros((256, 256), dtype=bool)
    metrics = score_prediction_set([(empty, empty)])
    verdict = evaluate_gate(metrics, fold=0)

    assert verdict.passed is False
