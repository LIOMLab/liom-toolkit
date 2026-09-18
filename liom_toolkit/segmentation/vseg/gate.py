"""Direction-aware ship-gate adjudication for vessel segmentation.

The ship gate is a per-metric matrix, not a composite score. Each row carries
its own comparison direction — recall and clDice rows are ``ge`` floors,
rate/delta rows are ``le`` ceilings, and ``reported`` rows are recorded but
structurally cannot gate. If a direction were inverted (an FPR ceiling
evaluated as a floor), the gate would pass failing models silently — this
module makes the direction explicit data so that cannot happen.

``GATE_ROWS`` encodes the pre-registered gate table verbatim; the values are
fixed before any evaluation run, and post-hoc edits invalidate the protocol.
The thresholds anchor to the measured baselines: the legacy ``VsegModel``
(Improved2D: clDice 0.903, centerline 0.898, FPR 0.013) and the honest
nnU-Net v2 benchmark (clDice 0.924, centerline 0.927, FPR 0.009).

Correctness contract: ``evaluate_gate`` fails closed — a gating metric that
is ``"undefined"`` on every slice of a fold fails its row (cannot-measure is
never a silent pass), and a missing gating key raises ``ValueError`` naming
the key (a misspelled row is a defect, not a pass). ``assert`` is never used
for validation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from operator import itemgetter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from numpy.typing import NDArray

import numpy as np

try:
    from liom_toolkit.segmentation.vseg import eval_metrics as _eval_metrics
except ImportError as e:
    raise ImportError("Please install liom-toolkit[seg] to use the ship-gate harness.") from e

__all__ = [
    "GATE_ROWS",
    "IMPROVEMENT_EPS",
    "FoldVerdict",
    "GateRow",
    "RowVerdict",
    "ShipDecision",
    "arm_passes",
    "evaluate_gate",
    "score_prediction_set",
    "ship_decision",
]

logger = logging.getLogger(__name__)

_UNDEFINED = "undefined"
_UNDEFINED_SLICE_MARKER = "vessel-free slice -- metric undefined"


@dataclass(frozen=True)
class GateRow:
    """One row of the ship-gate matrix with its comparison direction.

    Attributes
    ----------
    key : str
        Metric key into the scored-metrics dict. Dotted keys
        (``"a.b"``) resolve as ``metrics["a"]["b"]`` with a flat-key
        fallback for ``score_prediction_set`` output.
    direction : str
        ``"ge"`` (measured >= thresholds), ``"le"`` (measured <=
        thresholds), or ``"reported"`` (never gates).
    gating : bool
        Whether the row participates in pass/fail.
    bound : float | None
        Absolute floor (``ge``) or ceiling (``le``); ``None`` = skipped.
    legacy_ref : float | None
        Legacy-model baseline value anchoring the delta conjunct.
    legacy_margin : float | None
        Required margin over ``legacy_ref`` (``ge``: ``ref + margin``,
        ``le``: ``ref + margin`` as the ceiling).
    nnunet_ref : float | None
        nnU-Net baseline value anchoring the epsilon conjunct.
    nnunet_tolerance : float | None
        Allowed deviation from ``nnunet_ref`` (``ge``: ``ref - tol``,
        ``le``: ``ref + tol``).
    """

    key: str
    direction: str
    gating: bool
    bound: float | None = None
    legacy_ref: float | None = None
    legacy_margin: float | None = None
    nnunet_ref: float | None = None
    nnunet_tolerance: float | None = None


GATE_ROWS: tuple[GateRow, ...] = (
    GateRow(
        "centerline_recall",
        "ge",
        True,
        bound=0.90,
        legacy_ref=0.898,
        legacy_margin=0.01,
        nnunet_ref=0.927,
        nnunet_tolerance=0.02,
    ),
    GateRow(
        "caliber_stratified_recall.large_vessel_recall",
        "ge",
        True,
        bound=0.90,
        legacy_ref=0.898,
        legacy_margin=0.01,
        nnunet_ref=0.927,
        nnunet_tolerance=0.02,
    ),
    # Capillary bin: below the ~2-voxel Nyquist floor at 6.5 um — reported
    # only, re-gates on finer-resolution labeled data.
    GateRow("caliber_stratified_recall.capillary_recall", "reported", False),
    # regression_delta = interior - boundary; positive delta = seam artifact.
    GateRow("boundary_artifact_regression.regression_delta", "le", True, bound=0.02),
    GateRow(
        "spurious_thin_vessel_rate",
        "le",
        True,
        bound=0.005,
        nnunet_ref=0.0,
        nnunet_tolerance=0.005,
    ),
    GateRow(
        "fpr_on_empty",
        "le",
        True,
        bound=0.02,
        legacy_ref=0.013,
        legacy_margin=0.01,
        nnunet_ref=0.009,
        nnunet_tolerance=0.01,
    ),
    GateRow(
        "cl_dice_metric",
        "ge",
        True,
        bound=0.90,
        legacy_ref=0.903,
        legacy_margin=0.01,
        nnunet_ref=0.924,
        nnunet_tolerance=0.02,
    ),
    GateRow("reported_dice", "reported", False),
)

# Improvement epsilon per gating row (mean over folds): 0.02 for
# recall/clDice/delta rows, 0.005 for spurious-thin-vessel rate, 0.01 for
# FPR-on-empty. The same epsilon bounds per-fold regressions.
IMPROVEMENT_EPS: dict[str, float] = {
    "centerline_recall": 0.02,
    "caliber_stratified_recall.large_vessel_recall": 0.02,
    "boundary_artifact_regression.regression_delta": 0.02,
    "spurious_thin_vessel_rate": 0.005,
    "fpr_on_empty": 0.01,
    "cl_dice_metric": 0.02,
}


@dataclass(frozen=True)
class RowVerdict:
    """The verdict for one gate row on one fold.

    Attributes
    ----------
    row : GateRow
        The gate row evaluated.
    measured : float | str | None
        The measured value, or a string marker (``"undefined"`` / a
        metric-error message) when the metric could not produce a number.
    passed : bool
        Whether the row passed (``True`` for reported rows).
    detail : str
        Human-readable explanation of the verdict.
    """

    row: GateRow
    measured: float | str | None
    passed: bool
    detail: str


@dataclass(frozen=True)
class FoldVerdict:
    """The verdict for one fold: per-row results plus the fold-level pass.

    ``passed`` is the conjunction of every gating row's verdict — reported
    rows cannot affect it.
    """

    fold: int | str
    rows: tuple[RowVerdict, ...]
    passed: bool


@dataclass(frozen=True)
class ShipDecision:
    """The adjudicated ship verdict across all contender arms.

    Attributes
    ----------
    winner : str | None
        The winning arm name, or ``None`` on a NO-SHIP verdict.
    ship : bool
        Whether any arm ships.
    gate_passed : dict[str, bool]
        Per-arm gate result (every gating row on both folds).
    improvements : dict[str, dict[str, float]]
        Per-alternative-arm signed improvement over the baseline per gating
        row (positive = better in the row's direction, mean over folds).
    reasons : tuple[str, ...]
        Human-readable explanation of the decision.
    """

    winner: str | None
    ship: bool
    gate_passed: dict[str, bool]
    improvements: dict[str, dict[str, float]] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


def _resolve_key(metrics: Mapping[str, Any], key: str) -> tuple[bool, Any]:
    """Resolve a possibly-dotted gate key against ``metrics``.

    Nested lookup first (``"a.b"`` -> ``metrics["a"]["b"]``), then the flat
    ``"a.b"`` key produced by :func:`score_prediction_set`.

    Returns
    -------
    tuple[bool, Any]
        ``(True, value)`` when the key resolves, ``(False, None)`` otherwise.
    """
    if "." in key:
        head, _, tail = key.partition(".")
        nested = metrics.get(head)
        if isinstance(nested, Mapping) and tail in nested:
            return True, nested[tail]
    if key in metrics:
        return True, metrics[key]
    return False, None


def _conjuncts(row: GateRow) -> list[tuple[str, float]]:
    """Return the (label, threshold) conjuncts for a row, skipping None refs.

    Returns
    -------
    list[tuple[str, float]]
        The conjuncts as ``(label, threshold)`` pairs, in evaluation order.

    Raises
    ------
    ValueError
        If the row direction is not ``"ge"`` or ``"le"`` — an unknown
        direction is a defect, never a silent default.
    """
    if row.direction not in ("ge", "le"):
        raise ValueError(
            f"gate row {row.key!r} has unknown direction {row.direction!r} "
            "— expected 'ge', 'le', or 'reported'"
        )
    out: list[tuple[str, float]] = []
    if row.direction == "ge":
        if row.bound is not None:
            out.append((f"floor F={row.bound}", row.bound))
        if row.legacy_ref is not None and row.legacy_margin is not None:
            t = row.legacy_ref + row.legacy_margin
            out.append((f"legacy+margin={row.legacy_ref}+{row.legacy_margin}", t))
        if row.nnunet_ref is not None and row.nnunet_tolerance is not None:
            t = row.nnunet_ref - row.nnunet_tolerance
            out.append((f"nnunet-tol={row.nnunet_ref}-{row.nnunet_tolerance}", t))
    elif row.direction == "le":
        if row.bound is not None:
            out.append((f"ceiling={row.bound}", row.bound))
        if row.legacy_ref is not None and row.legacy_margin is not None:
            t = row.legacy_ref + row.legacy_margin
            out.append((f"legacy+tol={row.legacy_ref}+{row.legacy_margin}", t))
        if row.nnunet_ref is not None and row.nnunet_tolerance is not None:
            t = row.nnunet_ref + row.nnunet_tolerance
            out.append((f"nnunet+tol={row.nnunet_ref}+{row.nnunet_tolerance}", t))
    return out


def evaluate_gate(metrics: Mapping[str, Any], *, fold: int | str) -> FoldVerdict:
    """Evaluate every gate row against a fold's scored metrics.

    Parameters
    ----------
    metrics : Mapping[str, Any]
        Scored metric values for the fold — nested dicts or the flat
        ``"metric.subkey"`` output of :func:`score_prediction_set`.
    fold : int | str
        Fold identifier carried into the verdict.

    Returns
    -------
    FoldVerdict
        Per-row verdicts plus the fold-level pass (all gating rows).

    Raises
    ------
    ValueError
        If a gating row's metric key is absent — a misspelled key is a
        defect, not a pass.
    """
    verdicts: list[RowVerdict] = []
    for row in GATE_ROWS:
        found, measured = _resolve_key(metrics, row.key)
        if not found:
            if row.gating:
                raise ValueError(
                    f"gate metric {row.key!r} missing from scored metrics — "
                    "a missing gating row cannot pass"
                )
            verdicts.append(RowVerdict(row, None, True, "reported-only (metric missing)"))
            continue

        if row.direction == "reported":
            detail = "reported-only"
            if not isinstance(measured, (int, float)) or isinstance(measured, bool):
                detail = f"reported-only (value {measured!r})"
            verdicts.append(RowVerdict(row, measured, True, detail))
            continue

        if not isinstance(measured, (int, float)) or isinstance(measured, bool):
            # Fail closed: cannot-measure is never a silent pass.
            if measured == _UNDEFINED:
                detail = "undefined on every slice of the fold — fails closed"
            else:
                detail = f"non-numeric measured value {measured!r} — fails closed"
            verdicts.append(RowVerdict(row, measured, False, detail))
            continue

        failures = []
        for label, threshold in _conjuncts(row):
            ok = measured >= threshold if row.direction == "ge" else measured <= threshold
            if not ok:
                failures.append(f"measured {measured} fails {row.direction} {label}")
        passed = not failures
        detail = (
            "; ".join(failures)
            if failures
            else f"measured {measured} satisfies all {row.direction} conjuncts"
        )
        verdicts.append(RowVerdict(row, float(measured), passed, detail))

    return FoldVerdict(
        fold=fold,
        rows=tuple(verdicts),
        passed=all(v.passed for v in verdicts if v.row.gating),
    )


def arm_passes(fold_verdicts: Iterable[FoldVerdict]) -> bool:
    """Return True iff every fold verdict passed — the both-folds rule.

    Returns
    -------
    bool
        ``True`` when every fold verdict's gating rows all passed.

    Raises
    ------
    ValueError
        If ``fold_verdicts`` is empty — an arm with no folds cannot pass.
    """
    verdicts = list(fold_verdicts)
    if not verdicts:
        raise ValueError("arm_passes requires at least one fold verdict")
    return all(v.passed for v in verdicts)


def _row_values(verdict: FoldVerdict) -> dict[str, float]:
    """Numeric measured values for gating rows in one fold verdict.

    Returns
    -------
    dict[str, float]
        Gating-row key -> measured value for rows with numeric measurements.
    """
    return {
        rv.row.key: float(rv.measured)
        for rv in verdict.rows
        if rv.row.gating and isinstance(rv.measured, (int, float))
    }


def _improvement_deltas(
    arm_verdicts: Sequence[FoldVerdict],
    base_verdicts: Sequence[FoldVerdict],
) -> dict[str, float]:
    """Signed mean-over-folds improvement per gating row (positive = better).

    ``ge`` rows improve when ``arm - base > 0``; ``le`` rows improve when
    ``base - arm > 0``. Rows with a non-numeric measurement on either side
    are omitted from the comparison.

    Returns
    -------
    dict[str, float]
        Gating-row key -> signed improvement (positive = arm is better).
    """
    deltas: dict[str, float] = {}
    for row in GATE_ROWS:
        if not row.gating:
            continue
        arm_vals = [
            v for v in (_row_values(fv).get(row.key) for fv in arm_verdicts) if v is not None
        ]
        base_vals = [
            v for v in (_row_values(fv).get(row.key) for fv in base_verdicts) if v is not None
        ]
        if not arm_vals or not base_vals:
            continue
        arm_mean = float(np.mean(arm_vals))
        base_mean = float(np.mean(base_vals))
        deltas[row.key] = arm_mean - base_mean if row.direction == "ge" else base_mean - arm_mean
    return deltas


def _regresses_beyond_eps(
    arm_verdicts: Sequence[FoldVerdict],
    base_verdicts: Sequence[FoldVerdict],
    improvement_eps: Mapping[str, float],
) -> list[str]:
    """Per-fold regression check: rows where the arm is worse than baseline > eps.

    Returns
    -------
    list[str]
        Human-readable regression records, one per (fold, gating row)
        violation; empty when the arm stays within epsilon everywhere.
    """
    regressions: list[str] = []
    for arm_v, base_v in zip(arm_verdicts, base_verdicts, strict=False):
        arm_rows = _row_values(arm_v)
        base_rows = _row_values(base_v)
        for row in GATE_ROWS:
            if not row.gating:
                continue
            a, b = arm_rows.get(row.key), base_rows.get(row.key)
            if a is None or b is None:
                continue
            eps = improvement_eps.get(row.key, 0.02)
            if row.direction == "ge" and a < b - eps:
                regressions.append(f"{row.key} regressed on fold {arm_v.fold}: {a} < {b}-{eps}")
            elif row.direction == "le" and a > b + eps:
                regressions.append(f"{row.key} regressed on fold {arm_v.fold}: {a} > {b}+{eps}")
    return regressions


def ship_decision(
    per_arm: Mapping[str, Sequence[FoldVerdict]],
    *,
    baseline: str = "baseline",
    improvement_eps: Mapping[str, float] = IMPROVEMENT_EPS,
) -> ShipDecision:
    """Adjudicate the ship verdict across contender arms.

    The pre-registered rule: the baseline is the gate-of-record contender; an
    alternative arm replaces it iff it passes the gate on both folds AND
    improves at least one gating row beyond its improvement epsilon (mean
    over folds) AND regresses no gating row beyond that epsilon on either
    fold. Ties go to the baseline. If the baseline fails the gate, the
    passing alternative with the most beyond-epsilon improvements wins (ties
    broken by arm order). If no arm passes, the verdict is NO-SHIP.

    Parameters
    ----------
    per_arm : Mapping[str, Sequence[FoldVerdict]]
        Arm name -> its per-fold verdicts (from :func:`evaluate_gate`).
    baseline : str
        The arm name holding the gate-of-record contender.
    improvement_eps : Mapping[str, float]
        Per-row improvement/regression epsilon (mean over folds for
        improvements; per fold for regressions).

    Returns
    -------
    ShipDecision
        The winning arm, the ship flag, per-arm gate results, per-arm
        improvement deltas, and the reasons for the decision.

    Raises
    ------
    ValueError
        If ``baseline`` is absent from ``per_arm``.
    """
    if baseline not in per_arm:
        raise ValueError(
            f"baseline arm {baseline!r} missing from per_arm "
            f"(arms: {sorted(per_arm)}) — the gate-of-record must be present"
        )

    gate_passed = {arm: arm_passes(v) for arm, v in per_arm.items()}
    base_verdicts = per_arm[baseline]
    reasons: list[str] = []
    improvements: dict[str, dict[str, float]] = {}
    passing_alternatives: list[tuple[str, int]] = []

    for arm, verdicts in per_arm.items():
        if arm == baseline:
            continue
        if not gate_passed[arm]:
            reasons.append(f"{arm}: gate failed — not a contender")
            continue
        deltas = _improvement_deltas(verdicts, base_verdicts)
        improvements[arm] = deltas
        regressions = _regresses_beyond_eps(verdicts, base_verdicts, improvement_eps)
        n_improved = sum(
            1
            for row in GATE_ROWS
            if row.gating and deltas.get(row.key, 0.0) > improvement_eps.get(row.key, 0.02)
        )
        if regressions:
            reasons.append(
                f"{arm}: improved {n_improved} gating row(s) but regressed "
                f"beyond epsilon — {'; '.join(regressions)}"
            )
            continue
        passing_alternatives.append((arm, n_improved))

    if gate_passed[baseline]:
        winners = [(arm, n) for arm, n in passing_alternatives if n >= 1]
        if winners:
            winner = max(winners, key=itemgetter(1))[0]
            reasons.append(
                f"{winner}: passed the gate on both folds and improved "
                "gating row(s) beyond epsilon with no regressions — "
                "displaces the baseline"
            )
        else:
            winner = baseline
            reasons.append(
                "baseline: passed the gate; no alternative improved a gating "
                "row beyond epsilon without regressions — ties go to baseline"
            )
    else:
        reasons.append("baseline: gate failed — gate-of-record cannot ship")
        if passing_alternatives:
            winner = max(passing_alternatives, key=itemgetter(1))[0]
            reasons.append(
                f"{winner}: passed the gate on both folds while the baseline failed — wins the ship"
            )
        else:
            winner = None
            reasons.append("no arm passed the gate on both folds — NO-SHIP")

    return ShipDecision(
        winner=winner,
        ship=winner is not None,
        gate_passed=gate_passed,
        improvements=improvements,
        reasons=tuple(reasons),
    )


def _is_undefined_condition(name: str, pred: NDArray, gt: NDArray) -> bool:
    """Return True when a metric's ValueError matches its emptiness condition.

    The ``"vessel-free slice -- metric undefined"`` marker is only honest when
    the metric was undefined BECAUSE the slice was vessel-free (or the paired
    prediction empty). A ValueError raised for any other reason — shape
    mismatch, patch grid that does not fit, bad input rank — is a real defect
    and is labeled distinctly as ``"metric error: ..."``.

    Returns
    -------
    bool
        ``True`` when the metric's documented undefined-on-empty condition
        holds for this slice; ``False`` otherwise.
    """
    if name in ("centerline_recall", "caliber_stratified_recall", "fpr_on_empty"):
        return not bool(gt.any())
    if name == "spurious_thin_vessel_rate":
        return not bool(pred.any())
    if name in ("reported_dice", "boundary_artifact_regression"):
        return not bool(pred.any()) and not bool(gt.any())
    return False


def score_prediction_set(
    pairs: Sequence[tuple[NDArray, NDArray]],
    *,
    voxel_size_um: float = 6.5,
    capillary_radius_um: float = 5.0,
    boundary_patch_size: tuple[int, int] = (256, 256),
) -> dict[str, float | str]:
    """Score a fold's (prediction, GT) slice pairs through the eval matrix.

    Calls all seven eval metrics per slice, records the undefined marker (or
    a distinct ``"metric error: ..."`` label) when a metric raises, and
    aggregates each metric — or dict-metric sub-key — as the mean over the
    slices where it is defined. Dict-returning metrics flatten to
    ``"metric.subkey"`` keys so gate rows resolve them directly.

    Parameters
    ----------
    pairs : Sequence[tuple[NDArray, NDArray]]
        ``(predicted_mask, gt_mask)`` pairs, one per fold slice.
    voxel_size_um : float
        Voxel edge length in micrometres.
    capillary_radius_um : float
        Radius below which a vessel is binned as capillary.
    boundary_patch_size : tuple[int, int]
        The patch grid cell for the boundary-artifact regression.

    Returns
    -------
    dict[str, float | str]
        Aggregated metric values — floats where defined, ``"undefined"``
        when every slice raised (fail-closed input to the gate).

    Raises
    ------
    ValueError
        If ``pairs`` is empty — a fold with no slices cannot be scored.
    """
    if not pairs:
        raise ValueError("pairs must contain at least one (pred, gt) slice pair")

    scalar_metrics: list[tuple[str, Callable[..., float]]] = [
        ("centerline_recall", _eval_metrics.centerline_recall),
        ("spurious_thin_vessel_rate", _eval_metrics.spurious_thin_vessel_rate),
        ("fpr_on_empty", _eval_metrics.fpr_on_empty),
        ("cl_dice_metric", _eval_metrics.cl_dice_metric),
        ("reported_dice", _eval_metrics.reported_dice),
    ]
    dict_metrics: list[tuple[str, Callable[..., dict[str, float]]]] = [
        ("caliber_stratified_recall", _eval_metrics.caliber_stratified_recall),
        ("boundary_artifact_regression", _eval_metrics.boundary_artifact_regression),
    ]

    per_slice_rows: list[dict[str, Any]] = []
    for pred_raw, gt_raw in pairs:
        pred = np.asarray(pred_raw, dtype=bool)
        gt = np.asarray(gt_raw, dtype=bool)
        row: dict[str, Any] = {}
        for name, fn in scalar_metrics:
            kwargs = (
                {"voxel_size_um": voxel_size_um, "capillary_radius_um": capillary_radius_um}
                if name == "spurious_thin_vessel_rate"
                else {}
            )
            try:
                row[name] = fn(pred, gt, **kwargs)
            except ValueError as e:
                if _is_undefined_condition(name, pred, gt):
                    row[name] = _UNDEFINED_SLICE_MARKER
                    logger.debug("%s undefined on a vessel-free slice: %s", name, e)
                else:
                    row[name] = f"metric error: {e}"
                    logger.warning("%s raised on a non-empty slice: %s", name, e)
        for name, fn in dict_metrics:
            kwargs = (
                {"voxel_size_um": voxel_size_um, "capillary_radius_um": capillary_radius_um}
                if name == "caliber_stratified_recall"
                else {"patch_size": boundary_patch_size}
            )
            try:
                row[name] = fn(pred, gt, **kwargs)
            except ValueError as e:
                if _is_undefined_condition(name, pred, gt):
                    row[name] = _UNDEFINED_SLICE_MARKER
                    logger.debug("%s undefined on a vessel-free slice: %s", name, e)
                else:
                    row[name] = f"metric error: {e}"
                    logger.warning("%s raised on a non-empty slice: %s", name, e)
        per_slice_rows.append(row)

    # Aggregate: mean over slices where the metric (or sub-key) is defined;
    # "undefined" when no slice produced a number — fail-closed input to the
    # gate, never a silent NaN.
    agg: dict[str, float | str] = {}
    for name, _ in scalar_metrics:
        values = [r[name] for r in per_slice_rows if isinstance(r[name], (int, float))]
        agg[name] = float(np.mean(values)) if values else _UNDEFINED
    for name, _ in dict_metrics:
        dicts = [r[name] for r in per_slice_rows if isinstance(r[name], dict)]
        if dicts:
            for k in dicts[0]:
                values = [d[k] for d in dicts if k in d]
                agg[f"{name}.{k}"] = float(np.mean(values)) if values else _UNDEFINED
        else:
            # All slices raised — emit the undefined marker per sub-key the
            # gate knows, so a missing key cannot read as a pass.
            subkeys = (
                ("capillary_recall", "large_vessel_recall")
                if name == "caliber_stratified_recall"
                else ("boundary_quality", "interior_quality", "regression_delta")
            )
            for k in subkeys:
                agg[f"{name}.{k}"] = _UNDEFINED
    return agg
