"""Tests for the 2-fold leave-one-brain-out splits writer + verifier.

``write_loo_splits`` consumes the ``case_brain_manifest.json`` written by
``prepare_nnunet_2d`` and produces the hand-written ``splits_final.json``
nnU-Net's ``do_split`` honors verbatim — the gate-of-record split that makes
cross-brain evaluation auditable. ``verify_loo_splits`` is load-bearing:
nnU-Net only *warns* on train/val overlap, so every malformed variant must
raise here before the file is written:

* overlap — a case in both train and val of the same fold
* cross-brain contamination — a brain's cases split across train/val
  (per-volume split, never slice-level i.i.d.)
* wrong fold count — exactly 2 folds required
* non-complementary val brains — the two folds must swap brains
* manifest-missing cases — a splits case absent from the manifest
* coverage — every manifest case must appear in exactly one fold's val
* non-2-brain manifest — the LOO protocol is defined for exactly 2 brains

These tests do NOT need torch (pure json + pathlib IO) — no
``importorskip``.
"""

from __future__ import annotations

import json

import pytest

S23_CASES = [f"s23_{s}" for s in ("575", "700", "750", "800", "1000", "1110", "1200", "1350", "1500")]
S24_CASES = ["s24_500", "s24_1200"]


def _manifest() -> dict[str, str]:
    """The 11-slice / 2-brain labeled pool as a case → brain mapping."""
    return {c: "s23" for c in S23_CASES} | {c: "s24" for c in S24_CASES}


def test_write_loo_splits_produces_two_complementary_folds(tmp_path) -> None:
    """A 9+2 manifest yields 2 folds: fold 0 vals s24, fold 1 vals s23.

    Fold 0 reproduces the S23→S24 benchmark direction: val defaults to the
    lexicographically larger brain name (s24 > s23). Case lists are sorted.
    """
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import write_loo_splits

    out = tmp_path / "splits_final.json"
    splits = write_loo_splits(_manifest(), str(out))

    assert len(splits) == 2, "LOO over 2 brains must produce exactly 2 folds"
    assert splits[0]["val"] == sorted(S24_CASES)
    assert splits[0]["train"] == sorted(S23_CASES)
    assert splits[1]["val"] == sorted(S23_CASES)
    assert splits[1]["train"] == sorted(S24_CASES)


def test_write_loo_splits_every_case_in_exactly_one_val(tmp_path) -> None:
    """Every manifest case appears in exactly one fold's val set."""
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import write_loo_splits

    splits = write_loo_splits(_manifest(), str(tmp_path / "splits_final.json"))
    seen: list[str] = []
    for fold in splits:
        seen.extend(fold["val"])
    assert sorted(seen) == sorted(_manifest().keys())


def test_write_loo_splits_accepts_manifest_path(tmp_path) -> None:
    """The manifest argument may be a path to case_brain_manifest.json."""
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import write_loo_splits

    manifest_path = tmp_path / "case_brain_manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    splits = write_loo_splits(str(manifest_path), str(tmp_path / "splits_final.json"))
    assert len(splits) == 2


def test_write_loo_splits_val_first_flips_fold_order(tmp_path) -> None:
    """val_first="s23" makes fold 0 validate on s23 instead of s24."""
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import write_loo_splits

    splits = write_loo_splits(_manifest(), str(tmp_path / "splits.json"), val_first="s23")
    assert splits[0]["val"] == sorted(S23_CASES)
    assert splits[0]["train"] == sorted(S24_CASES)
    assert splits[1]["val"] == sorted(S24_CASES)
    assert splits[1]["train"] == sorted(S23_CASES)


def test_write_loo_splits_rejects_three_brain_manifest(tmp_path) -> None:
    """A non-2-brain manifest raises ValueError naming the brains found."""
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import write_loo_splits

    manifest = _manifest() | {"s25_100": "s25"}
    with pytest.raises(ValueError, match="s25"):
        write_loo_splits(manifest, str(tmp_path / "splits.json"))


def test_write_loo_splits_rejects_unknown_val_first(tmp_path) -> None:
    """val_first must be one of the manifest's brains."""
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import write_loo_splits

    with pytest.raises(ValueError, match="s99"):
        write_loo_splits(_manifest(), str(tmp_path / "splits.json"), val_first="s99")


def test_verify_loo_splits_rejects_train_val_overlap(tmp_path) -> None:
    """A case in both train and val of the same fold raises, naming it."""
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import verify_loo_splits

    manifest = _manifest()
    splits = [
        {"train": sorted(S23_CASES) + ["s24_500"], "val": ["s24_500", "s24_1200"]},
        {"train": sorted(S24_CASES), "val": sorted(S23_CASES)},
    ]
    with pytest.raises(ValueError, match="s24_500"):
        verify_loo_splits(splits, manifest)


def test_verify_loo_splits_rejects_brain_split_across_train_val(tmp_path) -> None:
    """A brain's cases split across train/val within a fold raises.

    Per-volume split is the whole point: s24_500 in train while s24_1200
    vals is a slice-level leak even with no literal case overlap.
    """
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import verify_loo_splits

    manifest = _manifest()
    splits = [
        {"train": sorted(S23_CASES) + ["s24_500"], "val": ["s24_1200"]},
        {"train": sorted(S24_CASES), "val": sorted(S23_CASES)},
    ]
    with pytest.raises(ValueError, match="s24"):
        verify_loo_splits(splits, manifest)


def test_verify_loo_splits_rejects_wrong_fold_count(tmp_path) -> None:
    """len(splits) != 2 raises — the LOO protocol is exactly 2 folds."""
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import verify_loo_splits

    manifest = _manifest()
    one_fold = [{"train": sorted(S23_CASES), "val": sorted(S24_CASES)}]
    with pytest.raises(ValueError, match="2"):
        verify_loo_splits(one_fold, manifest)


def test_verify_loo_splits_rejects_case_absent_from_manifest(tmp_path) -> None:
    """A splits case absent from the manifest raises, naming it."""
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import verify_loo_splits

    manifest = _manifest()
    splits = [
        {"train": sorted(S23_CASES), "val": ["s24_500", "s24_1200"]},
        {"train": ["s24_500", "s24_1200", "ghost_1"], "val": sorted(S23_CASES)},
    ]
    with pytest.raises(ValueError, match="ghost_1"):
        verify_loo_splits(splits, manifest)


def test_verify_loo_splits_rejects_manifest_case_in_neither_val(tmp_path) -> None:
    """A manifest case that never vals raises — it is never evaluated."""
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import verify_loo_splits

    manifest = _manifest()
    splits = [
        {"train": sorted(S23_CASES), "val": ["s24_500"]},  # s24_1200 never vals
        {"train": sorted(S24_CASES), "val": sorted(S23_CASES)},
    ]
    with pytest.raises(ValueError, match="s24_1200"):
        verify_loo_splits(splits, manifest)


def test_verify_loo_splits_rejects_non_complementary_val_brains(tmp_path) -> None:
    """Both folds valing the same brain raises — folds must swap brains."""
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import verify_loo_splits

    manifest = _manifest()
    splits = [
        {"train": sorted(S23_CASES), "val": sorted(S24_CASES)},
        {"train": sorted(S23_CASES), "val": sorted(S24_CASES)},
    ]
    with pytest.raises(ValueError):
        verify_loo_splits(splits, manifest)


def test_written_splits_round_trip_through_verify(tmp_path) -> None:
    """The written splits_final.json round-trips json.load + verify.

    This is the remote-run contract: the file lands at
    ``$nnUNet_preprocessed/<Dataset>/splits_final.json`` and the verifier
    re-runs on the written file before training.
    """
    from liom_toolkit.scripts.liom_prepare_nnunet_dataset import (
        verify_loo_splits,
        write_loo_splits,
    )

    manifest = _manifest()
    out = tmp_path / "splits_final.json"
    write_loo_splits(manifest, str(out))
    loaded = json.loads(out.read_text())
    verify_loo_splits(loaded, manifest)  # must not raise
    assert loaded[0]["val"] == sorted(S24_CASES)
