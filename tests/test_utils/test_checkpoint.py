"""TDD tests for ``liom_toolkit/utils/checkpoint.py``.

Covers the unified sidecar-JSON-manifest + per-step ``.done`` marker
checkpointing used by the three pipeline functions
(``create_full_zarr_volume``, ``build_template_for_resolution``,
``train_model``) to support ``resume=True``.

The manifest is a plain-JSON sidecar written atomically (temp file +
``os.replace``) so a crash mid-write never leaves a corrupt manifest. The
params-hash (sha256 of canonicalized JSON) detects stale checkpoints after
code/param changes — a mismatch invalidates the checkpoint so resume never
silently reuses a stale artifact. The ``.done`` marker for a step is written
ONLY after the step's artifact validates (``Path.exists()``), so a crash
mid-step re-runs the step rather than silently trusting a partial artifact.

All filesystem tests use the builtin ``tmp_path`` fixture (real tmp_path, no
filesystem mocking, per AGENTS.md §5).
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path
from unittest.mock import patch

import pytest

from liom_toolkit.utils.checkpoint import (
    ResumeManager,
    compute_params_hash,
    is_step_done,
    read_manifest,
    write_done_marker,
    write_manifest,
)

# ---------------------------------------------------------------------------
# compute_params_hash
# ---------------------------------------------------------------------------


def test_compute_params_hash_stable():
    """Dict insertion order does not affect the hash (sort_keys=True)."""
    h1 = compute_params_hash({"a": 1, "b": 2})
    h2 = compute_params_hash({"b": 2, "a": 1})
    assert h1 == h2


def test_compute_params_hash_format():
    """The hash is ``sha256:<64 hex chars>``."""
    h = compute_params_hash({"a": 1})
    assert h.startswith("sha256:")
    body = h[len("sha256:"):]
    assert len(body) == 64
    # hex chars only
    int(body, 16)


def test_compute_params_hash_different():
    """Different params produce different hashes."""
    assert compute_params_hash({"a": 1}) != compute_params_hash({"a": 2})


def test_compute_params_hash_non_serializable():
    """Non-JSON-native values (Path) do not raise — default=str handles them."""
    h = compute_params_hash({"path": Path("/tmp")})
    assert h.startswith("sha256:")
    # Deterministic for the str representation
    assert h == compute_params_hash({"path": Path("/tmp")})


# ---------------------------------------------------------------------------
# write_manifest / read_manifest
# ---------------------------------------------------------------------------


def test_write_manifest_atomic(tmp_path):
    """write_manifest writes JSON that round-trips via read_manifest."""
    dest = tmp_path / "manifest.json"
    data = {"params_hash": "sha256:abc", "completed_steps": [0, 1], "complete": False}
    write_manifest(dest, data)
    assert dest.exists()
    assert read_manifest(dest) == data


def test_write_manifest_crash_cleanup(tmp_path):
    """If os.replace raises, the temp file is unlinked and no dest is left.

    A crash mid-write must NOT leave a corrupt partial manifest at the
    destination path. The temp file is removed and the destination does not
    exist.
    """
    dest = tmp_path / "manifest.json"
    data = {"params_hash": "sha256:abc"}

    def _boom(*a, **k):
        raise OSError("simulated crash")

    # write_manifest uses pathlib.Path(tmp).replace(dest) for the atomic
    # swap; patching Path.replace simulates a crash on the final rename.
    with (
        patch.object(pathlib.Path, "replace", side_effect=_boom),
        pytest.raises(OSError),
    ):
        write_manifest(dest, data)

    # Destination must NOT exist (no corrupt partial manifest).
    assert not dest.exists()
    # No leftover .tmp / .partial files in the parent dir.
    leftovers = [p for p in dest.parent.iterdir() if p.suffix == ".tmp" or ".partial" in p.name]
    assert leftovers == [], f"leftover temp files: {leftovers}"


def test_read_manifest_missing(tmp_path):
    """read_manifest on a missing path returns None (friendly for resume flow)."""
    assert read_manifest(tmp_path / "does_not_exist.json") is None


def test_read_manifest_present(tmp_path):
    """read_manifest returns the dict written by write_manifest."""
    dest = tmp_path / "manifest.json"
    data = {"params_hash": "sha256:xyz", "completed_steps": []}
    write_manifest(dest, data)
    assert read_manifest(dest) == data


# ---------------------------------------------------------------------------
# .done markers
# ---------------------------------------------------------------------------


def test_done_marker_write_read(tmp_path):
    """write_done_marker creates the marker; is_step_done returns True when
    the marker exists AND the artifact validates."""
    out = tmp_path / "out"
    artifact = tmp_path / "artifact.nii.gz"
    artifact.write_bytes(b"data")
    write_done_marker(out, "create_full_zarr_volume", 0)
    assert is_step_done(out, "create_full_zarr_volume", 0, artifact_path=artifact) is True


def test_done_marker_missing(tmp_path):
    """is_step_done returns False if the marker file does not exist."""
    out = tmp_path / "out"
    assert is_step_done(out, "create_full_zarr_volume", 0) is False


def test_done_marker_artifact_missing(tmp_path):
    """is_step_done returns False if the marker exists BUT the artifact is
    missing (artifact validation gate — never silently use a partial artifact)."""
    out = tmp_path / "out"
    artifact = tmp_path / "missing.nii.gz"  # does not exist
    write_done_marker(out, "create_full_zarr_volume", 0)
    assert is_step_done(out, "create_full_zarr_volume", 0, artifact_path=artifact) is False


# ---------------------------------------------------------------------------
# ResumeManager — stale detection, idempotency, partial resume
# ---------------------------------------------------------------------------


def test_stale_checkpoint_detection(tmp_path):
    """A manifest with a params_hash that does not match the current params is
    detected as stale and invalidated (re-run from scratch)."""
    out = tmp_path / "out"
    out.mkdir()
    manifest_path = out / "_liom_checkpoints" / "create_full_zarr_volume.json"
    manifest_path.parent.mkdir(parents=True)
    # Write a manifest with a stale params_hash.
    write_manifest(
        manifest_path,
        {
            "params_hash": "sha256:stale",
            "completed_steps": [0, 1],
            "complete": False,
            "steps_total": 5,
        },
    )
    # ResumeManager with current params hashing to something else must detect
    # the stale checkpoint. Per the planner's choice, a stale checkpoint is
    # invalidated (re-run from scratch): is_complete() is False and start_step
    # returns True for step 0 (the manifest is treated as empty).
    mgr = ResumeManager(
        output_dir=out,
        pipeline="create_full_zarr_volume",
        params={"a": 1},
        steps_total=5,
    )
    assert mgr.is_complete() is False
    # Stale → step 0 must run (not skipped).
    assert mgr.start_step(0) is True


def test_resume_complete_pipeline_noop(tmp_path):
    """A manifest with complete=True → resume is a no-op (all steps skipped)."""
    out = tmp_path / "out"
    out.mkdir()
    params = {"a": 1}
    h = compute_params_hash(params)
    manifest_path = out / "_liom_checkpoints" / "create_full_zarr_volume.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(
        manifest_path,
        {
            "params_hash": h,
            "completed_steps": [0, 1, 2, 3, 4],
            "complete": True,
            "steps_total": 5,
        },
    )
    mgr = ResumeManager(
        output_dir=out,
        pipeline="create_full_zarr_volume",
        params=params,
        steps_total=5,
    )
    assert mgr.is_complete() is True
    # Every step is skipped (no work duplicated — idempotency).
    for i in range(5):
        assert mgr.start_step(i) is False


def test_resume_partial_skips_completed(tmp_path):
    """A manifest with completed_steps=[0, 1] and steps_total=5 → resume skips
    steps 0 and 1, starts at step 2."""
    out = tmp_path / "out"
    out.mkdir()
    params = {"a": 1}
    h = compute_params_hash(params)
    manifest_path = out / "_liom_checkpoints" / "create_full_zarr_volume.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Write .done markers + artifacts for steps 0 and 1 so the artifact
    # validation gate passes.
    art0 = out / "step0.out"
    art1 = out / "step1.out"
    art0.write_bytes(b"x")
    art1.write_bytes(b"x")
    write_done_marker(out, "create_full_zarr_volume", 0)
    write_done_marker(out, "create_full_zarr_volume", 1)
    write_manifest(
        manifest_path,
        {
            "params_hash": h,
            "completed_steps": [0, 1],
            "complete": False,
            "steps_total": 5,
            "artifacts": {0: str(art0), 1: str(art1)},
        },
    )
    mgr = ResumeManager(
        output_dir=out,
        pipeline="create_full_zarr_volume",
        params=params,
        steps_total=5,
    )
    # Steps 0 and 1 are skipped (marker + artifact valid).
    assert mgr.start_step(0, artifact_path=art0) is False
    assert mgr.start_step(1, artifact_path=art1) is False
    # Step 2 must run.
    assert mgr.start_step(2) is True


def test_manifest_complete_sentinel_atomic(tmp_path):
    """The complete=True sentinel is written via write_manifest (atomic).

    A crash before the final write does not leave complete=True. We simulate
    a crash inside mark_complete by patching os.replace to raise, and assert
    the manifest's complete flag is NOT True afterwards (the atomic write
    either fully succeeds or leaves the prior state, never a partial
    complete=True).
    """
    out = tmp_path / "out"
    out.mkdir()
    params = {"a": 1}
    mgr = ResumeManager(
        output_dir=out,
        pipeline="train_model",
        params=params,
        steps_total=3,
    )
    # Record one completed epoch so the manifest exists with complete=False.
    mgr.set_last_completed_epoch(0)
    manifest_path = out / "_liom_checkpoints" / "train_model.json"
    assert read_manifest(manifest_path)["complete"] is False

    # Now simulate a crash during the atomic complete-sentinel write.
    with (
        patch.object(
            pathlib.Path, "replace", side_effect=OSError("simulated crash on final write")
        ),
        pytest.raises(OSError),
    ):
        mgr.mark_complete()

    # The manifest must NOT have complete=True (the atomic write failed).
    after = read_manifest(manifest_path)
    assert after is not None
    assert after["complete"] is False


# ---------------------------------------------------------------------------
# ResumeManager — last_completed_epoch (train_model)
# ---------------------------------------------------------------------------


def test_resume_manager_last_completed_epoch(tmp_path):
    """set_last_completed_epoch / get_last_completed_epoch round-trip via the
    manifest (complementary to the per-epoch checkpoint.*.pth)."""
    out = tmp_path / "out"
    out.mkdir()
    mgr = ResumeManager(
        output_dir=out,
        pipeline="train_model",
        params={"lr": 0.001},
        steps_total=10,
    )
    assert mgr.get_last_completed_epoch() is None
    mgr.set_last_completed_epoch(5)
    assert mgr.get_last_completed_epoch() == 5
    # The manifest records the epoch index, NOT the weights bytes.
    manifest_path = out / "_liom_checkpoints" / "train_model.json"
    data = json.loads(manifest_path.read_text())
    assert data["last_completed_epoch"] == 5
    assert "weights" not in data
    assert "state_dict" not in data


def test_resume_manager_finish_step_updates_manifest(tmp_path):
    """finish_step writes the .done marker and appends the step to
    completed_steps in the manifest atomically."""
    out = tmp_path / "out"
    out.mkdir()
    artifact = out / "step0.out"
    artifact.write_bytes(b"x")
    mgr = ResumeManager(
        output_dir=out,
        pipeline="create_full_zarr_volume",
        params={"a": 1},
        steps_total=3,
    )
    mgr.finish_step(0, artifact_path=artifact)
    # Marker exists.
    assert is_step_done(out, "create_full_zarr_volume", 0, artifact_path=artifact)
    # Manifest records the completed step.
    manifest_path = out / "_liom_checkpoints" / "create_full_zarr_volume.json"
    data = read_manifest(manifest_path)
    assert 0 in data["completed_steps"]
    assert data["artifacts"]["0"] == str(artifact)


def test_resume_manager_rejects_negative_step(tmp_path):
    """Negative step_index raises ValueError (assert is not validation)."""
    out = tmp_path / "out"
    out.mkdir()
    mgr = ResumeManager(
        output_dir=out,
        pipeline="create_full_zarr_volume",
        params={"a": 1},
        steps_total=3,
    )
    with pytest.raises(ValueError):
        mgr.start_step(-1)
