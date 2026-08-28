"""Unified sidecar-JSON-manifest + per-step ``.done`` marker checkpointing for pipeline resume.

This module provides the resume bookkeeping shared by the three long-running
pipeline functions in the toolkit:

* :func:`liom_toolkit.conversion.conversion.create_full_zarr_volume`
* :func:`liom_toolkit.registration.templating.build_template_for_resolution`
* :func:`liom_toolkit.segmentation.vseg.training.train_model`

Each pipeline writes a sidecar manifest at
``{output_dir}/_liom_checkpoints/{pipeline}.json`` recording:

* ``params_hash`` — sha256 of the canonicalized params JSON. A mismatch
  between the stored hash and the current run's params invalidates the
  checkpoint so resume never silently reuses a stale artifact produced with
  different inputs/code.
* ``completed_steps`` — list of step indices whose ``.done`` marker +
  artifact validation have passed.
* ``artifacts`` — mapping ``{step_index: artifact_path}`` used by the
  artifact-existence validation gate.
* ``last_completed_epoch`` — (``train_model`` only) the index of the last
  fully-trained epoch; complementary to the existing per-epoch
  ``checkpoint.*.pth`` weights artifact (the manifest is the bookkeeper, the
  ``.pth`` is the weights).
* ``complete`` — atomic sentinel written LAST, after every step is done. A
  crash on the final step does NOT leave ``complete=True`` because the
  sentinel is written via the atomic :func:`write_manifest` (temp file +
  ``os.replace``).

Per-step ``.done`` markers (``{output_dir}/_liom_checkpoints/{pipeline}.step_{N}.done``)
are written ONLY after the step's artifact validates (``Path.exists()``), so
a crash mid-step re-runs the step rather than silently trusting a partial
artifact.

Limitation (1.0.0): the manifest records the epoch index for ``train_model``
resume; full-state ``.pth`` augmentation (optimizer / scheduler / RNG /
dataloader-epoch state) is deferred to 1.1. Resume continues from epoch
``N+1`` with a re-initialized optimizer state — this is a known, documented
limitation, not a silent wrong-data fallback. Concurrent runs on the same
checkpoint directory are out of scope (single-process); the manifest format
does not include a lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any

__all__ = [
    "ResumeManager",
    "compute_params_hash",
    "is_step_done",
    "read_manifest",
    "write_done_marker",
    "write_manifest",
]


def compute_params_hash(params: dict[str, Any]) -> str:
    """Return a content-addressing hash of ``params``.

    The hash is ``sha256:<64 hex chars>`` computed over the canonicalized
    JSON serialization of ``params`` (``sort_keys=True`` so dict insertion
    order does not affect the hash; ``default=str`` so non-JSON-native
    values like :class:`pathlib.Path` are stringified rather than raising).

    Parameters
    ----------
    params : dict
        The parameter dict that determines the pipeline output. A change in
        any value (or in the code that consumes them) produces a different
        hash, which invalidates a stale checkpoint on resume.

    Returns
    -------
    str
        ``"sha256:<64 hex chars>"``.
    """
    canonical = json.dumps(params, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def write_manifest(manifest_path: pathlib.Path, data: dict[str, Any]) -> None:
    """Write ``data`` as JSON to ``manifest_path`` atomically.

    The manifest is written to a temp file in the same directory, then
    atomically swapped into place via :func:`os.replace` (POSIX + Windows
    atomic). On any exception the temp file is unlinked — a crash mid-write
    never leaves a corrupt partial manifest at the destination path.

    Parameters
    ----------
    manifest_path : pathlib.Path
        The destination manifest path. The parent directory is created if
        needed.
    data : dict
        The manifest payload (must be JSON-serializable).
    """
    manifest_path = pathlib.Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(manifest_path.parent), suffix=".tmp", prefix=".manifest_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        pathlib.Path(tmp).replace(manifest_path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise


def read_manifest(manifest_path: pathlib.Path) -> dict[str, Any] | None:
    """Read the manifest at ``manifest_path``.

    Parameters
    ----------
    manifest_path : pathlib.Path
        The manifest path to read.

    Returns
    -------
    dict or None
        The manifest dict, or ``None`` if the file does not exist (friendly
        for the resume flow — a missing manifest means a fresh run).

        Note: a manifest file that exists but contains corrupt JSON raises
        ``json.JSONDecodeError`` (transitively from ``json.load``) rather
        than being silently treated as missing — this is an explicit
        failure, not a silent fallback, per the project's "no silent
        wrong-data" rule. ``write_manifest`` uses an atomic temp-file +
        ``Path.replace`` write, so corruption during writes is prevented;
        external corruption (disk error, manual edit) surfaces for the
        user to diagnose. Callers who want a graceful fallback can catch
        ``json.JSONDecodeError`` and treat it as a fresh run, but the
        default behavior is to fail loud.
    """
    manifest_path = pathlib.Path(manifest_path)
    if not manifest_path.exists():
        return None
    with manifest_path.open("r") as f:
        return json.load(f)


def write_done_marker(output_dir: pathlib.Path, pipeline: str, step_index: int) -> None:
    """Write the ``.done`` marker for ``step_index`` of ``pipeline``.

    The marker is a zero-byte file at
    ``{output_dir}/_liom_checkpoints/{pipeline}.step_{step_index}.done``.
    It MUST be written ONLY after the step's artifact has validated (the
    caller's responsibility) — writing it before the artifact exists would
    let a crash mid-step leave a "complete" marker for an incomplete step.

    Parameters
    ----------
    output_dir : pathlib.Path
        The pipeline output directory (the manifest + markers live under
        ``{output_dir}/_liom_checkpoints/``).
    pipeline : str
        The pipeline name (e.g. ``"create_full_zarr_volume"``).
    step_index : int
        The zero-based step index. Must be >= 0.

    Raises
    ------
    ValueError
        If ``step_index`` is negative.
    """
    if step_index < 0:
        raise ValueError(f"step_index must be >= 0, got {step_index}")
    output_dir = pathlib.Path(output_dir)
    marker = output_dir / "_liom_checkpoints" / f"{pipeline}.step_{step_index}.done"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def is_step_done(
    output_dir: pathlib.Path,
    pipeline: str,
    step_index: int,
    artifact_path: pathlib.Path | None = None,
) -> bool:
    """Return ``True`` if step ``step_index`` of ``pipeline`` is complete.

    A step is "done" only if BOTH:

    1. The ``.done`` marker file exists, AND
    2. (when ``artifact_path`` is given) the artifact at ``artifact_path``
       exists.

    The artifact-existence gate prevents silently trusting a partial
    artifact: a crash between the artifact write and the marker write leaves
    a marker without an artifact (or vice versa), and this function returns
    ``False`` so the step re-runs on resume.

    Parameters
    ----------
    output_dir : pathlib.Path
        The pipeline output directory.
    pipeline : str
        The pipeline name.
    step_index : int
        The zero-based step index. Must be >= 0.
    artifact_path : pathlib.Path or None
        Optional artifact path to validate. If ``None``, only the marker is
        checked.

    Returns
    -------
    bool
        ``True`` if the step is complete (marker + artifact valid).

    Raises
    ------
    ValueError
        If ``step_index`` is negative.
    """
    if step_index < 0:
        raise ValueError(f"step_index must be >= 0, got {step_index}")
    output_dir = pathlib.Path(output_dir)
    marker = output_dir / "_liom_checkpoints" / f"{pipeline}.step_{step_index}.done"
    if not marker.exists():
        return False
    return not (artifact_path is not None and not pathlib.Path(artifact_path).exists())


class ResumeManager:
    """Stateful resume bookkeeper for a single pipeline run.

    The manager reads any existing manifest for ``pipeline`` in
    ``output_dir``, compares its ``params_hash`` to the hash of the current
    ``params``, and decides for each step whether to skip it (already
    complete + artifact valid) or run it.

    Stale detection: if the stored ``params_hash`` does not match the
    current params hash, the checkpoint is invalidated — the manager treats
    the manifest as empty and re-runs every step (never silently reuses a
    stale artifact produced with different inputs).

    Parameters
    ----------
    output_dir : pathlib.Path
        The pipeline output directory. The manifest + markers live under
        ``{output_dir}/_liom_checkpoints/``.
    pipeline : str
        The pipeline name (e.g. ``"create_full_zarr_volume"``).
    params : dict
        The current run's params dict. Hashed via :func:`compute_params_hash`
        and compared to the stored hash for stale detection.
    steps_total : int
        The total number of steps in the pipeline.

    Attributes
    ----------
    output_dir : pathlib.Path
        The pipeline output directory.
    pipeline : str
        The pipeline name.
    params_hash : str
        The current run's params hash.
    steps_total : int
        The total number of steps.
    manifest_path : pathlib.Path
        The manifest file path.
    completed_steps : set[int]
        The set of completed step indices (from a non-stale manifest).

    Notes
    -----
    Concurrent runs on the same checkpoint directory are out of scope
    (single-process). The manifest format does not include a lock; two
    concurrent runs could corrupt the manifest.
    """

    def __init__(
        self,
        output_dir: pathlib.Path,
        pipeline: str,
        params: dict[str, Any],
        steps_total: int,
    ) -> None:
        if steps_total < 0:
            raise ValueError(f"steps_total must be >= 0, got {steps_total}")
        self.output_dir = pathlib.Path(output_dir)
        self.pipeline = pipeline
        self.params_hash = compute_params_hash(params)
        self.steps_total = steps_total
        self.manifest_path = self.output_dir / "_liom_checkpoints" / f"{pipeline}.json"
        self._artifacts: dict[str, str] = {}
        existing = read_manifest(self.manifest_path)
        if existing is None or existing.get("params_hash") != self.params_hash:
            # No manifest, or stale (params_hash mismatch) → start fresh.
            self.completed_steps: set[int] = set()
            self._complete = False
            self._last_completed_epoch: int | None = None
        else:
            self.completed_steps = set(existing.get("completed_steps", []))
            self._complete = bool(existing.get("complete", False))
            self._last_completed_epoch = existing.get("last_completed_epoch")
            self._artifacts = dict(existing.get("artifacts", {}))

    # -- step gating -------------------------------------------------------

    def start_step(
        self,
        step_index: int,
        artifact_path: pathlib.Path | None = None,
    ) -> bool:
        """Return ``True`` if the step should run (not done), ``False`` to skip.

        A step is skipped only if it is recorded in ``completed_steps`` AND
        its ``.done`` marker exists AND (when ``artifact_path`` is given) the
        artifact validates. Otherwise the step must run.

        Parameters
        ----------
        step_index : int
            The zero-based step index. Must be >= 0.
        artifact_path : pathlib.Path or None
            Optional artifact path for the artifact-existence gate.

        Returns
        -------
        bool
            ``True`` if the step should run; ``False`` if it is skipped
            (already complete + artifact valid).

        Raises
        ------
        ValueError
            If ``step_index`` is negative.
        """
        if step_index < 0:
            raise ValueError(f"step_index must be >= 0, got {step_index}")
        # The complete sentinel is authoritative: a complete pipeline is a
        # no-op on resume (no work duplicated — idempotency).
        if self._complete:
            return False
        return not (
            step_index in self.completed_steps
            and is_step_done(
                self.output_dir, self.pipeline, step_index, artifact_path=artifact_path
            )
        )

    def finish_step(
        self,
        step_index: int,
        artifact_path: pathlib.Path,
    ) -> None:
        """Mark step ``step_index`` as complete.

        Writes the ``.done`` marker (after the artifact has validated — the
        caller MUST ensure the artifact exists before calling this) and
        updates the manifest atomically: appends ``step_index`` to
        ``completed_steps`` and records the artifact path.

        Parameters
        ----------
        step_index : int
            The zero-based step index. Must be >= 0.
        artifact_path : pathlib.Path
            The step's primary output artifact path (recorded for the
            artifact-existence gate on the next resume).

        Raises
        ------
        ValueError
            If ``step_index`` is negative.
        """
        if step_index < 0:
            raise ValueError(f"step_index must be >= 0, got {step_index}")
        write_done_marker(self.output_dir, self.pipeline, step_index)
        self.completed_steps.add(step_index)
        self._artifacts[str(step_index)] = str(artifact_path)
        self._write_manifest()

    # -- complete sentinel -------------------------------------------------

    def is_complete(self) -> bool:
        """Return ``True`` if the manifest's ``complete`` sentinel is set.

        Returns
        -------
        bool
            ``True`` if the pipeline's complete sentinel is set.
        """
        return self._complete

    def mark_complete(self) -> None:
        """Atomically set the ``complete`` sentinel.

        Written LAST, after every step is done, via :func:`write_manifest`
        (atomic). A crash before this call completes does NOT leave
        ``complete=True``.
        """
        self._complete = True
        self._write_manifest()

    # -- train_model epoch bookkeeping -------------------------------------

    def get_last_completed_epoch(self) -> int | None:
        """Return the last fully-completed epoch index, or ``None``.

        Complementary to the per-epoch ``checkpoint.*.pth`` weights
        artifact — the manifest records the epoch index, NOT the weights
        bytes.

        Returns
        -------
        int or None
            The last fully-completed epoch index, or ``None`` if no epoch
            has completed yet.
        """
        return self._last_completed_epoch

    def set_last_completed_epoch(self, epoch: int) -> None:
        """Record that ``epoch`` has fully completed.

        Writes ``last_completed_epoch`` to the manifest atomically. A crash
        after epoch ``N`` leaves ``last_completed_epoch=N``; resume
        continues from epoch ``N+1``.

        Parameters
        ----------
        epoch : int
            The zero-based epoch index that just completed. Must be >= 0.

        Raises
        ------
        ValueError
            If ``epoch`` is negative.
        """
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        self._last_completed_epoch = epoch
        self._write_manifest()

    # -- internals ---------------------------------------------------------

    def _write_manifest(self) -> None:
        """Write the current state to the manifest atomically."""
        data: dict[str, Any] = {
            "params_hash": self.params_hash,
            "completed_steps": sorted(self.completed_steps),
            "artifacts": self._artifacts,
            "complete": self._complete,
            "steps_total": self.steps_total,
        }
        if self._last_completed_epoch is not None:
            data["last_completed_epoch"] = self._last_completed_epoch
        write_manifest(self.manifest_path, data)
