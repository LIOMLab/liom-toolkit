"""SSH-driven staged remote run: Dataset101 rebuild → 3 arms → gate eval → ship export.

The in-CI vseg tests (``test_gate.py``, ``test_splits_writer.py``,
``test_nnunet_trainer.py``, ``test_artifact_fetch.py``) prove the pieces on
synthetic data — the direction-aware gate rows, the LOO splits verifier, the
custom-loss trainer pair, the verified-artifact fetch. This file is the
remote run itself: it SSHes to the lab CUDA box and executes the whole
pipeline as seven STAGED test functions, each one idempotent remote stage
guarded by a ``$LIOM_VSEG_RUN_DIR/.done_stageNN`` sentinel plus
artifact-existence checks. A ~10-20 h GPU pipeline must be resumable at
stage boundaries — re-running a completed stage is a no-op, a crashed stage
resumes at its own boundary, and each stage is separately selectable via
``pytest -k stageNN``.

Stages:
* ``test_stage00_remote_probe`` — ``echo ok`` + ``nvidia-smi`` GPU count.
* ``test_stage10_dataset_and_splits`` — repo sync (``LIOM_VSEG_REF``
  checkout or ``git pull --ff-only``) + stale-checkout detection →
  ``uv sync --extra ai --extra benchmark --extra seg`` → brain-mapped
  Dataset101 rebuild via ``liom-prepare-nnunet-dataset`` →
  ``nnUNetv2_plan_and_preprocess`` → verified 2-fold LOO
  ``splits_final.json`` (``write_loo_splits`` + ``verify_loo_splits``
  re-read) → resolved ``network_class_name`` recorded to
  ``plans_arch.txt`` for the arm-3 gate.
* ``test_stage20_arm1_baseline`` — ``nnUNetTrainer_50epochs`` folds 0/1
  in-process via ``run_training``.
* ``test_stage30_arm2_cldice`` — ``LiomDiceFocalClDiceTrainer`` folds 0/1.
* ``test_stage40_arm3_warmstart`` — ``LiomDiceFocalClDiceWarmStartTrainer``
  folds 0/1 with ``pretrained_weights=$LIOM_VSEG_PRETRAINED``; gated on the
  plans-architecture check (FAILS naming the resolved arch on a
  non-PlainConvUNet plans result — a ResEnc-plans rebuild would crash
  ``load_pretrained_weights`` hours into the run).
* ``test_stage50_gate_eval`` — LOCAL pre-registration assertion
  (``16-EVAL-RESULTS.md`` must exist and contain ``PRE-REGISTERED`` before
  any remote eval runs), then remote per-arm × per-fold scoring through
  ``score_prediction_set`` → ``evaluate_gate`` → ``arm_passes`` →
  ``ship_decision`` → ``gate_results.json`` echoed between
  ``LIOM_VSEG_GATE_JSON_BEGIN``/``END`` markers and saved locally to
  ``LIOM_VSEG_RESULTS_JSON``.
* ``test_stage60_export_ship`` — reads the local verdict; NO-SHIP →
  ``pytest.skip`` naming the verdict (exporting a model that failed the
  gate is meaningless). Otherwise optional ``fold_all`` for the winner
  (``LIOM_VSEG_FOLD_ALL`` != ``0``), ``nnUNetv2_export_model_to_zip``,
  ``sha256sum`` → ``SHA256SUMS``, and an scp pull into
  ``LIOM_VSEG_ARTIFACT_DIR`` with a local hash + member-list check.

Skip policy (CI legs without the box pass cleanly — never fail):
* ``LIOM_CUDA_HOST`` env var unset → skipped at collection via
  ``pytest.mark.skipif``.
* The host is unreachable (SSH connect timeout / auth failure) →
  ``pytest.skip`` inside the body (a missing lab box is an environment
  condition, not a regression).
* ``nvidia-smi`` reports 0 GPUs → ``pytest.skip`` (no CUDA to train on).
* A NO-SHIP gate verdict → ``pytest.skip`` on the export stage only.
  The arm-3 architecture mismatch is a FAIL (a protocol violation), never
  a skip.

Configuration via env vars (defaults match the lab box layout; every
remote path is a parameter, never a hardcoded library default — AGENTS §1):
* ``LIOM_CUDA_HOST`` — the SSH host (no default; must be set or every test
  skips at collection).
* ``LIOM_CUDA_USER`` — the SSH user (default: current local user).
* ``LIOM_CUDA_REPO`` — the repo path on the remote (default
  ``~/code/liom-toolkit``).
* ``LIOM_VSEG_DATASET_DIRS`` — space-separated list of per-brain labeled-
  slice directories on the remote (default
  ``~/code/vseg/data/LSFM_dataset/s23 ~/code/vseg/data/LSFM_dataset/s24``).
* ``LIOM_VSEG_PRETRAINED`` — the arm-3 pretrained checkpoint on the remote
  (default ``/data/LSFM/ssl_pretrained/pretrained.pth``).
* ``LIOM_VSEG_RUN_DIR`` — remote run directory holding the stage sentinels,
  ``plans_arch.txt``, ``gate_results.json``, the exported zip, and
  ``SHA256SUMS`` (default ``~/phase16_vseg``).
* ``LIOM_VSEG_NGPUS`` — GPU count threaded to ``run_training(num_gpus=)``
  and ``nnUNetv2_train -num_gpus`` (default ``2``).
* ``LIOM_VSEG_REF`` — pushed ref/SHA the remote ``git fetch``es +
  checks out; unset → ``git pull --ff-only``.
* ``LIOM_VSEG_DATASET_ID`` / ``LIOM_VSEG_DATASET_NAME`` /
  ``LIOM_VSEG_PLANS_ID`` — the nnU-Net dataset id / name / plans identifier
  (defaults ``101`` / ``Dataset101_LIOM6p5`` / ``nnUNetPlans`` — the
  pre-registered pipeline pins; arm 3's warm-start key-match requires
  ``nnUNetPlans``).
* ``LIOM_VSEG_RESULTS_JSON`` — local path for the parsed gate verdict
  (default the phase dir's ``16-gate-results.json``).
* ``LIOM_VSEG_ARTIFACT_DIR`` — local directory for the pulled model zip +
  ``SHA256SUMS`` (default ``.planning/tmp/phase16`` — gitignored).
* ``LIOM_VSEG_FOLD_ALL`` — set to ``0`` to skip the winner's all-data
  ``fold_all`` run (default ``1``; the gated 2-fold ensemble ships either
  way — ``fold_all`` is appended to the export only when its checkpoint
  exists).

Uses ONLY the stdlib ``subprocess`` module to drive ``ssh``/``scp`` with a
list argv (never a shell — T-16-30 subprocess injection; no
paramiko / fabric dependency — AGENTS §3). Each remote stage is a single
``bash -s`` script piped over SSH: ``set -euo pipefail`` makes any failure
exit non-zero, ``~``/``$HOME`` prefixes are resolved against the REMOTE
home BEFORE ``shlex.quote`` (quoting prevents tilde expansion), the
``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results`` /
``nnUNet_extTrainer`` env vars are exported before any nnU-Net call
(unset vars cause opaque downstream failures), and
``multiprocessing.set_start_method("spawn", force=True)`` precedes the
torch import in every training heredoc (fork-after-CUDA deadlocks the
nnU-Net dataloader workers). A ``LIOM_VSEG_*_OK`` sentinel per stage
proves the stage reached its end; a non-zero exit surfaces the tail of
remote stdout/stderr in the assertion message. The ship decision itself is
pre-registered — the verdict is deterministic from ``gate_results.json``,
and this driver never edits thresholds after seeing results.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path
from typing import Any

import pytest

# Resolve ssh/scp to their full paths once so subprocess calls do not use a
# partial executable name (avoids the start-process-with-partial-path lint
# without an inline suppression; shutil.which is the stdlib way to resolve a
# PATH binary). Falls back to the bare name when absent — the SSH probe then
# fails fast and the test skips (a missing ssh is an environment condition,
# not a bug).
_SSH = shutil.which("ssh") or "ssh"
_SCP = shutil.which("scp") or "scp"

_LIOM_CUDA_HOST = os.environ.get("LIOM_CUDA_HOST")
_LIOM_CUDA_USER = os.environ.get("LIOM_CUDA_USER") or getpass.getuser()
_LIOM_CUDA_REPO = os.environ.get("LIOM_CUDA_REPO") or "~/code/liom-toolkit"

# Repo-root-relative local artifacts. parents[3]: vseg/ → test_segmentation/
# → tests/ → repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVAL_RESULTS_DOC = (
    _REPO_ROOT
    / ".planning"
    / "phases"
    / "liom-toolkit-16-vseg-training-evaluation-and-ship"
    / "16-EVAL-RESULTS.md"
)
_DEFAULT_RESULTS_JSON = (
    ".planning/phases/liom-toolkit-16-vseg-training-evaluation-and-ship/16-gate-results.json"
)
_DEFAULT_ARTIFACT_DIR = ".planning/tmp/phase16"

pytestmark = [
    pytest.mark.ai,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _LIOM_CUDA_HOST,
        reason="needs LIOM_CUDA_HOST env var (lab CUDA box) — CI legs skip cleanly",
    ),
]

# Arm name -> nnU-Net trainer class name. The trainer name is a component of
# the nnU-Net results dir (<trainer>__<plans>__<config>), so a distinct
# class name per arm keeps the fold output dirs collision-free — sharing a
# name would let an existing fold checkpoint silently resume the wrong arm.
# This map is fixed by the pre-registered protocol; the eval stage and the
# export stage both consume it so a typo'd arm name fails at the map, not
# as a missing directory.
_ARM_TRAINERS = {
    "baseline": "nnUNetTrainer_50epochs",
    "cldice": "LiomDiceFocalClDiceTrainer",
    "warmstart": "LiomDiceFocalClDiceWarmStartTrainer",
}


def _ssh(host: str, user: str, cmd: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run ``cmd`` on ``host`` via ssh and return the completed process.

    Uses ``-o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new``
    so an unreachable / no-key host fails fast instead of hanging the test
    (a missing lab box is an environment condition, not a regression — the
    caller skips on failure).
    """
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - controlled ssh invocation
        [
            _SSH,
            "-o",
            "ConnectTimeout=10",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"{user}@{host}",
            cmd,
        ],
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _scp_pull(host: str, user: str, remote_path: str, local_dir: Path) -> None:
    """scp ``remote_path`` from ``host`` into ``local_dir``; assert success.

    List argv with the same ``-o`` flags as ``_ssh``; the remote path is
    ``shlex.quote``d inside the ``user@host:`` spec so the remote shell
    resolves a quoted literal (no glob/tilde reinterpretation).
    """
    pull = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - controlled scp invocation
        [
            _SCP,
            "-o",
            "ConnectTimeout=10",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"{user}@{host}:{shlex.quote(remote_path)}",
            str(local_dir),
        ],
        capture_output=True,
        check=False,
        timeout=600,
    )
    assert pull.returncode == 0, (
        f"scp pull of {remote_path} failed (exit {pull.returncode}): "
        f"{pull.stderr.decode(errors='replace')[:500]}"
    )


def _connect() -> tuple[str, str, str]:
    """Probe SSH connectivity; return ``(host, user, remote_home)``.

    Skips (never fails) on an unreachable host — a missing lab box is an
    environment condition, not a regression.
    """
    host = _LIOM_CUDA_HOST
    assert host is not None  # skipif guards this; assert for type-checkers
    user = _LIOM_CUDA_USER
    probe = _ssh(host, user, "echo ok", timeout=20)
    if probe.returncode != 0:
        pytest.skip(
            f"LIOM_CUDA_HOST {host} unreachable (ssh exit {probe.returncode}): "
            f"{probe.stderr.decode(errors='replace')[:200]}"
        )
    home_probe = _ssh(host, user, "echo $HOME", timeout=15)
    if home_probe.returncode != 0:
        pytest.skip(f"$HOME probe failed on {host} (exit {home_probe.returncode})")
    home_lines = home_probe.stdout.decode(errors="replace").strip().splitlines()
    if not home_lines:
        pytest.skip(f"empty $HOME probe output on {host}")
    return host, user, home_lines[-1].strip()


def _require_gpus(host: str, user: str) -> None:
    """Skip when ``nvidia-smi`` reports 0 CUDA GPUs on the remote."""
    gpu_probe = _ssh(host, user, "nvidia-smi -L | grep -c 'GPU ' || echo 0", timeout=20)
    if gpu_probe.returncode != 0:
        pytest.skip(f"nvidia-smi probe failed on {host} (exit {gpu_probe.returncode})")
    gpu_lines = gpu_probe.stdout.decode(errors="replace").strip().splitlines()
    if not gpu_lines:
        pytest.skip(f"empty nvidia-smi probe output on {host}")
    gpu_count = gpu_lines[-1].strip()
    if not gpu_count.isdigit() or int(gpu_count) < 1:
        pytest.skip(f"no CUDA GPUs reported by nvidia-smi on {host} (got {gpu_count!r})")


def _resolve_remote(path: str, remote_home: str) -> str:
    """Resolve ``~``/``$HOME``/relative prefixes against the REMOTE home.

    ``shlex.quote`` wraps paths in single quotes, which prevents ``~`` and
    ``$HOME`` expansion on the remote (``cd '~/code/...'`` is a literal
    no-such-directory). Resolve the prefix here — against the remote home
    probed over SSH — so every quoted path lands on a real absolute path.
    A user-supplied absolute path is used verbatim.
    """
    if path.startswith(("~", "$HOME")):
        return remote_home + path[path.find("/") :]
    if not os.path.isabs(path):
        return f"{remote_home}/{path}"
    return path


def _remote_env(remote_home: str) -> dict[str, Any]:
    """Resolve every ``LIOM_VSEG_*`` remote path/config value.

    Defaults match the lab box layout; every value is overridable so the
    driver never hardcodes a lab-specific location (AGENTS §1).
    """
    dataset_id = os.environ.get("LIOM_VSEG_DATASET_ID") or "101"
    dataset_dirs_raw = os.environ.get("LIOM_VSEG_DATASET_DIRS") or (
        "~/code/vseg/data/LSFM_dataset/s23 ~/code/vseg/data/LSFM_dataset/s24"
    )
    return {
        "repo": _resolve_remote(_LIOM_CUDA_REPO, remote_home),
        "run_dir": _resolve_remote(
            os.environ.get("LIOM_VSEG_RUN_DIR") or "~/phase16_vseg", remote_home
        ),
        "dataset_dirs": [_resolve_remote(d, remote_home) for d in dataset_dirs_raw.split()],
        "pretrained": _resolve_remote(
            os.environ.get("LIOM_VSEG_PRETRAINED") or "/data/LSFM/ssl_pretrained/pretrained.pth",
            remote_home,
        ),
        "ref": os.environ.get("LIOM_VSEG_REF") or "",
        "ngpus": os.environ.get("LIOM_VSEG_NGPUS") or "2",
        "dataset_id": dataset_id,
        "dataset_name": os.environ.get("LIOM_VSEG_DATASET_NAME")
        or f"Dataset{int(dataset_id):03d}_LIOM6p5",
        "plans_id": os.environ.get("LIOM_VSEG_PLANS_ID") or "nnUNetPlans",
        "fold_all": os.environ.get("LIOM_VSEG_FOLD_ALL") or "1",
    }


def _local_path(raw: str) -> Path:
    """Resolve a ``LIOM_VSEG_*`` local path (absolute verbatim, else repo-root-relative)."""
    p = Path(raw)
    return p if p.is_absolute() else _REPO_ROOT / p


def _prelude(env: dict[str, Any]) -> str:
    """Shared strict-mode prelude: ``cd`` repo, export nnU-Net + LIOM_VSEG env.

    Every remote stage is a fresh ``bash -s`` shell, so the exports are
    re-emitted per stage. ``nnUNet_extTrainer`` points at the repo's
    ``segmentation`` directory so ``mp.spawn`` children can resolve the custom
    trainer class names (module-attribute registration does not exist in
    spawned interpreters). It must point at ``segmentation`` rather than
    ``vseg`` itself: nnU-Net imports every ``.py`` in the directory as a
    top-level module, and pointing it at ``vseg`` crashes on sibling modules'
    relative imports (``from .utils import ...``) while also letting
    ``vseg/ssl/`` shadow the stdlib ``ssl`` package. Scanning the parent lets
    the finder recurse into ``vseg`` as a package, where relative imports
    resolve normally.
    """
    return textwrap.dedent(
        f"""\
        set -euo pipefail
        cd {shlex.quote(env["repo"])}

        # nnU-Net v2 env vars MUST be exported before any nnU-Net call —
        # unset vars fail opaquely downstream (validate_nnunet_env raises
        # RuntimeError naming them; there is no silent fallback).
        export nnUNet_raw="${{nnUNet_raw:-/data/nnUNet_raw}}"
        export nnUNet_preprocessed="${{nnUNet_preprocessed:-/data/nnUNet_preprocessed}}"
        export nnUNet_results="${{nnUNet_results:-/data/nnUNet_results}}"
        export nnUNet_extTrainer="$PWD/liom_toolkit/segmentation"

        export LIOM_VSEG_NGPUS={shlex.quote(env["ngpus"])}
        export LIOM_VSEG_DS_ID={shlex.quote(env["dataset_id"])}
        export LIOM_VSEG_DS_NAME={shlex.quote(env["dataset_name"])}
        export LIOM_VSEG_PLANS={shlex.quote(env["plans_id"])}
        export LIOM_VSEG_RUN_DIR={shlex.quote(env["run_dir"])}
        export LIOM_VSEG_PRETRAINED={shlex.quote(env["pretrained"])}
        RUN_DIR="$LIOM_VSEG_RUN_DIR"
        mkdir -p "$RUN_DIR"
        """
    )


def _verify_splits_fn() -> str:
    """Bash function that re-verifies ``splits_final.json`` in-process.

    A missing or malformed splits file triggers nnU-Net's random-split
    fallback — the intra-brain leak this phase exists to close. Every arm
    stage runs this BEFORE any ``run_training`` call; the check exits
    non-zero naming the gap.
    """
    return textwrap.dedent(
        """\
        verify_splits() {
            uv run python - <<'PY'
        import json
        import os
        from pathlib import Path

        from liom_toolkit.scripts.liom_prepare_nnunet_dataset import verify_loo_splits

        ds = os.environ["LIOM_VSEG_DS_NAME"]
        manifest = Path(os.environ["nnUNet_raw"]) / ds / "case_brain_manifest.json"
        splits_path = Path(os.environ["nnUNet_preprocessed"]) / ds / "splits_final.json"
        if not splits_path.is_file():
            raise SystemExit(
                f"splits_final.json missing: {splits_path} — nnU-Net would "
                "silently fall back to a random split (the intra-brain leak)"
            )
        verify_loo_splits(json.loads(splits_path.read_text()), manifest)
        print("LIOM_VSEG_SPLITS_VERIFIED folds=2")
        PY
        }
        """
    )


def _run_stage(host: str, user: str, script: str, *, timeout: int) -> str:
    """Pipe ``script`` to ``ssh <host> bash -s``; assert exit 0; return stdout.

    ``check=False`` so a non-zero exit surfaces as a test failure carrying
    the tail of remote stdout/stderr, not a bare subprocess exception.
    """
    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - controlled ssh invocation
        [
            _SSH,
            "-o",
            "ConnectTimeout=10",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"{user}@{host}",
            "bash -s",
        ],
        input=script.encode(),
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"remote stage exited {proc.returncode}.\n"
        f"--- remote stdout ---\n{proc.stdout.decode(errors='replace')[-4000:]}\n"
        f"--- remote stderr ---\n{proc.stderr.decode(errors='replace')[-8000:]}"
    )
    return proc.stdout.decode(errors="replace")


def _arm_script(
    env: dict[str, Any],
    stage: int,
    trainer: str,
    sentinel: str,
    *,
    gate: str = "",
) -> str:
    """Build the per-arm training script (assert splits, loop folds 0/1).

    ``LIOM_VSEG_PRETRAINED_ARG`` is the warm-start path for arm 3 and empty
    otherwise — the heredoc maps empty → ``None`` so arms 1/2 train
    from scratch. ``gate`` is an optional bash fragment inserted between
    the splits verification and the fold loop (stage 40's
    plans-architecture gate).
    """
    pretrained_arg = env["pretrained"] if trainer == _ARM_TRAINERS["warmstart"] else ""
    return (
        _prelude(env)
        + textwrap.dedent(
            f"""\
            export LIOM_VSEG_TRAINER={shlex.quote(trainer)}
            export LIOM_VSEG_PRETRAINED_ARG={shlex.quote(pretrained_arg)}

            test -f "$RUN_DIR/.done_stage10" || {{
                echo "stage {stage} requires .done_stage10 — run stage10 first" >&2
                exit 3
            }}
            """
        )
        + _verify_splits_fn()
        + "\nverify_splits\n"
        + gate
        + textwrap.dedent(
            """\

            RESULTS_BASE="$nnUNet_results/$LIOM_VSEG_DS_NAME/${LIOM_VSEG_TRAINER}__${LIOM_VSEG_PLANS}__2d"
            for FOLD in 0 1; do
                FD="$RESULTS_BASE/fold_$FOLD"
                if [ -f "$FD/checkpoint_final.pth" ] \\
                    && [ -d "$FD/validation" ] \\
                    && [ -n "$(ls -A "$FD/validation" 2>/dev/null)" ]; then
                    echo "LIOM_VSEG_SKIP_FOLD $FOLD (checkpoint + validation/ already present)"
                    continue
                fi
                export LIOM_VSEG_FOLD="$FOLD"
                # The training entry point must be a REAL FILE, not a stdin
                # heredoc: nnU-Net's validation export pool hardcodes a spawn
                # context and the dataloader workers spawn too, and spawned
                # children re-import __main__ as __mp_main__ — '<stdin>' is not
                # a file, so a heredoc crashes every worker. The __main__
                # guard keeps the re-import inert in spawned children.
                TRAIN_PY="$RUN_DIR/_liom_train_fold.py"
                cat > "$TRAIN_PY" <<'PYEOF'
            import multiprocessing
            import os


            def _main() -> None:
                # Fork-after-CUDA deadlocks the nnU-Net dataloader workers;
                # spawn each worker cleanly instead.
                multiprocessing.set_start_method("spawn", force=True)

                import torch

                from nnunetv2.run.run_training import run_training

                run_training(
                    os.environ["LIOM_VSEG_DS_NAME"],
                    "2d",
                    int(os.environ["LIOM_VSEG_FOLD"]),
                    trainer_class_name=os.environ["LIOM_VSEG_TRAINER"],
                    plans_identifier=os.environ["LIOM_VSEG_PLANS"],
                    pretrained_weights=os.environ.get("LIOM_VSEG_PRETRAINED_ARG") or None,
                    num_gpus=int(os.environ["LIOM_VSEG_NGPUS"]),
                    export_validation_probabilities=False,
                    device=torch.device("cuda"),
                )


            if __name__ == "__main__":
                _main()
            PYEOF
                uv run python "$TRAIN_PY"
                test -f "$FD/checkpoint_final.pth" || {
                    echo "fold $FOLD ended without checkpoint_final.pth — trainer crashed" >&2
                    exit 5
                }
            done
            """
        )
        + textwrap.dedent(
            f"""\
            touch "$RUN_DIR/.done_stage{stage}"
            echo "{sentinel}"
            """
        )
    )


def test_stage00_remote_probe():
    """Probe the lab box: SSH ``echo ok`` + ``nvidia-smi`` GPU count.

    Skips cleanly on an unreachable host or a box with 0 CUDA GPUs —
    environment conditions, never regressions. Passes when the box answers
    and reports at least one GPU.
    """
    pytest.importorskip("torch")

    host, user, _remote_home = _connect()
    _require_gpus(host, user)


def test_stage10_dataset_and_splits():
    """Remote stage 10: repo sync → brain-mapped Dataset101 rebuild → verified LOO splits.

    Syncs the remote checkout (``LIOM_VSEG_REF`` checkout or
    ``git pull --ff-only``), fails fast naming the missing file when the
    checkout is stale (missing ``nnunet_trainer.py`` / ``gate.py`` /
    ``write_loo_splits``), then rebuilds Dataset101 from the per-brain
    labeled-slice dirs, runs ``nnUNetv2_plan_and_preprocess``, writes +
    re-verifies the 2-fold ``splits_final.json``, and records the resolved
    ``network_class_name`` for the arm-3 architecture gate. Idempotent:
    ``.done_stage10`` + verified splits → skip sentinel, no rebuild.
    """
    pytest.importorskip("torch")

    host, user, remote_home = _connect()
    _require_gpus(host, user)
    env = _remote_env(remote_home)

    dirs_cli = " \\\n                ".join(shlex.quote(d) for d in env["dataset_dirs"])
    script = (
        _prelude(env)
        + textwrap.dedent(
            f"""\
            export LIOM_VSEG_REF={shlex.quote(env["ref"])}
            export LIOM_VSEG_INPUT_DIRS={shlex.quote(os.pathsep.join(env["dataset_dirs"]))}

            # Sync to the pushed ref when pinned, else fast-forward — then
            # verify the checkout actually carries the Phase-16 modules (a
            # stale remote trains the WRONG code; the post-sync file
            # assertions are the only guarantee).
            if [ -n "$LIOM_VSEG_REF" ]; then
                git fetch origin
                git checkout "$LIOM_VSEG_REF"
            else
                git pull --ff-only
            fi
            for f in \\
                liom_toolkit/segmentation/vseg/nnunet_trainer.py \\
                liom_toolkit/segmentation/vseg/gate.py; do
                if [ ! -f "$f" ]; then
                    echo "stale remote checkout — required module missing: $f" >&2
                    exit 3
                fi
            done
            if ! grep -q "def write_loo_splits" \\
                liom_toolkit/scripts/liom_prepare_nnunet_dataset.py; then
                echo "stale remote checkout — write_loo_splits missing from" \\
                    "liom_prepare_nnunet_dataset.py" >&2
                exit 3
            fi
            uv sync --extra ai --extra benchmark --extra seg

            """
        )
        + _verify_splits_fn()
        + textwrap.dedent(
            """\
            SPLITS="$nnUNet_preprocessed/$LIOM_VSEG_DS_NAME/splits_final.json"
            if [ -f "$RUN_DIR/.done_stage10" ] && [ -f "$SPLITS" ]; then
                verify_splits
                echo "LIOM_VSEG_SKIP_STAGE10 (dataset + verified splits already present)"
                echo "LIOM_VSEG_DATASET_OK"
                exit 0
            fi

            # The anonymous-case dataset is being REPLACED with the
            # brain-mapped rebuild — the removal is explicit and logged,
            # never silent.
            echo "replacing anonymous-case dataset at $nnUNet_raw/$LIOM_VSEG_DS_NAME"
            rm -rf "$nnUNet_raw/$LIOM_VSEG_DS_NAME" \\
                   "$nnUNet_preprocessed/$LIOM_VSEG_DS_NAME"
            """
        )
        + f"""
            uv run liom-prepare-nnunet-dataset \\
                {dirs_cli} \\
                "$nnUNet_raw/$LIOM_VSEG_DS_NAME" --dataset-id "$LIOM_VSEG_DS_ID"
        """
        + textwrap.dedent(
            """\

            # Every discovered pair must have landed in the manifest — the
            # case→brain mapping is the split's audit record. Require each
            # brain ≥ 1 case and total ≥ 4 (a 2-fold gate needs both brains
            # populated); exit naming the counts otherwise.
            uv run python - <<'PY'
            import json
            import os
            from pathlib import Path

            ds = os.environ["LIOM_VSEG_DS_NAME"]
            manifest = json.loads(
                (Path(os.environ["nnUNet_raw"]) / ds / "case_brain_manifest.json").read_text()
            )
            dirs = os.environ["LIOM_VSEG_INPUT_DIRS"].split(os.pathsep)
            counts = {}
            for d in dirs:
                imgs = [f for f in Path(d).glob("*.png") if not f.name.endswith("_mask.png")]
                counts[Path(d).name] = len(imgs)
            per_brain = {}
            for brain in manifest.values():
                per_brain[brain] = per_brain.get(brain, 0) + 1
            counts_str = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print("LIOM_VSEG_CASE_COUNTS " + counts_str)
            for brain, n in sorted(counts.items()):
                if n < 1:
                    raise SystemExit(f"brain {brain} contributed 0 cases from {dirs}")
                if per_brain.get(brain, 0) != n:
                    raise SystemExit(
                        f"case-count mismatch for {brain}: discovered {n} pairs "
                        f"but manifest records {per_brain.get(brain, 0)}"
                    )
            total = sum(counts.values())
            if total < 4:
                raise SystemExit(f"total case count {total} < 4 — too few slices for a 2-fold gate")
            PY

            uv run nnUNetv2_plan_and_preprocess -d "$LIOM_VSEG_DS_ID"

            # Verified 2-fold LOO splits — write_loo_splits runs
            # verify_loo_splits BEFORE writing; the re-read + verify below
            # catches any post-write corruption (nnU-Net only warns on
            # overlap — this check is the gate).
            uv run python - <<'PY'
            import json
            import os
            from pathlib import Path

            from liom_toolkit.scripts.liom_prepare_nnunet_dataset import (
                verify_loo_splits,
                write_loo_splits,
            )

            ds = os.environ["LIOM_VSEG_DS_NAME"]
            manifest = Path(os.environ["nnUNet_raw"]) / ds / "case_brain_manifest.json"
            out = Path(os.environ["nnUNet_preprocessed"]) / ds / "splits_final.json"
            write_loo_splits(manifest, str(out))
            verify_loo_splits(json.loads(out.read_text()), manifest)
            print(f"LIOM_VSEG_SPLITS_WRITTEN {out}")
            PY
            verify_splits

            # Record the resolved 2d architecture for the arm-3 gate — a
            # ResEnc-plans rebuild would crash load_pretrained_weights hours
            # into arm 3, so stage 40 fails fast on a non-PlainConvUNet name.
            uv run python - <<'PY'
            import json
            import os
            from pathlib import Path

            ds = os.environ["LIOM_VSEG_DS_NAME"]
            plans_path = (
                Path(os.environ["nnUNet_preprocessed"]) / ds
                / f"{os.environ['LIOM_VSEG_PLANS']}.json"
            )
            plans = json.loads(plans_path.read_text())
            arch = plans["configurations"]["2d"]["architecture"]["network_class_name"]
            out = Path(os.environ["LIOM_VSEG_RUN_DIR"]) / "plans_arch.txt"
            out.write_text(arch + "\\n")
            print(f"LIOM_VSEG_PLANS_ARCH {arch}")
            PY

            touch "$RUN_DIR/.done_stage10"
            echo "LIOM_VSEG_DATASET_OK"
            """
        )
    )

    # Dataset rebuild + fingerprint/planning + preprocessing can take a
    # couple of hours on a cold box; the timeout is generous so a slow run
    # does not flake.
    stdout = _run_stage(host, user, script, timeout=7200)
    assert "LIOM_VSEG_DATASET_OK" in stdout, (
        f"stage 10 did not reach the verification echo; stdout:\n{stdout[-4000:]}"
    )
    assert "LIOM_VSEG_CASE_COUNTS" in stdout or "LIOM_VSEG_SKIP_STAGE10" in stdout, (
        f"stage 10 produced no case-count record; stdout:\n{stdout[-2000:]}"
    )


def test_stage20_arm1_baseline():
    """Remote stage 20: baseline arm — ``nnUNetTrainer_50epochs`` folds 0/1.

    Asserts ``.done_stage10`` + re-verifies ``splits_final.json`` BEFORE any
    training call (a missing/malformed file must stop the stage — nnU-Net's
    random fallback is the leak this phase exists to close). Per fold,
    skips when ``checkpoint_final.pth`` + a non-empty ``validation/``
    already exist; otherwise trains in-process via ``run_training`` under
    the spawn start method.
    """
    pytest.importorskip("torch")

    host, user, remote_home = _connect()
    _require_gpus(host, user)
    env = _remote_env(remote_home)

    script = _arm_script(env, 20, _ARM_TRAINERS["baseline"], "LIOM_VSEG_ARM1_OK")

    # Two 50-epoch folds on real data — the timeout is generous so a slow
    # box or single-GPU fallback does not flake.
    stdout = _run_stage(host, user, script, timeout=28800)
    assert "LIOM_VSEG_ARM1_OK" in stdout, (
        f"stage 20 did not reach the verification echo; stdout:\n{stdout[-4000:]}"
    )


def test_stage30_arm2_cldice():
    """Remote stage 30: custom-loss arm — ``LiomDiceFocalClDiceTrainer`` folds 0/1.

    Same splits-first + per-fold-skip protocol as stage 20, with the
    composite Dice-Focal + soft-clDice trainer resolved via
    ``nnUNet_extTrainer`` (required under DDP spawn).
    """
    pytest.importorskip("torch")

    host, user, remote_home = _connect()
    _require_gpus(host, user)
    env = _remote_env(remote_home)

    script = _arm_script(env, 30, _ARM_TRAINERS["cldice"], "LIOM_VSEG_ARM2_OK")

    stdout = _run_stage(host, user, script, timeout=28800)
    assert "LIOM_VSEG_ARM2_OK" in stdout, (
        f"stage 30 did not reach the verification echo; stdout:\n{stdout[-4000:]}"
    )


def test_stage40_arm3_warmstart():
    """Remote stage 40: warm-start arm — custom loss + SSL pretrained weights.

    Two protocol gates BEFORE any training call: the plans-architecture
    check (``plans_arch.txt`` must name ``PlainConvUNet`` — a ResEnc-plans
    rebuild crashes ``load_pretrained_weights`` hours into the run, so the
    stage FAILS naming the resolved architecture rather than launching) and
    the pretrained-checkpoint existence check. Then the same splits-first +
    per-fold-skip loop with ``pretrained_weights=$LIOM_VSEG_PRETRAINED``.
    """
    pytest.importorskip("torch")

    host, user, remote_home = _connect()
    _require_gpus(host, user)
    env = _remote_env(remote_home)

    arch_gate = textwrap.dedent(
        """\
        # Arm-3 plans-architecture gate: the pretrained checkpoint was built
        # against the PlainConvUNet architecture; a plans rebuild that
        # resolved something else cannot accept its weights. Fail loudly
        # naming the mismatch — never launch, never skip.
        if [ ! -f "$RUN_DIR/plans_arch.txt" ]; then
            echo "plans_arch.txt missing — stage 10 did not record the resolved architecture" >&2
            exit 4
        fi
        ARCH=$(cat "$RUN_DIR/plans_arch.txt")
        if ! grep -q "PlainConvUNet" "$RUN_DIR/plans_arch.txt"; then
            echo "arm-3 plans-architecture gate FAILED: resolved network_class_name is" >&2
            echo "  ${ARCH} — expected PlainConvUNet for the pretrained checkpoint;" >&2
            echo "  refusing to launch arm 3 (pretrained keys would not match)" >&2
            exit 4
        fi
        if [ ! -f "$LIOM_VSEG_PRETRAINED" ]; then
            echo "pretrained checkpoint missing on remote: $LIOM_VSEG_PRETRAINED" >&2
            exit 4
        fi

        """
    )
    script = _arm_script(env, 40, _ARM_TRAINERS["warmstart"], "LIOM_VSEG_ARM3_OK", gate=arch_gate)

    stdout = _run_stage(host, user, script, timeout=28800)
    assert "LIOM_VSEG_ARM3_OK" in stdout, (
        f"stage 40 did not reach the verification echo; stdout:\n{stdout[-4000:]}"
    )


def test_stage50_gate_eval():
    """Remote stage 50: gate eval — per-arm × per-fold scoring → ship verdict.

    LOCAL pre-registration gate FIRST: ``16-EVAL-RESULTS.md`` must exist and
    contain ``PRE-REGISTERED`` before any remote eval runs (the eval cannot
    run ahead of the committed gate table — post-hoc thresholds are
    indistinguishable from tuning-on-test). Remote: assert all six arm fold
    dirs carry ``checkpoint_final.pth`` + a non-empty ``validation/``,
    then score each arm × fold's validation predictions against
    ``gt_segmentations`` through ``score_prediction_set`` →
    ``evaluate_gate`` → ``ship_decision``, writing ``gate_results.json``
    and echoing it between ``LIOM_VSEG_GATE_JSON_BEGIN``/``END`` markers.
    The test parses the verdict, asserts its shape, and saves it to
    ``LIOM_VSEG_RESULTS_JSON``.
    """
    pytest.importorskip("torch")

    # --- LOCAL pre-registration gate (runs before any SSH) ----------------
    assert _EVAL_RESULTS_DOC.is_file(), (
        f"pre-registration document missing: {_EVAL_RESULTS_DOC} — the gate "
        "table must be committed before the eval runs"
    )
    doc_text = _EVAL_RESULTS_DOC.read_text()
    assert "PRE-REGISTERED" in doc_text, (
        f"{_EVAL_RESULTS_DOC} does not contain the PRE-REGISTERED marker — "
        "the eval cannot run ahead of the committed gate table"
    )

    host, user, remote_home = _connect()
    _require_gpus(host, user)
    env = _remote_env(remote_home)

    arm_map_lines = "\n".join(
        f'    "{name}": "{trainer}",' for name, trainer in _ARM_TRAINERS.items()
    )
    script = (
        _prelude(env)
        + textwrap.dedent(
            """\
            for s in 20 30 40; do
                test -f "$RUN_DIR/.done_stage$s" || {
                    echo "stage 50 requires .done_stage$s — run the arm stages first" >&2
                    exit 3
                }
            done

            uv run python - <<'PY'
            import json
            import os
            from pathlib import Path

            import numpy as np
            import SimpleITK as sitk
            from imageio.v3 import imread

            from liom_toolkit.segmentation.vseg.gate import (
                arm_passes,
                evaluate_gate,
                score_prediction_set,
                ship_decision,
            )

            res_dir = Path(os.environ["nnUNet_results"])
            pre_dir = Path(os.environ["nnUNet_preprocessed"])
            ds = os.environ["LIOM_VSEG_DS_NAME"]
            plans = os.environ["LIOM_VSEG_PLANS"]
            run_dir = Path(os.environ["LIOM_VSEG_RUN_DIR"])

            ARMS = {
            __ARM_MAP__
            }
            FOLDS = (0, 1)
            _EXTS = (".nii.gz", ".npz", ".png", ".nrrd", ".mha")

            def _stem(path):
                name = os.path.basename(path)
                for ext in _EXTS:
                    if name.endswith(ext):
                        return name[: -len(ext)]
                return os.path.splitext(name)[0]

            def _load_mask(path):
                if path.endswith(".png"):
                    return np.asarray(imread(path)) > 0
                if path.endswith(".npz"):
                    with np.load(path) as z:
                        arr = np.asarray(z[z.files[0]])
                    return arr.argmax(0) > 0 if arr.ndim > 2 else arr > 0
                return np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(path))) > 0

            # All six arm fold dirs must be complete BEFORE scoring — an
            # eval over partial results is a protocol violation, never a
            # partial verdict. Name every gap.
            missing = []
            for arm, trainer in ARMS.items():
                for f in FOLDS:
                    fd = res_dir / ds / f"{trainer}__{plans}__2d" / f"fold_{f}"
                    if not (fd / "checkpoint_final.pth").is_file():
                        missing.append(f"{arm} fold_{f}: missing {fd}/checkpoint_final.pth")
                    vdir = fd / "validation"
                    if not vdir.is_dir() or not any(vdir.iterdir()):
                        missing.append(f"{arm} fold_{f}: empty/missing {vdir}")
            if missing:
                raise SystemExit(
                    "incomplete arm fold dirs — refusing to eval partial results: "
                    + "; ".join(missing)
                )

            gt_dir = pre_dir / ds / "gt_segmentations"
            gt = {
                _stem(p): str(p)
                for p in sorted(gt_dir.iterdir())
                if p.is_file()
            }

            per_arm = {}
            for arm, trainer in ARMS.items():
                verdicts = []
                for f in FOLDS:
                    vdir = res_dir / ds / f"{trainer}__{plans}__2d" / f"fold_{f}" / "validation"
                    preds = {
                        _stem(p): str(p)
                        for p in sorted(vdir.iterdir())
                        if p.is_file() and _stem(p) not in ("summary", "progress")
                    }
                    cases = sorted(set(preds) & set(gt))
                    if not cases:
                        raise SystemExit(
                            f"{arm} fold_{f}: no overlapping validation cases "
                            f"between {vdir} and {gt_dir}"
                        )
                    pairs = [(_load_mask(preds[c]), _load_mask(gt[c])) for c in cases]
                    matrix = score_prediction_set(pairs)
                    verdicts.append((f, matrix, evaluate_gate(matrix, fold=f)))
                per_arm[arm] = verdicts

            decision = ship_decision(
                {arm: [fv for _, _, fv in vs] for arm, vs in per_arm.items()},
                baseline="baseline",
            )
            result = {
                "per_arm": {
                    arm: {
                        "gate_passed": arm_passes([fv for _, _, fv in vs]),
                        "folds": {
                            str(f): {
                                "passed": fv.passed,
                                "metrics": matrix,
                                "rows": [
                                    {
                                        "key": r.row.key,
                                        "direction": r.row.direction,
                                        "gating": r.row.gating,
                                        "measured": r.measured,
                                        "passed": r.passed,
                                        "detail": r.detail,
                                    }
                                    for r in fv.rows
                                ],
                            }
                            for f, matrix, fv in vs
                        },
                    }
                    for arm, vs in per_arm.items()
                },
                "verdict": {
                    "winner": decision.winner,
                    "ship": decision.ship,
                    "gate_passed": decision.gate_passed,
                    "improvements": decision.improvements,
                    "reasons": list(decision.reasons),
                },
            }
            out = run_dir / "gate_results.json"
            out.write_text(json.dumps(result, indent=2) + "\\n")
            print(f"LIOM_VSEG_GATE_WRITTEN {out}")
            PY

            echo "LIOM_VSEG_GATE_JSON_BEGIN"
            cat "$RUN_DIR/gate_results.json"
            echo "LIOM_VSEG_GATE_JSON_END"
            touch "$RUN_DIR/.done_stage50"
            echo "LIOM_VSEG_EVAL_OK"
            """
        )
    ).replace("__ARM_MAP__", arm_map_lines)

    stdout = _run_stage(host, user, script, timeout=7200)
    assert "LIOM_VSEG_EVAL_OK" in stdout, (
        f"stage 50 did not reach the verification echo; stdout:\n{stdout[-4000:]}"
    )

    # Extract + parse the verdict JSON between the markers — the ship
    # decision is consumed from the parsed payload, never from filenames.
    begin = stdout.find("LIOM_VSEG_GATE_JSON_BEGIN")
    end = stdout.find("LIOM_VSEG_GATE_JSON_END")
    assert begin != -1 and end != -1 and end > begin, (
        f"gate JSON markers missing from remote stdout:\n{stdout[-2000:]}"
    )
    payload = json.loads(stdout[begin + len("LIOM_VSEG_GATE_JSON_BEGIN") : end])
    verdict = payload.get("verdict") or {}
    assert verdict.get("ship") in (True, False), (
        f"verdict.ship has unexpected value {verdict.get('ship')!r}"
    )
    assert verdict.get("winner") in (*_ARM_TRAINERS, None), (
        f"verdict.winner has unexpected value {verdict.get('winner')!r}"
    )

    results_path = _local_path(os.environ.get("LIOM_VSEG_RESULTS_JSON") or _DEFAULT_RESULTS_JSON)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2) + "\n")


def test_stage60_export_ship():
    """Remote stage 60: ship export — optional fold_all → zip → sha256 → scp pull.

    Reads the local ``LIOM_VSEG_RESULTS_JSON``; a NO-SHIP verdict skips
    naming the verdict (exporting a model that failed the gate is
    meaningless). Remote: assert ``.done_stage50``, optionally train the
    winner's ``fold_all`` (``LIOM_VSEG_FOLD_ALL`` != ``0``, skip-if-done —
    the flag IS the "GPU time remains" knob), export
    ``-f 0 1 all`` when the fold_all checkpoint exists else ``-f 0 1``
    (the gated 2-fold ensemble ships either way), ``sha256sum`` →
    ``SHA256SUMS``. Test side: scp-pull the zip + sums into
    ``LIOM_VSEG_ARTIFACT_DIR``, verify the local hash matches the
    remote-printed one, and assert the zip carries ``dataset.json``,
    ``plans.json``, ``fold_0/``, ``fold_1/``.
    """
    pytest.importorskip("torch")

    results_path = _local_path(os.environ.get("LIOM_VSEG_RESULTS_JSON") or _DEFAULT_RESULTS_JSON)
    if not results_path.is_file():
        pytest.fail(f"gate results JSON missing: {results_path} — run test_stage50_gate_eval first")
    payload = json.loads(results_path.read_text())
    verdict = payload.get("verdict") or {}
    winner = verdict.get("winner")
    if not verdict.get("ship"):
        pytest.skip(f"NO-SHIP verdict — export blocked ({winner!r})")
    winner_trainer = _ARM_TRAINERS.get(winner)
    assert winner_trainer is not None, (
        f"verdict winner {winner!r} is not a known arm {sorted(_ARM_TRAINERS)}"
    )

    host, user, remote_home = _connect()
    _require_gpus(host, user)
    env = _remote_env(remote_home)

    # The winner's fold_all reproduces the winning RECIPE — the warm-start
    # arm's all-data run warm-starts too (a cold fold_all would not be the
    # gated arm's model).
    pretrained_flag = (
        f" -pretrained_weights {shlex.quote(env['pretrained'])}" if winner == "warmstart" else ""
    )
    script = _prelude(env) + textwrap.dedent(
        f"""\
            export LIOM_VSEG_WINNER={shlex.quote(winner)}
            export LIOM_VSEG_WINNER_TRAINER={shlex.quote(winner_trainer)}
            export LIOM_VSEG_FOLD_ALL={shlex.quote(env["fold_all"])}

            test -f "$RUN_DIR/.done_stage50" || {{
                echo "stage 60 requires .done_stage50 — run test_stage50_gate_eval first" >&2
                exit 3
            }}

            RESULTS_BASE="$nnUNet_results/$LIOM_VSEG_DS_NAME/${{LIOM_VSEG_WINNER_TRAINER}}__${{LIOM_VSEG_PLANS}}__2d"
            FD="$RESULTS_BASE/fold_all"
            if [ "$LIOM_VSEG_FOLD_ALL" != "0" ]; then
                if [ -f "$FD/checkpoint_final.pth" ]; then
                    echo "LIOM_VSEG_SKIP_FOLD_ALL (checkpoint_final.pth already present)"
                else
                    uv run nnUNetv2_train "$LIOM_VSEG_DS_NAME" 2d all \\
                        -tr "$LIOM_VSEG_WINNER_TRAINER" -p "$LIOM_VSEG_PLANS" \\
                        -num_gpus "$LIOM_VSEG_NGPUS"{pretrained_flag}
                fi
            fi

            FOLD_ARGS=(0 1)
            if [ -f "$FD/checkpoint_final.pth" ]; then
                FOLD_ARGS=(0 1 all)
            fi
            ZIP="$RUN_DIR/liom_vseg_${{LIOM_VSEG_WINNER}}_v1p2.zip"
            uv run nnUNetv2_export_model_to_zip -d "$LIOM_VSEG_DS_ID" -c 2d \\
                -tr "$LIOM_VSEG_WINNER_TRAINER" -p "$LIOM_VSEG_PLANS" \\
                -f "${{FOLD_ARGS[@]}}" -o "$ZIP"
            SHA=$(sha256sum "$ZIP" | awk '{{print $1}}')
            printf '%s  %s\\n' "$SHA" "$(basename "$ZIP")" > "$RUN_DIR/SHA256SUMS"
            touch "$RUN_DIR/.done_stage60"
            echo "LIOM_VSEG_EXPORT_OK $ZIP $SHA"
            """
    )

    # Optional fold_all adds a third 50-epoch run; the timeout covers it.
    stdout = _run_stage(host, user, script, timeout=21600)
    assert "LIOM_VSEG_EXPORT_OK" in stdout, (
        f"stage 60 did not reach the verification echo; stdout:\n{stdout[-4000:]}"
    )
    sentinel_line = next(
        (line for line in stdout.splitlines() if line.startswith("LIOM_VSEG_EXPORT_OK ")),
        None,
    )
    assert sentinel_line is not None, (
        f"LIOM_VSEG_EXPORT_OK sentinel missing from stdout:\n{stdout[-2000:]}"
    )
    _ok, remote_zip, remote_sha = sentinel_line.split()

    # Pull the zip + SHA256SUMS into the local artifact dir.
    artifact_dir = _local_path(os.environ.get("LIOM_VSEG_ARTIFACT_DIR") or _DEFAULT_ARTIFACT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _scp_pull(host, user, remote_zip, artifact_dir)
    _scp_pull(host, user, f"{env['run_dir']}/SHA256SUMS", artifact_dir)

    local_zip = artifact_dir / Path(remote_zip).name
    local_sha = hashlib.sha256(local_zip.read_bytes()).hexdigest()
    assert local_sha == remote_sha, (
        f"pulled zip hash mismatch: local {local_sha} != remote {remote_sha} "
        f"({local_zip}) — the artifact did not survive transport intact"
    )

    with zipfile.ZipFile(local_zip) as zf:
        names = zf.namelist()
    for member in ("dataset.json", "plans.json", "fold_0/", "fold_1/"):
        assert any(member in n for n in names), (
            f"exported zip is missing {member}: members = {names[:20]}"
        )
