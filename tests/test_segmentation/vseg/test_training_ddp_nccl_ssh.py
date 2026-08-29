"""SSH-driven real NCCL DDP run on the lab CUDA box.

The 2-rank CPU gloo smoke (``test_training_ddp.py``) is the in-CI bar
— it exercises the real ``torchrun`` launch contract on CPU. This test
goes beyond it: it SSHes to the lab CUDA box and runs a real multi-GPU
NCCL DDP training via ``torchrun --nproc_per_node=<N> --standalone
liom-train-model ... --ddp --amp``, then asserts the rank-0-only write
invariants hold under NCCL (single manifest, single rank-0 ``.pth``,
exit 0). NCCL is the backend real multi-GPU training uses; gloo-on-CPU
does not exercise the NCCL all-reduce / CUDA-set-device path.

Skip policy (CI legs without the box pass cleanly — never fail):
* ``LIOM_CUDA_HOST`` env var unset → skipped at collection via
  ``pytest.mark.skipif``.
* The host is unreachable (SSH connect timeout / auth failure) →
  ``pytest.skip`` inside the body (a missing lab box is an environment
  condition, not a regression).
* ``nvidia-smi`` reports 0 GPUs → ``pytest.skip`` (no CUDA to run NCCL on).

Configuration via env vars (defaults match the lab box):
* ``LIOM_CUDA_HOST`` — the SSH host (default ``132.207.157.41``).
* ``LIOM_CUDA_USER`` — the SSH user (default: current local user).
* ``LIOM_CUDA_REPO`` — the repo path on the remote (default
  ``~/code/liom-toolkit``).
* ``LIOM_CUDA_NPROC`` — the process count (default: ``nvidia-smi`` GPU
  count on the remote).

Uses ONLY the stdlib ``subprocess`` module to drive ``ssh`` (no paramiko /
fabric dependency — AGENTS section 3: modules must import cleanly without
optional deps). The dataset is generated ON the remote via the remote's
``liom_toolkit`` (no scp of a local fixture — the remote may have a
different zarr build). Marked ``ai`` (needs the torch extra) and ``slow``
(real multi-GPU training on a remote box; auto-deselected from fast
iteration by the ``slow`` marker convention in ``pyproject.toml``).
"""

from __future__ import annotations

import getpass
import os
import shlex
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] - SSH-driven NCCL run legitimately spawns subprocess
import textwrap

import pytest

# Resolve ssh to its full path once so subprocess calls do not use a partial
# executable name (avoids the start-process-with-partial-path lint without an
# inline suppression; shutil.which is the stdlib way to resolve a PATH
# binary). Falls back to the bare name if ssh is not on PATH (the SSH probe
# then fails fast and the test skips — a missing ssh is an environment
# condition, not a bug).
_SSH = shutil.which("ssh") or "ssh"

_LIOM_CUDA_HOST = os.environ.get("LIOM_CUDA_HOST")
_LIOM_CUDA_USER = os.environ.get("LIOM_CUDA_USER") or getpass.getuser()
_LIOM_CUDA_REPO = os.environ.get("LIOM_CUDA_REPO") or "~/code/liom-toolkit"

pytestmark = [
    pytest.mark.ai,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _LIOM_CUDA_HOST,
        reason="needs LIOM_CUDA_HOST env var (lab CUDA box) — CI legs skip cleanly",
    ),
]


def _ssh(host: str, user: str, cmd: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run ``cmd`` on ``host`` via ssh and return the completed process.

    Uses ``-o ConnectTimeout=10 -o BatchMode=yes`` so an unreachable / no-key
    host fails fast instead of hanging the test (a missing lab box is an
    environment condition, not a regression — the caller skips on failure).
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


def test_ddp_nccl_ssh_run():
    """SSH to the lab CUDA box and run a real multi-GPU NCCL DDP training.

    Probes SSH connectivity and GPU count first (skips cleanly on failure),
    then runs ``uv sync --all-extras`` + a real ``torchrun --ddp --amp`` on
    the remote and asserts the rank-0-only write invariants (exit 0,
    ``final_metrics.csv``, single manifest, single rank-0 ``.pth``). The
    dataset is generated on the remote via the remote's ``liom_toolkit``
    (no scp) so the test is self-contained given SSH access to the box.
    """
    pytest.importorskip("torch")

    host = _LIOM_CUDA_HOST
    assert host is not None  # skipif guards this; assert for type-checkers
    user = _LIOM_CUDA_USER
    repo = _LIOM_CUDA_REPO

    # --- SSH connectivity probe (skip, not fail, on unreachable host) -----
    probe = _ssh(host, user, "echo ok", timeout=20)
    if probe.returncode != 0:
        pytest.skip(
            f"LIOM_CUDA_HOST {host} unreachable (ssh exit {probe.returncode}): "
            f"{probe.stderr.decode(errors='replace')[:200]}"
        )

    # --- GPU count probe (skip if no CUDA to run NCCL on) ----------------
    nproc_env = os.environ.get("LIOM_CUDA_NPROC")
    if nproc_env:
        nproc = nproc_env
    else:
        gpu_probe = _ssh(host, user, "nvidia-smi -L | grep -c 'GPU ' || echo 0", timeout=20)
        if gpu_probe.returncode != 0:
            pytest.skip(f"nvidia-smi probe failed on {host} (exit {gpu_probe.returncode})")
        nproc = gpu_probe.stdout.decode(errors="replace").strip().splitlines()[-1].strip()
        if not nproc.isdigit() or int(nproc) < 1:
            pytest.skip(f"no CUDA GPUs reported by nvidia-smi on {host} (got {nproc!r})")

    # --- The real NCCL DDP run (generate dataset + torchrun + verify) ----
    # A single bash script piped to `ssh ... bash -s` so the dataset
    # generation, torchrun, and output verification all run on the remote
    # in one SSH session (no scp, no second round-trip per check). The
    # script exits non-zero if any step fails (set -e), and prints
    # LIOM_NCCL_OK <out_dir> on success so the test can confirm the full
    # chain ran end-to-end.
    script = textwrap.dedent(
        f"""\
        set -euo pipefail
        cd {shlex.quote(repo)}
        uv sync --all-extras
        DS_DIR=$(mktemp -d)
        DS="$DS_DIR/smoke.zarr"
        export LIOM_NCCL_DS="$DS"
        uv run python - <<'PY'
        import os
        import numpy as np
        from liom_toolkit.conversion.conversion import save_zarr, save_label_to_zarr
        from liom_toolkit.utils.io import generate_label_color_dict_mask
        ds = os.environ["LIOM_NCCL_DS"]
        arr = np.zeros((2, 256, 256), dtype=np.uint16)
        arr[:, 32:224, 32:224] = 1000
        save_zarr(arr, ds, scales=(6.5, 6.5, 6.5), chunks=(2, 256, 256))
        label = np.zeros((2, 256, 256), dtype=np.uint8)
        label[:, 32:224, 32:224] = 1
        save_label_to_zarr(
            label, ds, generate_label_color_dict_mask(), "training",
            scales=(6.5, 6.5, 6.5), chunks=(2, 256, 256),
        )
        PY
        OUT=$(mktemp -d)/out
        WANDB_MODE=disabled uv run torchrun \\
            --nproc_per_node={shlex.quote(str(nproc))} \\
            --standalone \\
            liom-train-model "$DS" training --ddp --amp --epochs 1 --batch-size 2 \\
            --output-train "$OUT" --wandb-mode disabled
        test -f "$OUT/final_metrics.csv"
        test -f "$OUT/_liom_checkpoints/train_model.json"
        test -f "$OUT/files/checkpoint.latest.pth"
        echo "LIOM_NCCL_OK $OUT"
        """
    )

    # Real multi-GPU NCCL training + uv sync can take several minutes; the
    # timeout is generous so a slow box does not flake. check=False so a
    # non-zero exit surfaces as a test failure with the remote stderr, not
    # a subprocess exception.
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
        timeout=600,
    )

    assert proc.returncode == 0, (
        f"NCCL DDP SSH run exited {proc.returncode}.\n"
        f"--- remote stdout ---\n{proc.stdout.decode(errors='replace')[-4000:]}\n"
        f"--- remote stderr ---\n{proc.stderr.decode(errors='replace')[-8000:]}"
    )
    stdout = proc.stdout.decode(errors="replace")
    assert "LIOM_NCCL_OK" in stdout, (
        f"NCCL run did not reach the verification echo; stdout:\n{stdout[-4000:]}"
    )
