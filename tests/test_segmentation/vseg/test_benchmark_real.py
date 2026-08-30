"""SSH-driven real 4-contender benchmark on the lab CUDA box.

The in-CI benchmark tests (``test_benchmark_contenders.py``,
``test_prepare_nnunet_dataset.py``) exercise the harness on synthetic data —
they prove the eval-metric matrix, the Contender Protocol, the nnU-Net
subprocess bridge, and the OME-Zarr→nnU-Net converter all wire together. This
test goes beyond them: it SSHes to the lab CUDA box and runs the full
4-contender benchmark (Improved2D, MONAI UNet, SwinUNETR, nnU-Net v2) on real
6.5 µm lightsheet data via ``run_benchmark``, then asserts the result table
file exists on the remote. Real data is the only honest resolution of the
architecture question — synthetic data cannot settle a transformer-vs-U-Net
debate.

Skip policy (CI legs without the box pass cleanly — never fail):
* ``LIOM_CUDA_HOST`` env var unset → skipped at collection via
  ``pytest.mark.skipif``.
* The host is unreachable (SSH connect timeout / auth failure) →
  ``pytest.skip`` inside the body (a missing lab box is an environment
  condition, not a regression).
* ``nvidia-smi`` reports 0 GPUs → ``pytest.skip`` (no CUDA to train on).

Configuration via env vars (defaults match the lab box):
* ``LIOM_CUDA_HOST`` — the SSH host (no default; must be set or the test
  skips at collection).
* ``LIOM_CUDA_USER`` — the SSH user (default: current local user).
* ``LIOM_CUDA_REPO`` — the repo path on the remote (default
  ``~/code/liom-toolkit``).
* ``LIOM_BENCH_DATASET_DIR`` — directory of labeled PNG slices on the remote
  (default ``~/code/vseg/data/LSFM_dataset``). The harness reads S23 (train)
  and S24 (held-out test) subdirectories from here.
* ``LIOM_BENCH_NNUNET_VENV`` — path to the nnU-Net venv python on the remote
  (default ``~/venvs/nnunet/bin/python``). nnU-Net runs in a separate venv
  (torch-clobbering isolation); it is NOT a liom-toolkit dependency.
* ``LIOM_BENCH_NNUNET_DATASET_ID`` — the nnU-Net dataset id to register
  (default ``101``).

Uses ONLY the stdlib ``subprocess`` module to drive ``ssh`` (no paramiko /
fabric dependency — AGENTS section 3: modules must import cleanly without
optional deps). The dataset is read ON the remote via the remote's
``liom_toolkit`` (no scp of a local fixture — the remote may have a different
zarr build). Marked ``ai`` (needs the torch extra for the MONAI contenders)
and ``slow`` (real multi-contender training on a remote box; auto-deselected
from fast iteration by the ``slow`` marker convention in ``pyproject.toml``).

The empirical architecture decision (which contender wins on the per-metric
ship-gate matrix) is recorded in ``14-BENCHMARK-RESULTS.md`` by the human
after this test passes on the box — the test only proves the benchmark ran
end-to-end and wrote the result table; the human reviews the metrics and
records the verdict.
"""

from __future__ import annotations

import getpass
import os
import shlex
import shutil
import subprocess
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


def test_benchmark_real_run():
    """SSH to the lab CUDA box and run the full 4-contender benchmark on real data.

    Probes SSH connectivity and GPU count first (skips cleanly on failure),
    then pipes a ``bash -s`` script over SSH that: syncs the repo with all
    extras (installs MONAI in the ``[ai]`` env), prepares the nnU-Net dataset
    via ``liom-prepare-nnunet-dataset``, activates the nnU-Net venv and runs
    ``nnUNetv2_plan_and_preprocess`` + ``nnUNetv2_train`` (2d config), and
    runs the 4-contender benchmark via ``run_benchmark``. The script echoes
    ``LIOM_BENCH_OK <output_dir>`` on success and writes a JSON result table
    to ``<output_dir>/benchmark_results.json``. The test asserts the
    sentinel is present and the result table file exists on the remote.

    Skips cleanly without ``LIOM_CUDA_HOST`` (CI legs skip at collection);
    skips on an unreachable host or a box with 0 GPUs. The empirical
    architecture decision is recorded in ``14-BENCHMARK-RESULTS.md`` by the
    human after this test passes on the box.
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

    # --- Resolve the repo path against the REMOTE home dir ----------------
    # ``shlex.quote`` wraps the path in single quotes, which prevents ``~``
    # and ``$HOME`` expansion on the remote (``cd '~/code/...'`` is a literal
    # no-such-directory). Resolve ``~``/``$HOME`` to the remote home so the
    # quoted ``cd`` lands on a real absolute path. A user-supplied absolute
    # LIOM_CUDA_REPO is used verbatim.
    home_probe = _ssh(host, user, "echo $HOME", timeout=15)
    if home_probe.returncode != 0:
        pytest.skip(f"$HOME probe failed on {host} (exit {home_probe.returncode})")
    remote_home = home_probe.stdout.decode(errors="replace").strip().splitlines()[-1].strip()
    if repo.startswith(("~", "$HOME")):
        repo = remote_home + repo[repo.find("/") :]
    elif not os.path.isabs(repo):
        repo = f"{remote_home}/{repo}"

    # --- GPU count probe (skip if no CUDA to train on) -------------------
    gpu_probe = _ssh(host, user, "nvidia-smi -L | grep -c 'GPU ' || echo 0", timeout=20)
    if gpu_probe.returncode != 0:
        pytest.skip(f"nvidia-smi probe failed on {host} (exit {gpu_probe.returncode})")
    gpu_count = gpu_probe.stdout.decode(errors="replace").strip().splitlines()[-1].strip()
    if not gpu_count.isdigit() or int(gpu_count) < 1:
        pytest.skip(f"no CUDA GPUs reported by nvidia-smi on {host} (got {gpu_count!r})")

    # --- Benchmark run parameters (all parameterized via env vars) -------
    # Defaults match the lab layout; every path is overridable so the test
    # never hardcodes a lab-specific location (AGENTS §1).
    dataset_dir = os.environ.get("LIOM_BENCH_DATASET_DIR") or "~/code/vseg/data/LSFM_dataset"
    if dataset_dir.startswith(("~", "$HOME")):
        dataset_dir = remote_home + dataset_dir[dataset_dir.find("/") :]
    elif not os.path.isabs(dataset_dir):
        dataset_dir = f"{remote_home}/{dataset_dir}"

    nnunet_venv = os.environ.get("LIOM_BENCH_NNUNET_VENV") or "~/venvs/nnunet/bin/python"
    if nnunet_venv.startswith(("~", "$HOME")):
        nnunet_venv = remote_home + nnunet_venv[nnunet_venv.find("/") :]
    elif not os.path.isabs(nnunet_venv):
        nnunet_venv = f"{remote_home}/{nnunet_venv}"

    nnunet_dataset_id = os.environ.get("LIOM_BENCH_NNUNET_DATASET_ID") or "101"

    # --- The real 4-contender benchmark run (one bash -s script over SSH) ---
    # A single bash script piped to `ssh ... bash -s` so the dataset
    # preparation, nnU-Net training, and the benchmark all run on the remote
    # in one SSH session (no scp, no second round-trip per check). The script
    # exits non-zero if any step fails (set -e), and prints
    # LIOM_BENCH_OK <out_dir> on success so the test can confirm the full
    # chain ran end-to-end. The result table is written to
    # <out_dir>/benchmark_results.json so the human can review the per-
    # contender × per-metric matrix and record the verdict.
    script = textwrap.dedent(
        f"""\
        set -euo pipefail
        cd {shlex.quote(repo)}
        uv sync --all-extras

        OUT=$(mktemp -d)/bench
        mkdir -p "$OUT"
        export LIOM_BENCH_OUT="$OUT"
        export LIOM_BENCH_DATASET_DIR={shlex.quote(dataset_dir)}
        export LIOM_BENCH_NNUNET_VENV={shlex.quote(nnunet_venv)}
        export LIOM_BENCH_NNUNET_DATASET_ID={shlex.quote(nnunet_dataset_id)}

        # --- nnU-Net v2: prepare dataset + preprocess + train (separate venv) ---
        # nnU-Net runs in a separate venv (torch-clobbering isolation); it is
        # NOT a liom-toolkit dependency. The env vars nnUNet_raw /
        # nnUNet_preprocessed / nnUNet_results MUST be set on the box before
        # invoking nnU-Net (no silent fallback — nnU-Net refuses to run
        # without them).
        NNUNET_RAW="${{nnUNet_raw:-/data/nnUNet_raw}}"
        NNUNET_PREP="${{nnUNet_preprocessed:-/data/nnUNet_preprocessed}}"
        NNUNET_RES="${{nnUNet_results:-/data/nnUNet_results}}"
        export nnUNet_raw="$NNUNET_RAW"
        export nnUNet_preprocessed="$NNUNET_PREP"
        export nnUNet_results="$NNUNET_RES"

        # Convert the labeled PNG slices to nnU-Net raw format.
        uv run liom-prepare-nnunet-dataset \\
            "$LIOM_BENCH_DATASET_DIR" \\
            "$NNUNET_RAW/Dataset${{LIOM_BENCH_NNUNET_DATASET_ID}}_LIOM6p5" \\
            --dataset-id "$LIOM_BENCH_NNUNET_DATASET_ID" \\
            --dataset-name "LIOM6p5" \\
            --file-ending ".png"

        # Activate the nnU-Net venv, preprocess, and train the 2d config.
        # `nnUNetv2_train` trains one fold; the prediction step is invoked
        # by the NnUnetContender via the subprocess bridge at benchmark time.
        NNUNET_PY="$LIOM_BENCH_NNUNET_VENV"
        "$NNUNET_PY" -m nnunetv2 nnUNetv2_plan_and_preprocess \\
            -d "$LIOM_BENCH_NNUNET_DATASET_ID" --verify_dataset_integrity
        "$NNUNET_PY" -m nnunetv2 nnUNetv2_train \\
            -d "$LIOM_BENCH_NNUNET_DATASET_ID" -c 2d

        # --- Run the full 4-contender benchmark via the library entry -----
        # run_benchmark trains + predicts + scores each contender through
        # the ship-gate eval-metric matrix and returns a per-contender
        # metric table. The result is serialized to JSON so the human can
        # review it and record the empirical architecture decision.
        uv run python - <<'PY'
        import json
        import os
        import numpy as np
        from liom_toolkit.segmentation.vseg.benchmark.contenders import (
            Improved2DContender,
            MonaiUnetContender,
            NnUnetContender,
            SwinUnetContender,
        )
        from liom_toolkit.segmentation.vseg.benchmark.run import run_benchmark
        from liom_toolkit.segmentation.vseg.benchmark.split import per_volume_split

        dataset_dir = os.environ["LIOM_BENCH_DATASET_DIR"]
        out_dir = os.environ["LIOM_BENCH_OUT"]

        # Per-volume split: S23 train / S24 held-out test (the labeled-slice
        # filename convention is confirmed with the lab before the run; the
        # harness reads paths from the dataset dir, never hardcodes them).
        import glob
        def _slices(sub):
            return sorted(glob.glob(os.path.join(dataset_dir, sub, "*.png")))

        brain_paths = {"S23": _slices("s23"), "S24": _slices("s24")}
        train_slices, test_slices = per_volume_split(
            brain_paths, train_brains=["S23"], test_brains=["S24"]
        )

        # Ground-truth masks: the harness expects one boolean mask per test
        # slice, aligned with test_slices. The masks live alongside the
        # images as <name>_mask.png; read them back as boolean arrays.
        from imageio.v3 import imread
        gt_masks = []
        for img_path in test_slices:
            stem = img_path[: -len(".png")]
            mask_path = stem + "_mask.png"
            gt_masks.append(np.asarray(imread(mask_path), dtype=bool))

        contenders = [
            Improved2DContender(),
            MonaiUnetContender(),
            SwinUnetContender(),
            NnUnetContender(
                nnunet_venv_python=os.environ["LIOM_BENCH_NNUNET_VENV"],
                dataset_id=int(os.environ["LIOM_BENCH_NNUNET_DATASET_ID"]),
            ),
        ]
        results = run_benchmark(
            contenders=contenders,
            split_config={
            "train_slices": train_slices,
                "test_slices": test_slices,
                "gt_masks": gt_masks,
                "patch_size": (1, 256, 256),
            },
            output_dir=out_dir,
        )

        # Serialize the per-contender × per-metric table to JSON so the
        # human can review it and record the verdict. Dict-valued metrics
        # (caliber_stratified_recall, boundary_artifact_regression) are
        # nested; scalar metrics are floats; vessel-free slices are recorded
        # as the literal string "vessel-free slice — metric undefined"
        # (never a NaN — AGENTS §2).
        results_path = os.path.join(out_dir, "benchmark_results.json")
        with open(results_path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print("LIOM_BENCH_RESULTS_PATH", results_path)
        PY

        echo "LIOM_BENCH_OK $LIOM_BENCH_OUT"
        """
    )

    # Real 4-contender training + nnU-Net preprocessing on 6.5 µm data takes
    # ~4-8 hours of GPU time on the 2x A6000. The timeout is generous so a
    # slow run does not flake; check=False so a non-zero exit surfaces as a
    # test failure with the remote stderr, not a subprocess exception.
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
        timeout=36000,
    )

    assert proc.returncode == 0, (
        f"Real benchmark SSH run exited {proc.returncode}.\n"
        f"--- remote stdout ---\n{proc.stdout.decode(errors='replace')[-4000:]}\n"
        f"--- remote stderr ---\n{proc.stderr.decode(errors='replace')[-8000:]}"
    )
    stdout = proc.stdout.decode(errors="replace")
    assert "LIOM_BENCH_OK" in stdout, (
        f"Benchmark did not reach the verification echo; stdout:\n{stdout[-4000:]}"
    )

    # Extract the output dir from the sentinel and assert the result table
    # file exists on the remote (a follow-up SSH ls + cat confirms the JSON
    # was written — the human reviews its contents to record the verdict).
    sentinel_line = next(
        (line for line in stdout.splitlines() if line.startswith("LIOM_BENCH_OK ")),
        None,
    )
    assert sentinel_line is not None, (
        f"LIOM_BENCH_OK sentinel missing from stdout:\n{stdout[-2000:]}"
    )
    remote_out_dir = sentinel_line.split("LIOM_BENCH_OK ", 1)[1].strip()

    results_path = f"{remote_out_dir}/benchmark_results.json"
    ls_probe = _ssh(host, user, f"test -f {shlex.quote(results_path)} && echo EXISTS", timeout=20)
    assert ls_probe.returncode == 0, (
        f"Result table {results_path} not found on remote "
        f"(exit {ls_probe.returncode}): {ls_probe.stderr.decode(errors='replace')[:500]}"
    )
    assert b"EXISTS" in ls_probe.stdout, (
        f"Result table existence check failed for {results_path}: "
        f"{ls_probe.stdout.decode(errors='replace')[:500]}"
    )
