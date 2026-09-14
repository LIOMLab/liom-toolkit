"""SSH-driven real SSL pretraining + warm-start + full eval on the lab CUDA box.

The in-CI SSL tests (``test_corpus.py``, ``test_masking.py``,
``test_pretrain.py``, ``test_warmstart.py``, ``test_compare.py``) exercise
the pipeline on tiny synthetic volumes — they prove the corpus sampler, the
vessel-aware mask, the checkpoint format, the key-match guarantee, and the
comparison aggregation on CPU. This test goes beyond them: it SSHes to the
lab CUDA box and runs the full Phase-18 chain on the real ~20-brain corpus
— ``liom-pretrain`` (masked-inpainting pretraining on unlabeled OME-Zarr
volumes) → warm-start fine-tune via the in-process ``NNUNetTrainer`` +
``-pretrained_weights`` → a from-scratch baseline run under the same
epoch budget (equalized compute, D-06b) → the pretrained-vs-from-scratch
comparison through the REUSED ``eval_metrics`` gate via ``run_comparison``
— then asserts the ``LIOM_PRETRAIN_OK`` sentinel and verifies the
pretrained checkpoint + comparison JSON + ``18-PRETRAIN-RESULTS.md`` doc
exist on the remote. Real GPUs + real data are the only honest check that
the unit-tested chain produces the per-metric matrix the ship decision
(PRETRAIN-01 / SC-3) is judged against.

Skip policy (CI legs without the box pass cleanly — never fail):
* ``LIOM_CUDA_HOST`` env var unset → skipped at collection via
  ``pytest.mark.skipif``.
* The host is unreachable (SSH connect timeout / auth failure) →
  ``pytest.skip`` inside the body (a missing lab box is an environment
  condition, not a regression).
* ``nvidia-smi`` reports 0 GPUs → ``pytest.skip`` (no CUDA to pretrain on).

Configuration via env vars (defaults match the lab box layout):
* ``LIOM_CUDA_HOST`` — the SSH host (no default; must be set or the test
  skips at collection).
* ``LIOM_CUDA_USER`` — the SSH user (default: current local user).
* ``LIOM_CUDA_REPO`` — the repo path on the remote (default
  ``~/code/liom-toolkit``).
* ``LIOM_SSL_CORPUS_GLOB`` — remote shell glob for the unlabeled OME-Zarr
  corpus volumes (default ``/data/LSFM/S*/S*.ome.zarr`` — the ~20-brain
  layout; a PARAMETER, never a hardcoded library default — AGENTS §1).
* ``LIOM_SSL_PRETRAINED_OUT`` — the pretrained checkpoint output path on
  the remote (default ``/data/LSFM/ssl_pretrained/pretrained.pth``).
* ``LIOM_SSL_DATASET_ID`` / ``LIOM_SSL_DATASET_NAME`` — the nnU-Net dataset
  the warm-start fine-tunes on (defaults ``101`` / ``Dataset101_LIOM6p5``).
* ``LIOM_SSL_PLANS_ID`` — the nnU-Net plans identifier the dataset was
  preprocessed with (default ``nnUNetPlans`` — matches the on-disk
  preprocessed Dataset101_LIOM6p5).
* ``LIOM_SSL_FOLD`` — the cross-validation fold to train (default ``0``).
* ``LIOM_SSL_EPOCHS`` / ``LIOM_SSL_ITERS_PER_EPOCH`` — the equalized compute
  budget threaded into BOTH the warm-start and from-scratch runs plus the
  comparison record (defaults ``50`` / ``250`` — the pre-registered
  Phase-14 budget).
* ``LIOM_SSL_RESULTS_DOC`` — the results doc path relative to the repo
  root the run verifies on the remote (default the Phase-18
  ``18-PRETRAIN-RESULTS.md``).

Uses ONLY the stdlib ``subprocess`` module to drive ``ssh`` with a list argv
(never a shell — T-18-03 subprocess injection; no paramiko / fabric
dependency — AGENTS section 3). The remote command is a single ``bash -s``
script piped over SSH: ``set -euo pipefail`` makes any stage failure exit
non-zero, the ``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results``
env vars are exported BEFORE ``liom-pretrain`` and the in-process
``NNUNetTrainer`` (RESEARCH Pitfall 5 — unset vars cause opaque downstream
failures), and a ``LIOM_PRETRAIN_OK <out_dir>`` sentinel proves the full
chain reached the end. Marked ``ai`` (needs the torch extra) and ``slow``
(real pretraining + two fold trainings + eval on a remote box;
auto-deselected from fast iteration by the ``slow`` marker convention in
``pyproject.toml``).

The ship decision itself is a human judgment over the recorded comparison
(18-VALIDATION.md §Manual-Only Verifications) — this test only proves the
chain ran end-to-end and produced the artifacts; the human reviews the
per-metric matrix against the pre-registered gate.
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


def test_pretrain_real_ssh_run():
    """SSH to the lab CUDA box and run the real pretrain → warm-start → eval chain.

    Probes SSH connectivity and GPU count first (skips cleanly on failure),
    then pipes a ``bash -s`` script over SSH that: exports the three
    ``nnUNet_*`` env vars (RESEARCH Pitfall 5), syncs the repo with the
    ``ai`` + ``benchmark`` extras, runs ``liom-pretrain`` on the real
    unlabeled corpus, runs the warm-start (``nnUNetTrainer`` +
    ``-pretrained_weights``) AND a from-scratch baseline under the same
    epoch budget in-process, and scores both through ``run_comparison``
    (the REUSED ``eval_metrics`` gate). The script echoes
    ``LIOM_PRETRAIN_OK <out_dir>`` on success and writes the comparison
    result to ``<out_dir>/ssl_comparison.json``. The test asserts exit 0,
    the sentinel in stdout, and that the pretrained checkpoint, the
    comparison JSON, and the results doc exist on the remote.

    Skips cleanly without ``LIOM_CUDA_HOST`` (CI legs skip at collection);
    skips on an unreachable host or a box with 0 GPUs. The ship decision is
    recorded in ``18-PRETRAIN-RESULTS.md`` by the human after this test
    passes on the box.
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

    # --- GPU count probe (skip if no CUDA to pretrain on) ----------------
    gpu_probe = _ssh(host, user, "nvidia-smi -L | grep -c 'GPU ' || echo 0", timeout=20)
    if gpu_probe.returncode != 0:
        pytest.skip(f"nvidia-smi probe failed on {host} (exit {gpu_probe.returncode})")
    gpu_count = gpu_probe.stdout.decode(errors="replace").strip().splitlines()[-1].strip()
    if not gpu_count.isdigit() or int(gpu_count) < 1:
        pytest.skip(f"no CUDA GPUs reported by nvidia-smi on {host} (got {gpu_count!r})")

    # --- Run parameters (all parameterized via env vars) -----------------
    # Defaults match the lab layout; every path is overridable so the test
    # never hardcodes a lab-specific location (AGENTS §1).
    corpus_glob = os.environ.get("LIOM_SSL_CORPUS_GLOB") or "/data/LSFM/S*/S*.ome.zarr"
    pretrained_out = os.environ.get("LIOM_SSL_PRETRAINED_OUT") or (
        "/data/LSFM/ssl_pretrained/pretrained.pth"
    )
    dataset_id = os.environ.get("LIOM_SSL_DATASET_ID") or "101"
    dataset_name = os.environ.get("LIOM_SSL_DATASET_NAME") or (
        f"Dataset{int(dataset_id):03d}_LIOM6p5"
    )
    plans_id = os.environ.get("LIOM_SSL_PLANS_ID") or "nnUNetPlans"
    fold = os.environ.get("LIOM_SSL_FOLD") or "0"
    epochs = os.environ.get("LIOM_SSL_EPOCHS") or "50"
    iters_per_epoch = os.environ.get("LIOM_SSL_ITERS_PER_EPOCH") or "250"
    results_doc = os.environ.get("LIOM_SSL_RESULTS_DOC") or (
        ".planning/phases/"
        "liom-toolkit-18-vseg-self-supervised-pretraining-parallel/"
        "18-PRETRAIN-RESULTS.md"
    )

    # --- The real chain (one bash -s script over SSH) ---------------------
    # A single bash script piped to `ssh ... bash -s` so the pretraining,
    # warm-start + from-scratch training, comparison, and artifact
    # verification all run on the remote in one SSH session (no scp, no
    # second round-trip per check). The script exits non-zero if any step
    # fails (set -euo pipefail), and prints LIOM_PRETRAIN_OK <out_dir> on
    # success so the test can confirm the full chain ran end-to-end.
    #
    # Two nnUNetTrainer subclasses are defined and registered into the
    # nnunetv2 module namespace so get_trainer_from_args /
    # recursive_find_python_class resolve them: one for the warm-start and
    # one for the from-scratch baseline, BOTH capped at the same epoch
    # budget (equalized compute, D-06b). The distinct class names keep the
    # two nnU-Net output directories from colliding. Fork-after-CUDA
    # deadlocks the nnU-Net dataloader workers (the real-run failure mode
    # from 18-05), so the multiprocessing start method is pinned to spawn.
    script = textwrap.dedent(
        f"""\
        set -euo pipefail
        cd {shlex.quote(repo)}
        git pull --ff-only
        uv sync --extra ai --extra benchmark

        # nnU-Net v2 env vars MUST be set before the in-process NNUNetTrainer
        # is instantiated (RESEARCH Pitfall 5; warmstart.validate_nnunet_env
        # raises RuntimeError naming missing vars — no silent fallback).
        export nnUNet_raw="${{nnUNet_raw:-/data/nnUNet_raw}}"
        export nnUNet_preprocessed="${{nnUNet_preprocessed:-/data/nnUNet_preprocessed}}"
        export nnUNet_results="${{nnUNet_results:-/data/nnUNet_results}}"

        export LIOM_SSL_CORPUS_GLOB={shlex.quote(corpus_glob)}
        export LIOM_SSL_PRETRAINED_OUT={shlex.quote(pretrained_out)}
        export LIOM_SSL_DATASET_ID={shlex.quote(dataset_id)}
        export LIOM_SSL_DATASET_NAME={shlex.quote(dataset_name)}
        export LIOM_SSL_PLANS_ID={shlex.quote(plans_id)}
        export LIOM_SSL_FOLD={shlex.quote(fold)}
        export LIOM_SSL_EPOCHS={shlex.quote(epochs)}
        export LIOM_SSL_ITERS_PER_EPOCH={shlex.quote(iters_per_epoch)}

        OUT=$(mktemp -d)/ssl
        mkdir -p "$OUT"
        mkdir -p "$(dirname "$LIOM_SSL_PRETRAINED_OUT")"
        export LIOM_SSL_OUT="$OUT"

        # --- Stage 1: real masked-inpainting pretraining on the corpus ---
        mapfile -t VOLS < <(compgen -G "$LIOM_SSL_CORPUS_GLOB" | sort)
        if [ "${{#VOLS[@]}}" -lt 1 ]; then
            echo "no corpus volumes matched $LIOM_SSL_CORPUS_GLOB" >&2
            exit 3
        fi
        echo "pretraining on ${{#VOLS[@]}} volumes: ${{VOLS[*]}}"
        PLANS_JSON="$nnUNet_preprocessed/$LIOM_SSL_DATASET_NAME/$LIOM_SSL_PLANS_ID.json"
        DATASET_JSON="$nnUNet_raw/$LIOM_SSL_DATASET_NAME/dataset.json"
        test -f "$PLANS_JSON"
        test -f "$DATASET_JSON"
        uv run liom-pretrain \\
            --volume-paths "${{VOLS[@]}}" \\
            --plans "$PLANS_JSON" \\
            --dataset-json "$DATASET_JSON" \\
            --pretrained-output "$LIOM_SSL_PRETRAINED_OUT" \\
            --epochs "$LIOM_SSL_EPOCHS" \\
            --amp
        test -f "$LIOM_SSL_PRETRAINED_OUT"

        # --- Stage 2: warm-start + from-scratch fold training (in-process) -
        uv run python - <<'PY'
        import inspect
        import multiprocessing
        import os

        # Fork-after-CUDA deadlocks the nnU-Net dataloader workers; spawn
        # each worker cleanly instead (the fix the real 18-05 run needed).
        multiprocessing.set_start_method("spawn", force=True)

        import torch

        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_mod
        from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

        from liom_toolkit.segmentation.vseg.ssl.warmstart import warm_start

        epochs = int(os.environ["LIOM_SSL_EPOCHS"])
        dataset_id = int(os.environ["LIOM_SSL_DATASET_ID"])
        fold = int(os.environ["LIOM_SSL_FOLD"])
        plans_id = os.environ["LIOM_SSL_PLANS_ID"]
        ckpt = os.environ["LIOM_SSL_PRETRAINED_OUT"]

        class nnUNetTrainerSSLPretrained(trainer_mod.nnUNetTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.num_epochs = epochs

        class nnUNetTrainerSSLScratch(trainer_mod.nnUNetTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.num_epochs = epochs

        # Register into the nnunetv2 trainer module so
        # recursive_find_python_class resolves both names.
        trainer_mod.nnUNetTrainerSSLPretrained = nnUNetTrainerSSLPretrained
        trainer_mod.nnUNetTrainerSSLScratch = nnUNetTrainerSSLScratch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Warm-start: load_pretrained_weights BEFORE run_training
        # (encoder+decoder transfer; .seg_layers. keys skipped).
        warm_start(
            dataset_id,
            fold,
            ckpt,
            configuration="2d",
            device=device,
            trainer_name="nnUNetTrainerSSLPretrained",
            plans_identifier=plans_id,
        )

        # From-scratch baseline: same architecture + epoch budget, no
        # pretrained weights (equalized compute — D-06b).
        scratch = get_trainer_from_args(
            dataset_id,
            configuration="2d",
            fold=fold,
            trainer_name="nnUNetTrainerSSLScratch",
            plans_identifier=plans_id,
            device=device,
        )
        maybe_load_checkpoint(scratch, continue_training=False, validation_only=False)
        scratch.run_training()
        # nnU-Net v2 versions differ on the validation-probabilities
        # parameter name (same fix warmstart.py applies); pass False under
        # the name the installed trainer accepts.
        sig = inspect.signature(scratch.perform_actual_validation)
        if "save_probabilities" in sig.parameters:
            scratch.perform_actual_validation(save_probabilities=False)
        else:
            scratch.perform_actual_validation(export_validation_probabilities=False)
        PY

        # --- Stage 3: pretrained-vs-from-scratch via run_comparison --------
        # Score both contenders' validation predictions against the fold's
        # gt_segmentations through the REUSED eval_metrics gate. The
        # per-volume bookkeeping (train cases from splits_final.json vs the
        # fold's validation cases) is passed to run_comparison, which
        # enforces per_volume_split (no brain overlap, no patch-level).
        uv run python - <<'PY'
        import glob
        import json
        import os

        import numpy as np
        import SimpleITK as sitk
        from imageio.v3 import imread

        from liom_toolkit.segmentation.vseg.ssl.compare import run_comparison

        res_dir = os.environ["nnUNet_results"]
        pre_dir = os.environ["nnUNet_preprocessed"]
        ds = os.environ["LIOM_SSL_DATASET_NAME"]
        plans = os.environ["LIOM_SSL_PLANS_ID"]
        fold = int(os.environ["LIOM_SSL_FOLD"])
        iters = int(os.environ["LIOM_SSL_ITERS_PER_EPOCH"])
        out_dir = os.environ["LIOM_SSL_OUT"]

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

        def _case_preds(trainer_name):
            vdir = os.path.join(
                res_dir, ds,
                trainer_name + "__" + plans + "__2d",
                "fold_" + str(fold), "validation",
            )
            return {{
                _stem(p): p
                for p in sorted(glob.glob(os.path.join(vdir, "*")))
                if os.path.isfile(p) and _stem(p) not in ("summary", "progress")
            }}

        pre_preds = _case_preds("nnUNetTrainerSSLPretrained")
        scr_preds = _case_preds("nnUNetTrainerSSLScratch")

        gt_dir = os.path.join(pre_dir, ds, "gt_segmentations")
        gt = {{
            _stem(p): p
            for p in sorted(glob.glob(os.path.join(gt_dir, "*")))
            if os.path.isfile(p)
        }}

        cases = sorted(set(pre_preds) & set(scr_preds) & set(gt))
        if not cases:
            raise RuntimeError(
                "no overlapping validation cases across the three dirs: "
                "pretrained=" + str(sorted(pre_preds))
                + " scratch=" + str(sorted(scr_preds))
                + " gt=" + str(sorted(gt))
            )

        with open(os.path.join(pre_dir, ds, "splits_final.json")) as fh:
            splits = json.load(fh)
        fold_split = splits[fold] if isinstance(splits, list) else splits[str(fold)]
        train_cases = list(fold_split["train"])

        brain_paths = {{c: [gt[c]] for c in train_cases if c in gt}}
        brain_paths.update({{c: [gt[c]] for c in cases}})

        result = run_comparison(
            brain_paths,
            train_brains=train_cases,
            test_brains=cases,
            pretrained_predictions=[_load_mask(pre_preds[c]) for c in cases],
            from_scratch_predictions=[_load_mask(scr_preds[c]) for c in cases],
            gt_masks=[_load_mask(gt[c]) for c in cases],
            iterations_per_epoch=iters,
        )
        out_path = os.path.join(out_dir, "ssl_comparison.json")
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print("LIOM_SSL_COMPARISON_PATH", out_path)
        PY

        # --- Verify the run artifacts + the results doc on the remote ------
        test -f "$LIOM_SSL_PRETRAINED_OUT"
        test -f "$LIOM_SSL_OUT/ssl_comparison.json"
        test -f {shlex.quote(results_doc)}
        echo "LIOM_PRETRAIN_OK $LIOM_SSL_OUT"
        """
    )

    # Real pretraining on the ~20-brain corpus + two 50-epoch fold trainings
    # + eval takes hours of GPU time on the A6000 box. The timeout is
    # generous so a slow run does not flake; check=False so a non-zero exit
    # surfaces as a test failure with the remote stderr, not a subprocess
    # exception.
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
        f"SSL pretrain+warm-start+eval SSH run exited {proc.returncode}.\n"
        f"--- remote stdout ---\n{proc.stdout.decode(errors='replace')[-4000:]}\n"
        f"--- remote stderr ---\n{proc.stderr.decode(errors='replace')[-8000:]}"
    )
    stdout = proc.stdout.decode(errors="replace")
    assert "LIOM_PRETRAIN_OK" in stdout, (
        f"Pretrain chain did not reach the verification echo; stdout:\n{stdout[-4000:]}"
    )

    # Extract the output dir from the sentinel and assert the comparison
    # result file exists on the remote (a follow-up SSH test -f confirms
    # the JSON was written — the human reviews its contents against the
    # pre-registered gate to record the ship decision).
    sentinel_line = next(
        (line for line in stdout.splitlines() if line.startswith("LIOM_PRETRAIN_OK ")),
        None,
    )
    assert sentinel_line is not None, (
        f"LIOM_PRETRAIN_OK sentinel missing from stdout:\n{stdout[-2000:]}"
    )
    remote_out_dir = sentinel_line.split("LIOM_PRETRAIN_OK ", 1)[1].strip()

    comparison_path = f"{remote_out_dir}/ssl_comparison.json"
    ls_probe = _ssh(
        host,
        user,
        f"test -f {shlex.quote(comparison_path)} && echo EXISTS",
        timeout=20,
    )
    assert ls_probe.returncode == 0, (
        f"Comparison result {comparison_path} not found on remote "
        f"(exit {ls_probe.returncode}): {ls_probe.stderr.decode(errors='replace')[:500]}"
    )
    assert b"EXISTS" in ls_probe.stdout, (
        f"Comparison result existence check failed for {comparison_path}: "
        f"{ls_probe.stdout.decode(errors='replace')[:500]}"
    )
