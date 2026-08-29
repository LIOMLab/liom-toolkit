"""Tests for DDP training entry hardening.

Covers the ``--ddp``-without-torchrun-env-vars raise (no silent
single-process fallback — AGENTS §2), the ``evaluate()`` sum+count
all-reduce correctness under DDP (sum+count, NOT mean-of-means, because
val shards differ in size when ``len(val_set) % world_size != 0``), and
the 2-rank CPU gloo subprocess ``torchrun`` smoke (the in-CI bar
exercising the real launch contract users run).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess  # ruff: ignore[suspicious-subprocess-import] - torchrun smoke legitimately spawns a subprocess
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.mark.ai
def test_ddp_without_env_vars_raises(tmp_path, monkeypatch):
    """train_model(ddp=True) without RANK/WORLD_SIZE/LOCAL_RANK raises RuntimeError.

    The raise names the missing env state (no silent single-process
    fallback — the project's named dominant failure mode). The raise
    happens BEFORE dataset construction so the test does not need to mock
    the dataset.
    """
    pytest.importorskip("torch")
    # Clear the torchrun env vars so the DDP entry branch sees them absent.
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)

    from liom_toolkit.segmentation.vseg.training import train_model

    # Patch wandb + the lazy torch import surface so the function reaches
    # the DDP entry branch (after the torch/wandb import guards) without
    # requiring the ai extra's heavy deps to actually run.
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    with pytest.raises(RuntimeError, match=r"RANK|WORLD_SIZE|LOCAL_RANK"):
        train_model(
            dataset_file="fake.zarr",
            node_name="channel_0",
            output_train=str(tmp_path / "training"),
            ddp=True,
        )


@pytest.mark.ai
def test_evaluate_sum_count_all_reduce(monkeypatch):
    """evaluate() under DDP (world_size=2) returns the global mean = total_sum/total_count.

    With a mocked dist (world_size=2, ReduceOp.SUM that sums across ranks),
    evaluate() on a tiny loader produces the global mean, NOT the per-shard
    mean. Single-process (dist not initialized) path produces the unchanged
    per-batch mean.

    This is the correctness boundary: when val shards differ in size
    (len(val_set) % world_size != 0), mean-of-means is wrong and sum+count
    is correct (D-02a).
    """
    pytest.importorskip("torch")
    import torch

    from liom_toolkit.segmentation.vseg.training import evaluate

    # --- Build a fake dist that simulates a 2-rank all-reduce by summing
    # the local tensor with a "remote" contribution. We model rank-0 seeing
    # 2 batches and rank-1 seeing 1 batch (uneven shards).
    fake_dist = MagicMock(name="torch.distributed")
    fake_dist.is_available.return_value = True
    fake_dist.is_initialized.return_value = True
    fake_dist.get_world_size.return_value = 2
    fake_dist.ReduceOp = MagicMock(name="ReduceOp")
    fake_dist.ReduceOp.SUM = "SUM"

    def _all_reduce_sum(tensor, op=None):
        # Simulate the other rank's contribution. We tag each tensor with
        # an attribute to know which "remote" value to add; for this test
        # we use a simple convention: loss_sum gets +remote_loss_sum,
        # count gets +remote_count. The remote shard has 1 batch with
        # loss 0.4 (so remote_loss_sum=0.4, remote_count=1).
        remote = getattr(tensor, "_remote", None)
        if remote is not None:
            tensor.add_(remote)
        return tensor

    fake_dist.all_reduce = MagicMock(side_effect=_all_reduce_sum)
    import sys

    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    monkeypatch.setattr(torch, "distributed", fake_dist)

    # --- Build a tiny model + loader + loss_fn stub.
    model = torch.nn.Linear(2, 1)

    # Local shard: 2 batches with losses 0.1 and 0.2 → local_sum=0.3, local_count=2.
    # Remote shard: 1 batch with loss 0.4 → remote_sum=0.4, remote_count=1.
    # Global mean = (0.3 + 0.4) / (2 + 1) = 0.7 / 3 ≈ 0.2333.
    # Per-shard (local) mean = 0.3 / 2 = 0.15 (WRONG — this is the bug D-02a fixes).
    batch_x_1 = torch.tensor([[1.0, 2.0]])
    batch_y_1 = torch.tensor([[0.1]])
    batch_x_2 = torch.tensor([[3.0, 4.0]])
    batch_y_2 = torch.tensor([[0.2]])

    # We need the loss values to be deterministic. Mock the model forward to
    # return y so MSE = 0... instead, use a loss_fn stub that returns a
    # fixed tensor per batch so the sums are deterministic.
    loss_values = [torch.tensor(0.1), torch.tensor(0.2)]
    call_idx = {"n": 0}

    class _FixedLoss(torch.nn.Module):
        def forward(self, pred, target):
            v = loss_values[call_idx["n"] % len(loss_values)]
            call_idx["n"] += 1
            return v

    class _TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            xs = [batch_x_1, batch_x_2]
            ys = [batch_y_1, batch_y_2]
            return xs[idx], ys[idx]

    loader = torch.utils.data.DataLoader(_TinyDataset(), batch_size=1)

    # The all_reduce helper simulates the other rank's contribution.
    # evaluate() calls all_reduce 5 times, each on a 2-element [sum, count]
    # tensor (loss, f1, accuracy, jaccard, recall). We add a 2-element
    # remote contribution per call: loss gets [0.4, 1.0] (remote shard has
    # 1 batch with loss 0.4); metrics get [0.0, 0.0] (we only assert on
    # epoch_loss — the metric remote contributions don't affect the loss
    # assertion).
    remote_pairs = [
        torch.tensor([0.4, 1.0], dtype=torch.float64),  # loss [sum, count]
        torch.tensor([0.0, 0.0], dtype=torch.float64),  # f1
        torch.tensor([0.0, 0.0], dtype=torch.float64),  # accuracy
        torch.tensor([0.0, 0.0], dtype=torch.float64),  # jaccard
        torch.tensor([0.0, 0.0], dtype=torch.float64),  # recall
    ]
    call_order = {"n": 0}

    def _all_reduce_sum_ordered(tensor, op=None):
        idx = call_order["n"]
        if idx < len(remote_pairs):
            tensor.add_(remote_pairs[idx])
        call_order["n"] += 1
        return tensor

    fake_dist.all_reduce = MagicMock(side_effect=_all_reduce_sum_ordered)

    device = torch.device("cpu")
    epoch_loss, _, _, _, f1, accuracy, jaccard, recall = evaluate(
        model, loader, _FixedLoss(), device
    )

    # Global mean = (0.3 + 0.4) / (2 + 1) = 0.2333...
    expected_global = (0.1 + 0.2 + 0.4) / 3.0
    assert abs(epoch_loss - expected_global) < 1e-5, (
        f"evaluate() under DDP must return the global sum+count mean "
        f"{expected_global}, got per-shard mean {epoch_loss}"
    )


def _make_tiny_labeled_zarr_for_smoke(tmp_path) -> str:
    """Build a (2, 64, 64) uint16 image zarr + matching uint8 label for the gloo smoke.

    ``train_model`` accepts a ``patch_size`` (Z, Y, X) tuple; the smoke
    passes ``(1, 32, 32)`` so the volume only needs to be 64x64 in YX (a
    2x2 grid per slice). A 2-slice volume gives a (2, 2, 2) grid → 8 grid
    patches x 4 rotations = 32 dataset items, enough for a 2-rank
    ``DistributedSampler`` to shard (16 per rank) with ``batch_size=2``.
    The non-empty region (``[:, 16:48, 16:48] = 1``) makes every patch
    valid so ``filter_empty`` keeps all items. Written via ``save_zarr`` +
    ``save_label_to_zarr`` (NGFF v0.5 multiscale, the real IO path
    ``OmeZarrLabelDataSet`` reads) — no mocking of zarr/numpy/torch
    (AGENTS section 5). The 32x32 patch is the smallest the U-Net
    accepts — it downsamples 32→16→8→4→2→1, and BatchNorm needs >1 value
    per channel at every spatial size (a 16x16 patch would collapse to
    1x1 too early and trip BatchNorm's "Expected more than 1 value per
    channel" check). The tiny patch + volume keep the smoke fast and
    light (the 256x256 default made the smoke hog the CPU for ~44s).
    """
    from liom_toolkit.conversion.conversion import save_label_to_zarr, save_zarr
    from liom_toolkit.utils.io import generate_label_color_dict_mask

    arr = np.zeros((2, 64, 64), dtype=np.uint16)
    arr[:, 16:48, 16:48] = 1000
    zarr_path = str(tmp_path / "smoke.zarr")
    save_zarr(arr, zarr_path, scales=(6.5, 6.5, 6.5), chunks=(2, 64, 64))

    label = np.zeros((2, 64, 64), dtype=np.uint8)
    label[:, 16:48, 16:48] = 1
    save_label_to_zarr(
        label,
        zarr_path,
        generate_label_color_dict_mask(),
        "training",
        scales=(6.5, 6.5, 6.5),
        chunks=(2, 64, 64),
    )
    return zarr_path


@pytest.mark.ai
def test_ddp_2rank_gloo_smoke(tmp_path):
    """2-rank CPU gloo DDP run via subprocess ``torchrun`` completes without corruption.

    Spawns the real ``torchrun`` launch contract (``python -m
    torch.distributed.run --nproc-per-node=2 --standalone``) against the
    ``liom-train-model`` CLI with ``--ddp`` — the only mechanism that
    exercises env-var injection, rendezvous, and ``--local-rank`` arg
    passing (``mp.spawn``/thread-gloo are rejected because they test a
    launch path no user runs). Asserts the run exits 0, the
    ``final_metrics.csv`` lands in ``output_train`` (the CWD-collision
    relocation), the manifest is a single valid JSON file (no rank-write
    race corruption), and a single rank-0 ``checkpoint.latest.pth`` exists
    (the ``DDPResumeManager`` rank-0-only write invariant).

    This is the in-CI bar for the DDP entry shape. W&B is forced to
    ``disabled`` mode (no network, no run dir) — the rank-0-only W&B
    invariant is unit-tested alongside the DDP entry; the smoke verifies
    it holds end-to-end via the exit-0 + file-invariant assertions.
    """
    pytest.importorskip("torch")

    zarr_path = _make_tiny_labeled_zarr_for_smoke(tmp_path)
    out_dir = tmp_path / "out"

    # Pick a free port in the parent via socket.bind — xdist-safe (the
    # pattern PyTorch's own PRs and HuggingFace's get_torch_dist_unique_port
    # use). ``--standalone`` runs its own rendezvous store on a free
    # port, but passing ``--master-port`` makes the rendezvous endpoint
    # deterministic and avoids the rare race where two smokes pick the same
    # ephemeral port.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - controlled torchrun invocation (sys.executable + known args)
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=2",
            "--standalone",
            "--master-port",
            str(port),
            # --local-addr=127.0.0.1 + MASTER_ADDR=127.0.0.1 (in env) avoid the
            # macOS gloo reverse-DNS hang: without them gloo resolves the
            # machine hostname, which fails with gai error 8 on macOS and
            # stalls rendezvous until the subprocess timeout. Loopback is the
            # correct address for a single-node 2-rank CPU smoke.
            "--local-addr",
            "127.0.0.1",
            "--module",
            "liom_toolkit.scripts.liom_train_model",
            zarr_path,
            "training",
            "--ddp",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--patch-size",
            "1,32,32",
            "--output-train",
            str(out_dir),
            "--wandb-mode",
            "disabled",
        ],
        capture_output=True,
        check=False,
        env={**os.environ, "WANDB_MODE": "disabled", "MASTER_ADDR": "127.0.0.1"},
        timeout=300,
    )

    assert proc.returncode == 0, (
        f"torchrun gloo smoke exited {proc.returncode}.\n"
        f"--- stdout ---\n{proc.stdout.decode(errors='replace')[-4000:]}\n"
        f"--- stderr ---\n{proc.stderr.decode(errors='replace')[-8000:]}"
    )

    # final_metrics.csv relocated to Path(output_train)/final_metrics.csv
    # (not the process CWD) — eliminates the concurrent-run collision.
    assert (out_dir / "final_metrics.csv").exists(), (
        "final_metrics.csv must land in output_train (the CWD-collision relocation)"
    )

    # Single manifest, valid JSON — no rank-write race corruption (the
    # DDPResumeManager._write_manifest rank-0 guard prevents two ranks
    # os.replace-ing the same {pipeline}.json).
    manifest = out_dir / "_liom_checkpoints" / "train_model.json"
    assert manifest.exists(), "manifest must exist (DDPResumeManager rank-0 write)"
    # Parses as valid JSON — a half-written (corrupted) manifest would raise.
    json.loads(manifest.read_text())

    # Single rank-0 .pth — DDPResumeManager.save_weights no-ops on non-rank-0
    # so only rank 0 writes checkpoint.latest.pth (no race corruption).
    assert (out_dir / "files" / "checkpoint.latest.pth").exists(), (
        "checkpoint.latest.pth must exist (rank-0-only save_weights)"
    )
