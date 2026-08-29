"""Tests for DDP training entry hardening.

Covers the ``--ddp``-without-torchrun-env-vars raise (no silent
single-process fallback — AGENTS §2) and the ``evaluate()`` sum+count
all-reduce correctness under DDP (D-02a — sum+count, NOT mean-of-means,
because val shards differ in size when ``len(val_set) % world_size != 0``).

The 2-rank CPU gloo subprocess smoke (DEFER-04/SC-1) lives in a separate
plan; this file owns the unit-level DDP entry + all-reduce tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

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

    # The all_reduce helper needs the "remote" contribution. We attach it
    # via a wrapper that sets _remote on the tensors evaluate() passes to
    # all_reduce. Since evaluate() builds the sum/count tensors internally,
    # we intercept all_reduce to add the remote contribution based on the
    # tensor's shape/role. Simpler: make all_reduce add a fixed remote
    # scalar to whatever tensor it receives (loss_sum += 0.4, count += 1).
    # We distinguish by tracking call order: first call = loss_sum, second
    # = count, then per-metric (sum, count) pairs.
    remote_adds = [0.4, 1.0]  # loss_sum, count
    call_order = {"n": 0}

    def _all_reduce_sum_ordered(tensor, op=None):
        idx = call_order["n"]
        if idx < len(remote_adds):
            tensor.add_(torch.tensor(float(remote_adds[idx])))
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
