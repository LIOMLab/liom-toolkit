"""TDD tests for the masked-inpainting pretraining loop (ssl.pretrain).

These are the RED-phase tests for the Wave-1 tracer slice: they assert the
behaviors the pretraining loop must provide before any implementation exists.
The loop builds the nnU-Net 2D ResEnc network via
``get_network_from_plans`` (so the saved checkpoint's state_dict keys match
what ``load_pretrained_weights`` expects -- the D-01a key-match guarantee),
runs a masked-inpainting reconstruction objective (MSE on the masked regions
vs the original unmasked image), and saves the checkpoint as
``{'network_weights': state_dict}`` -- the exact format the warm-start
loader consumes. An empty corpus raises ``ValueError`` (no zero-fill / no
silent pass -- AGENTS section 2).

All tests run on CPU and are gated on the ``[ai]`` extra (torch + nnunetv2)
via ``pytest.importorskip`` at the first line of each test body and the
``@pytest.mark.ai`` marker.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.ai
def test_build_pretrain_network_returns_resenc_module_with_expected_keys(
    tiny_2d_resenc_plans, tiny_2d_resenc_dataset_json
):
    """build_pretrain_network returns a 2D ResEnc UNet with warm-start-matching keys.

    The D-01a key-match guarantee: the pretraining network is built via the
    SAME ``get_network_from_plans`` call the warm-start uses, so its
    state_dict keys (``encoder.*``, ``decoder.*``, ``decoder.seg_layers.*``)
    are exactly the keys ``load_pretrained_weights`` will look for. A
    generic MONAI UNet's keys would NOT match and would raise
    ``AssertionError`` at warm-start load. This test asserts the network is
    a ``torch.nn.Module``, the state_dict is non-empty, and the keys start
    with the expected nnU-Net module paths.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    import torch

    from liom_toolkit.segmentation.vseg.ssl.pretrain import build_pretrain_network

    net = build_pretrain_network(
        tiny_2d_resenc_plans,
        tiny_2d_resenc_dataset_json,
        configuration="2d",
        device=torch.device("cpu"),
    )
    assert isinstance(net, torch.nn.Module)
    sd = net.state_dict()
    assert len(sd) > 0
    keys = list(sd.keys())
    # The encoder + decoder module paths are the nnU-Net ResEnc convention.
    assert any(k.startswith("encoder.") for k in keys), (
        f"expected encoder.* keys (nnU-Net ResEnc layout), got {keys[:5]}"
    )
    assert any(k.startswith("decoder.") for k in keys), (
        f"expected decoder.* keys (nnU-Net ResEnc layout), got {keys[:5]}"
    )
    # The seg heads (skipped by load_pretrained_weights) are present so the
    # warm-start loader has something to skip.
    assert any(".seg_layers." in k for k in keys), (
        f"expected decoder.seg_layers.* keys, got {keys[:5]}"
    )


@pytest.mark.ai
def test_masked_inpainting_pretrain_runs_and_saves_checkpoint(
    tiny_2d_resenc_plans, tiny_2d_resenc_dataset_json, tmp_path
):
    """The masked-inpainting loop runs N steps on a tiny volume and saves a loadable checkpoint.

    Runs a 2-step pretraining loop on a tiny synthetic (2, 16, 16) volume on
    CPU, asserts the checkpoint file is saved at the parameterized output
    path and ``torch.load(..., weights_only=False)`` returns a dict
    containing the ``'network_weights'`` key (the exact format
    ``load_pretrained_weights`` expects). This is the tracer-tiny end-to-end
    proof: pretrain -> checkpoint on CPU.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    import torch

    from liom_toolkit.segmentation.vseg.ssl.pretrain import (
        build_pretrain_network,
        masked_inpainting_pretrain,
    )

    net = build_pretrain_network(
        tiny_2d_resenc_plans,
        tiny_2d_resenc_dataset_json,
        configuration="2d",
        device=torch.device("cpu"),
    )
    # A tiny synthetic 2-channel (2, 16, 16) volume -- one "batch" of one slice.
    volume = torch.randn(1, 2, 16, 16, dtype=torch.float32)
    out_path = tmp_path / "pretrained.pth"

    masked_inpainting_pretrain(
        net,
        [volume],
        epochs=1,
        output_path=str(out_path),
        device=torch.device("cpu"),
    )

    assert out_path.is_file(), f"checkpoint not saved at {out_path}"
    saved = torch.load(str(out_path), weights_only=False)
    assert isinstance(saved, dict)
    assert "network_weights" in saved, (
        f"checkpoint must contain 'network_weights' key (load_pretrained_weights format), "
        f"got keys {list(saved.keys())}"
    )
    assert isinstance(saved["network_weights"], dict)
    assert len(saved["network_weights"]) > 0


@pytest.mark.ai
def test_masked_inpainting_pretrain_loss_is_finite(
    tiny_2d_resenc_plans, tiny_2d_resenc_dataset_json, tmp_path
):
    """The reconstruction loss is finite (no NaN / inf) across the pretraining steps.

    The masked-inpainting objective is MSE on the masked regions vs the
    original unmasked image. A NaN / inf loss would propagate garbage into
    the pretrained weights and silently corrupt the warm-start. This test
    asserts the returned per-epoch loss is finite (``torch.isfinite``).
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    import torch

    from liom_toolkit.segmentation.vseg.ssl.pretrain import (
        build_pretrain_network,
        masked_inpainting_pretrain,
    )

    net = build_pretrain_network(
        tiny_2d_resenc_plans,
        tiny_2d_resenc_dataset_json,
        configuration="2d",
        device=torch.device("cpu"),
    )
    volume = torch.randn(1, 2, 16, 16, dtype=torch.float32)
    out_path = tmp_path / "pretrained_finite.pth"

    losses = masked_inpainting_pretrain(
        net,
        [volume],
        epochs=2,
        output_path=str(out_path),
        device=torch.device("cpu"),
    )
    assert losses, "pretrain must return at least one per-epoch loss"
    for loss in losses:
        assert torch.isfinite(torch.tensor(float(loss))), (
            f"reconstruction loss must be finite (no NaN/inf), got {loss}"
        )


@pytest.mark.ai
def test_masked_inpainting_pretrain_raises_on_empty_corpus(
    tiny_2d_resenc_plans, tiny_2d_resenc_dataset_json, tmp_path
):
    """The loop raises ValueError on an empty corpus (no zero-fill / no silent pass).

    An empty corpus teaches the network nothing; silently producing a
    checkpoint with the freshly-initialized weights would be a plausible-
    shaped-but-wrong artifact (AGENTS section 2 -- no silent wrong-data
    fallback). The loop must raise ``ValueError`` naming the empty-dataset
    cause so the caller sees the real problem.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    import torch

    from liom_toolkit.segmentation.vseg.ssl.pretrain import (
        build_pretrain_network,
        masked_inpainting_pretrain,
    )

    net = build_pretrain_network(
        tiny_2d_resenc_plans,
        tiny_2d_resenc_dataset_json,
        configuration="2d",
        device=torch.device("cpu"),
    )
    out_path = tmp_path / "pretrained_empty.pth"

    with pytest.raises(ValueError):
        masked_inpainting_pretrain(
            net,
            [],
            epochs=1,
            output_path=str(out_path),
            device=torch.device("cpu"),
        )
    # No checkpoint should be written on the empty-corpus failure path.
    assert not Path(out_path).exists()


@pytest.mark.ai
def test_pretrain_module_has_no_assert_validation_statements():
    """pretrain.py contains no `assert` validation statements (AGENTS section 2).

    ``assert`` is stripped under ``python -O``; input validation must use
    ``if ...: raise ValueError(...)`` with the offending value in the
    message. This is a config-as-data test (parsing the committed module as
    text is permitted for the no-assert invariant -- it is a repo-wide
    invariant, not a behavior assertion).
    """
    pytest.importorskip("torch")
    import liom_toolkit.segmentation.vseg.ssl.pretrain as pretrain_mod

    src = Path(pretrain_mod.__file__).read_text()
    # Strip comment lines so a commented-out `# assert ...` does not trip the check.
    code_lines = [line for line in src.splitlines() if not line.strip().startswith("#")]
    code = "\n".join(code_lines)
    assert " assert " not in code, (
        "pretrain.py must not use `assert` for validation (AGENTS section 2 -- "
        "stripped under python -O); use `if ...: raise ValueError(...)` instead"
    )
