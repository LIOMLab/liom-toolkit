"""TDD tests for the warm-start checkpoint-load helper (ssl.warmstart).

These are the RED-phase tests for the Wave-1 tracer slice warm-start half:
they assert the behaviors the warm-start helper must provide before any
implementation exists. The helper loads a pretrained checkpoint into a
fresh nnU-Net 2D ResEnc network via upstream ``load_pretrained_weights`` --
the D-01a tracer proof that a checkpoint built by ``build_pretrain_network``
loads into a second same-architecture network with NO ``AssertionError``.
It catches upstream ``AssertionError`` (key mismatch) and re-raises as
``RuntimeError`` with an actionable message (AGENTS section 2 -- surface
explicitly, no silent partial load), validates the nnU-Net env vars are
set before instantiating the trainer, and raises ``ValueError`` on a
nonexistent checkpoint path with the offending path in the message.

All tests run on CPU and are gated on the ``[ai]`` extra (torch + nnunetv2)
via ``pytest.importorskip`` at the first line of each test body and the
``@pytest.mark.ai`` marker.
"""

from __future__ import annotations

import pytest


@pytest.mark.ai
def test_load_pretrained_checkpoint_loads_same_architecture_no_assertion_error(
    tiny_2d_resenc_plans, tiny_2d_resenc_dataset_json, tmp_path
):
    """A same-architecture checkpoint loads via load_pretrained_weights with NO AssertionError.

    This is the D-01a tracer proof: save a checkpoint
    ``{'network_weights': net.state_dict()}`` from a fresh
    ``build_pretrain_network`` 2D ResEnc, call ``load_pretrained_checkpoint``
    into a SECOND fresh network of the SAME architecture, and assert no
    ``AssertionError`` is raised and the encoder weights match (the load
    actually transferred the weights, not just silently no-op'd).
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    import torch

    from liom_toolkit.segmentation.vseg.ssl.pretrain import build_pretrain_network
    from liom_toolkit.segmentation.vseg.ssl.warmstart import load_pretrained_checkpoint

    net_a = build_pretrain_network(
        tiny_2d_resenc_plans,
        tiny_2d_resenc_dataset_json,
        configuration="2d",
        device=torch.device("cpu"),
    )
    # Mutate the encoder weights so we can detect the transfer (otherwise
    # two freshly-initialized networks could coincidentally match).
    with torch.no_grad():
        for p in net_a.parameters():
            p.add_(0.123)
    ckpt_path = tmp_path / "pretrained.pth"
    torch.save({"network_weights": net_a.state_dict()}, str(ckpt_path))

    net_b = build_pretrain_network(
        tiny_2d_resenc_plans,
        tiny_2d_resenc_dataset_json,
        configuration="2d",
        device=torch.device("cpu"),
    )
    # Capture a pre-load encoder weight for the match assertion.
    pre_load_encoder_w = next(k for k in net_b.state_dict() if k.startswith("encoder."))
    pre_load_val = net_b.state_dict()[pre_load_encoder_w].clone()

    load_pretrained_checkpoint(str(ckpt_path), net_b)

    post_load_val = net_b.state_dict()[pre_load_encoder_w]
    # The encoder weight must have changed (the pretrained weights loaded).
    assert not torch.allclose(pre_load_val, post_load_val), (
        "load_pretrained_checkpoint did not transfer the encoder weights -- "
        "the post-load encoder weight is identical to the pre-load value"
    )
    # And it must match the source network's encoder weight.
    assert torch.allclose(post_load_val, net_a.state_dict()[pre_load_encoder_w]), (
        "load_pretrained_checkpoint transferred the wrong encoder weight -- "
        "post-load value does not match the source checkpoint"
    )


@pytest.mark.ai
def test_load_pretrained_checkpoint_raises_runtime_error_on_key_mismatch(
    tiny_2d_resenc_plans, tiny_2d_resenc_dataset_json, tmp_path
):
    """A mismatched-key checkpoint raises RuntimeError (catches upstream AssertionError).

    When ``load_pretrained_weights`` raises ``AssertionError`` (a key is
    missing or shape-mismatched in the pretrained checkpoint), the wrapper
    catches it and re-raises as ``RuntimeError`` with an actionable message
    naming the key-mismatch + the fix (build the pretraining network with
    the same ``get_network_from_plans`` call). This is the AGENTS section 2
    no-silent-fallback discipline: surface the key-mismatch explicitly
    rather than silently partial-loading.
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    import torch

    from liom_toolkit.segmentation.vseg.ssl.pretrain import build_pretrain_network
    from liom_toolkit.segmentation.vseg.ssl.warmstart import load_pretrained_checkpoint

    net = build_pretrain_network(
        tiny_2d_resenc_plans,
        tiny_2d_resenc_dataset_json,
        configuration="2d",
        device=torch.device("cpu"),
    )
    # Build a deliberately mismatched checkpoint: rename every key so none
    # match the network's state_dict -> upstream load_pretrained_weights
    # raises AssertionError ("Key X is missing in the pretrained model").
    bad_weights = {f"WRONG.{k}": v for k, v in net.state_dict().items()}
    ckpt_path = tmp_path / "mismatched.pth"
    torch.save({"network_weights": bad_weights}, str(ckpt_path))

    with pytest.raises(RuntimeError) as exc_info:
        load_pretrained_checkpoint(str(ckpt_path), net)
    msg = str(exc_info.value)
    # The message must name the key-mismatch and the fix (actionable, not
    # just a re-raise of the raw AssertionError).
    assert "mismatch" in msg.lower() or "key" in msg.lower(), (
        f"RuntimeError message must name the key-mismatch, got: {msg}"
    )


@pytest.mark.ai
def test_validate_nnunet_env_raises_on_missing_vars(monkeypatch):
    """validate_nnunet_env raises RuntimeError naming the missing nnUNet_* env vars.

    The in-process ``NNUNetTrainer`` reads ``nnUNet_raw`` /
    ``nnUNet_preprocessed`` / ``nnUNet_results`` from the environment; if
    unset it raises or silently mislocates data. The helper validates they
    are set BEFORE instantiating the trainer and raises ``RuntimeError``
    naming the missing vars (explicit failure, no silent pass).
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")

    from liom_toolkit.segmentation.vseg.ssl.warmstart import validate_nnunet_env

    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        validate_nnunet_env()
    msg = str(exc_info.value)
    # The message must name the missing vars so the caller can act.
    assert "nnUNet_raw" in msg or "nnUNet_preprocessed" in msg or "nnUNet_results" in msg, (
        f"RuntimeError must name the missing nnUNet_* vars, got: {msg}"
    )


@pytest.mark.ai
def test_load_pretrained_checkpoint_raises_value_error_on_nonexistent_path(
    tiny_2d_resenc_plans, tiny_2d_resenc_dataset_json, tmp_path
):
    """A nonexistent checkpoint path raises ValueError with the offending path.

    The helper validates the checkpoint path exists before passing it to
    ``load_pretrained_weights`` (which would raise an opaque
    ``FileNotFoundError`` otherwise). The ``ValueError`` message includes
    the offending path so the caller can act (AGENTS section 2 -- include
    the offending value in the message).
    """
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2")
    import torch

    from liom_toolkit.segmentation.vseg.ssl.pretrain import build_pretrain_network
    from liom_toolkit.segmentation.vseg.ssl.warmstart import load_pretrained_checkpoint

    net = build_pretrain_network(
        tiny_2d_resenc_plans,
        tiny_2d_resenc_dataset_json,
        configuration="2d",
        device=torch.device("cpu"),
    )
    bad_path = str(tmp_path / "does_not_exist.pth")

    with pytest.raises(ValueError) as exc_info:
        load_pretrained_checkpoint(bad_path, net)
    assert bad_path in str(exc_info.value), (
        f"ValueError must include the offending path {bad_path!r}, got: {exc_info.value}"
    )


@pytest.mark.ai
def test_warmstart_module_has_no_assert_validation_statements():
    """warmstart.py contains no `assert` validation statements (AGENTS section 2).

    ``assert`` is stripped under ``python -O``; input validation must use
    ``if ...: raise ValueError(...)`` / ``RuntimeError(...)`` with the
    offending value in the message. Config-as-data test for the no-assert
    invariant (a repo-wide invariant, not a behavior assertion).
    """
    pytest.importorskip("torch")
    from pathlib import Path

    import liom_toolkit.segmentation.vseg.ssl.warmstart as warmstart_mod

    src = Path(warmstart_mod.__file__).read_text()
    code_lines = [line for line in src.splitlines() if not line.strip().startswith("#")]
    code = "\n".join(code_lines)
    assert " assert " not in code, (
        "warmstart.py must not use `assert` for validation (AGENTS section 2 -- "
        "stripped under python -O); use `if ...: raise ValueError/RuntimeError` instead"
    )


@pytest.mark.ai
def test_warmstart_does_not_import_nnunet_bridge():
    """warmstart.py does NOT import the subprocess nnunet_bridge (uses in-process nnunetv2).

    The subprocess ``nnunet_bridge.py`` is slated for deletion; the
    warm-start path uses the in-process ``nnunetv2.run.run_training`` API
    directly. Config-as-data test for the no-bridge-import invariant --
    checks import statements, not the bare substring (the module docstring
    references the bridge by name to explain why it is NOT used).
    """
    pytest.importorskip("torch")
    import re
    from pathlib import Path

    import liom_toolkit.segmentation.vseg.ssl.warmstart as warmstart_mod

    src = Path(warmstart_mod.__file__).read_text()
    # Match import / from-import lines that reference nnunet_bridge (a
    # docstring mention is fine; an actual import is the violation).
    import_lines = [
        line
        for line in src.splitlines()
        if re.match(r"^\s*(import |from )", line) and "nnunet_bridge" in line
    ]
    assert not import_lines, (
        "warmstart.py must NOT import nnunet_bridge (the subprocess bridge is "
        "superseded -- use in-process nnunetv2.run.run_training directly); "
        f"found import lines: {import_lines}"
    )
