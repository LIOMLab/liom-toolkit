"""Self-supervised pretraining utilities for the nnU-Net v2 2D vessel-segmentation model.

This subpackage implements the masked-inpainting pretraining loop and the
warm-start transfer path that loads the pretrained encoder into the nnU-Net
2D ResEnc trainer. It is a lab reproducibility utility, not a shipped library
function: it stays OUT of the eager ``vseg/__init__.py`` import chain so a
plain ``import liom_toolkit`` (or ``from liom_toolkit.segmentation.vseg
import predict_one``) never pulls torch / MONAI / nnunetv2.

The torch / MONAI / nnunetv2 lazy-import guards live INSIDE each module of
this subpackage (per-module ``try/except ImportError: raise
ImportError("install liom-toolkit[ai,benchmark]")``), NOT in this barrel.
The barrel re-exports only the high-level entry points (the corpus builder,
the pretraining loop, and the warm-start helper).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .compare import run_comparison
    from .corpus import SSLCorpus, extract_plane_slice, mip_qc, z_score_per_channel
    from .masking import vessel_aware_block_mask, vesselness_probability_map
    from .pretrain import build_pretrain_network, masked_inpainting_pretrain
    from .warmstart import load_pretrained_checkpoint, validate_nnunet_env, warm_start

__all__ = [
    "SSLCorpus",
    "build_pretrain_network",
    "extract_plane_slice",
    "load_pretrained_checkpoint",
    "masked_inpainting_pretrain",
    "mip_qc",
    "run_comparison",
    "validate_nnunet_env",
    "vessel_aware_block_mask",
    "vesselness_probability_map",
    "warm_start",
    "z_score_per_channel",
]


def __getattr__(name: str) -> Any:
    """Lazy-import the SSL symbols so the barrel imports without the [ai] extra.

    Each module's top-level torch import guard raises
    ``ImportError("install liom-toolkit[ai,benchmark]")`` when torch is
    absent; deferring the import to attribute access keeps that honest signal
    at call time rather than at ``import liom_toolkit.segmentation.vseg.ssl``
    time (mirrors the ``vseg/benchmark/__init__.py`` barrel pattern).

    Returns
    -------
    Any
        The requested SSL symbol.

    Raises
    ------
    AttributeError
        If ``name`` is not a curated SSL symbol.
    """
    if name in {
        "SSLCorpus",
        "extract_plane_slice",
        "mip_qc",
        "z_score_per_channel",
    }:
        from . import corpus

        return getattr(corpus, name)
    if name in {"build_pretrain_network", "masked_inpainting_pretrain"}:
        from . import pretrain

        return getattr(pretrain, name)
    if name in {"vessel_aware_block_mask", "vesselness_probability_map"}:
        from . import masking

        return getattr(masking, name)
    if name in {"load_pretrained_checkpoint", "validate_nnunet_env", "warm_start"}:
        from . import warmstart

        return getattr(warmstart, name)
    if name == "run_comparison":
        from . import compare

        return getattr(compare, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
