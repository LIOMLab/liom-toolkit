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
The barrel re-exports only the high-level entry points as later plans add
them; for now only the corpus builder is wired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .corpus import SSLCorpus, extract_plane_slice, mip_qc, z_score_per_channel

__all__ = ["SSLCorpus", "extract_plane_slice", "mip_qc", "z_score_per_channel"]


def __getattr__(name: str) -> Any:
    """Lazy-import the corpus symbols so the barrel imports without the [ai] extra.

    The corpus module's top-level torch import guard raises
    ``ImportError("install liom-toolkit[ai,benchmark]")`` when torch is
    absent; deferring the import to attribute access keeps that honest signal
    at call time rather than at ``import liom_toolkit.segmentation.vseg.ssl``
    time (mirrors the ``vseg/benchmark/__init__.py`` barrel pattern).

    Returns
    -------
    Any
        The requested corpus symbol (``SSLCorpus``, ``extract_plane_slice``,
        ``mip_qc``, or ``z_score_per_channel``).

    Raises
    ------
    AttributeError
        If ``name`` is not a curated corpus symbol.
    """
    if name in {"SSLCorpus", "extract_plane_slice", "mip_qc", "z_score_per_channel"}:
        from . import corpus

        return getattr(corpus, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
