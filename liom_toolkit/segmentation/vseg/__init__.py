"""PyTorch U-Net vessel segmentation subpackage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .prediction import predict_one, predict_volume

if TYPE_CHECKING:
    # Type-checker visibility only; the name is NOT in __all__ on purpose
    # (see below), so the bare import reads as unused to the linter.
    from .model_v2 import NnUnetV2Model  # ruff: ignore[unused-import]

# ``NnUnetV2Model`` is deliberately absent from ``__all__``: it is [ai]-gated
# (the class's module-top torch guard raises ImportError when the extra is
# missing), so keeping it out of the curated list keeps
# ``from liom_toolkit.segmentation.vseg import *`` safe on core-only installs.
# The name still resolves on attribute access via ``__getattr__`` below.
__all__ = ["predict_one", "predict_volume"]


def __getattr__(name: str) -> Any:
    """Lazy-import the nnU-Net v2 wrapper so the barrel stays heavy-dep-free.

    ``model_v2`` carries a module-top torch guard; deferring the import to
    attribute access keeps ``import liom_toolkit.segmentation.vseg`` working
    on core-only installs (mirrors the ``ssl/__init__.py`` barrel pattern).

    Returns
    -------
    Any
        The requested attribute.

    Raises
    ------
    AttributeError
        If ``name`` is not a lazily-exported symbol.
    """
    if name == "NnUnetV2Model":
        from .model_v2 import NnUnetV2Model

        return NnUnetV2Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
