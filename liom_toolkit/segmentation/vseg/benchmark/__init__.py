"""Empirical benchmark harness for the vessel-segmentation architecture decision."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .run import run_benchmark

__all__ = ["run_benchmark"]


def __getattr__(name: str) -> Any:
    """Lazy-import run_benchmark so the barrel imports without the [ai] extra.

    Returns
    -------
    Any
        The ``run_benchmark`` callable when ``name == "run_benchmark"``.

    Raises
    ------
    AttributeError
        If ``name`` is not ``"run_benchmark"``.
    """
    if name == "run_benchmark":
        from .run import run_benchmark

        return run_benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
