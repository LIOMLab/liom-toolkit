"""Tests for the LOG-01 logging foundation (``liom_toolkit/_logging.py``).

Covers the library-side logging contract:

* ``import liom_toolkit`` attaches a ``NullHandler`` to the ``liom_toolkit``
  logger and does NOT touch the root logger (zero import side effects for
  notebook/library consumers).
* ``liom_toolkit.configure_logging(level=...)`` attaches a ``StreamHandler``
  to the ``liom_toolkit`` logger (NOT root), sets its level, accepts both
  string and integer levels, honours a custom ``stream``, and raises
  ``ValueError`` on an unknown level string.
* ``liom_toolkit.configure_logging`` is re-exported from the package root.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pytest

import liom_toolkit


def _liom_logger() -> logging.Logger:
    return logging.getLogger("liom_toolkit")


def test_null_handler() -> None:
    """``import liom_toolkit`` attaches a NullHandler to the liom_toolkit logger."""
    assert any(isinstance(h, logging.NullHandler) for h in _liom_logger().handlers)


def test_configure_logging_attaches_stream_handler() -> None:
    """configure_logging adds a StreamHandler to the liom_toolkit logger, not root."""
    root_before = list(logging.getLogger().handlers)
    logger = _liom_logger()
    handlers_before = list(logger.handlers)
    try:
        liom_toolkit.configure_logging(level="INFO")
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        # Root logger handlers unchanged by configure_logging.
        assert logging.getLogger().handlers == root_before
    finally:
        # Restore: drop the handlers we just added so the test is hermetic.
        logger.handlers[:] = handlers_before


def test_configure_logging_sets_level() -> None:
    """configure_logging(level='DEBUG') sets the liom_toolkit logger level to DEBUG."""
    logger = _liom_logger()
    original_level = logger.level
    try:
        liom_toolkit.configure_logging(level="DEBUG")
        assert logger.level == logging.DEBUG
    finally:
        logger.setLevel(original_level)


def test_configure_logging_int_level() -> None:
    """configure_logging accepts an integer level."""
    logger = _liom_logger()
    original_level = logger.level
    handlers_before = list(logger.handlers)
    try:
        liom_toolkit.configure_logging(level=logging.WARNING)
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(original_level)
        logger.handlers[:] = handlers_before


def test_configure_logging_unknown_level_raises() -> None:
    """configure_logging('NOT_A_LEVEL') raises ValueError."""
    with pytest.raises(ValueError):
        liom_toolkit.configure_logging("NOT_A_LEVEL")


def test_configure_logging_custom_stream() -> None:
    """passing stream=io.StringIO() makes the handler write to that stream."""
    logger = _liom_logger()
    handlers_before = list(logger.handlers)
    original_level = logger.level
    stream = io.StringIO()
    try:
        liom_toolkit.configure_logging(level=logging.INFO, stream=stream)
        matching = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream is stream
        ]
        assert len(matching) == 1
    finally:
        logger.handlers[:] = handlers_before
        logger.setLevel(original_level)


def test_no_root_handler_on_import() -> None:
    """After import, the root logger has no StreamHandler added by liom_toolkit.

    The library attaches only a NullHandler to the ``liom_toolkit`` logger; it
    must not install any handler on the root logger (the application / CLI
    owns root configuration via ``basicConfig``).
    """
    # The liom_toolkit format string is "%(levelname)s %(name)s: %(message)s";
    # no root handler should carry that formatter (which would indicate the
    # library leaked a handler onto root).
    liom_fmt = "%(levelname)s %(name)s: %(message)s"
    for h in logging.getLogger().handlers:
        fmt = h.formatter
        if fmt is not None and fmt._fmt == liom_fmt:  # type: ignore[attr-defined]
            raise AssertionError(
                f"Root logger has a handler with the liom_toolkit format string: {h}"
            )


def test_configure_logging_exported() -> None:
    """liom_toolkit.configure_logging is callable and re-exported from the package."""
    assert hasattr(liom_toolkit, "configure_logging")
    assert callable(liom_toolkit.configure_logging)


# --- print→logger sweep behavior tests (LOG-01) ---------------------------
#
# The full print-free enforcement is the ruff T20 hard-gate (dropped in the
# CLI task). These caplog tests verify the sweep actually wired module-level
# loggers into the domain functions that previously used print(). Only
# core-dep-only functions are exercised here; the templating debug print
# (ants extra) and the training finished message (torch extra) are gated on
# optional deps and verified by the T20 gate + grep audit.


def test_volume_segmentation_logs_stage(
    caplog: pytest.LogCaptureFixture, synthetic_volume: np.ndarray
) -> None:
    """segment_3d emits 'Segmenting 3D volume...' at INFO on its module logger."""
    from liom_toolkit.segmentation.volume_segmentation import segment_3d

    with caplog.at_level(logging.INFO, logger="liom_toolkit.segmentation.volume_segmentation"):
        segment_3d(synthetic_volume, k=3, fill_holes=False)
    messages = [r.getMessage() for r in caplog.records]
    assert "Segmenting 3D volume..." in messages
    stage_records = [r for r in caplog.records if r.getMessage() == "Segmenting 3D volume..."]
    assert all(r.levelno == logging.INFO for r in stage_records)
    assert all(r.name == "liom_toolkit.segmentation.volume_segmentation" for r in stage_records)


def test_volume_segmentation_logs_threshold_stage(
    caplog: pytest.LogCaptureFixture, synthetic_volume: np.ndarray
) -> None:
    """segment_3d emits the 'Thresholding image...' stage announcement at INFO."""
    from liom_toolkit.segmentation.volume_segmentation import segment_3d

    with caplog.at_level(logging.INFO, logger="liom_toolkit.segmentation.volume_segmentation"):
        segment_3d(synthetic_volume, k=3, fill_holes=False)
    messages = [r.getMessage() for r in caplog.records]
    assert "Thresholding image..." in messages
