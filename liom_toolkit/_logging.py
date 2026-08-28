"""One-call logging setup for notebook/library users.

Attaches a ``StreamHandler`` to the ``liom_toolkit`` logger (NOT the root
logger) so notebook/library consumers get visible log output without
configuring the root logger themselves. This is mutually-exclusive with the
CLI usage pattern: each CLI ``main()`` calls ``logging.basicConfig`` on the
root logger, which propagates records from the ``liom_toolkit`` logger via
its ``NullHandler``. Calling ``configure_logging`` from a notebook while a
CLI is also running would double-log; pick one pattern per process.

``configure_logging`` is idempotent: a second call removes the
``StreamHandler`` installed by the first call (tagged with
``_liom_toolkit_handler``) before attaching a new one, so repeated calls in
a notebook iteration loop do not stack handlers and double-log. The
``NullHandler`` attached at import time and any user-installed handlers are
preserved.

Example
-------
::

    import liom_toolkit
    liom_toolkit.configure_logging(level="INFO")
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO


def configure_logging(level: int | str = logging.INFO, stream: TextIO | None = None) -> None:
    """Attach a ``StreamHandler`` to the ``liom_toolkit`` logger.

    Parameters
    ----------
    level : int or str
        Logging level, either as an integer (``logging.DEBUG``) or a string
        name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``). String
        names are resolved via ``getattr(logging, level.upper())``.
    stream : TextIO, optional
        The stream the handler writes to. Defaults to ``sys.stderr`` when
        ``None``.

    Raises
    ------
    ValueError
        If ``level`` is a string that does not name a ``logging`` level
        attribute (e.g. ``"NOT_A_LEVEL"``).
    TypeError
        If ``level`` is a string that names a ``logging`` attribute which is
        not an integer level constant (defense-in-depth; not reachable with
        the standard ``logging`` module attributes).
    """
    if isinstance(level, str):
        upper = level.upper()
        if not hasattr(logging, upper):
            raise ValueError(f"Unknown log level: {level!r}")
        resolved = getattr(logging, upper)
        if not isinstance(resolved, int):
            raise TypeError(f"Resolved log level is not an integer: {level!r}")
        level = resolved
    logger = logging.getLogger("liom_toolkit")
    # Remove handlers added by a previous configure_logging call so
    # repeated calls do not stack handlers (a common notebook iteration
    # pattern — without this, every record is emitted once per stacked
    # handler). Only handlers we tagged ourselves are removed; the
    # NullHandler attached at import time and any user-installed handlers
    # are preserved.
    for h in list(logger.handlers):
        if getattr(h, "_liom_toolkit_handler", False):
            logger.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler._liom_toolkit_handler = True
    logger.addHandler(handler)
    logger.setLevel(level)
