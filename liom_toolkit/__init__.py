"""LIOM Toolkit: processing and analysis of light-sheet fluorescence microscopy data.

For one-call logging setup in notebooks::

    import liom_toolkit
    liom_toolkit.configure_logging(level="INFO")
"""

from __future__ import annotations

import logging

# Library-side: NullHandler only (Python logging HOWTO). Zero import side
# effects for notebook/library consumers — the application (each CLI main())
# configures real handlers via basicConfig.
logging.getLogger(__name__).addHandler(logging.NullHandler())  # ruff: ignore[non-empty-init-module] — sanctioned library logger setup

from ._logging import (  # ruff: ignore[module-import-not-at-top-of-file] — deliberate scoped exception to AGENTS §7
    configure_logging,
)

__all__ = ["configure_logging"]
