"""CLI entry points for the LIOM Toolkit.

Each ``liom_*`` module defines ``_build_argument_parser()`` and ``main()``
and is registered as a console script in ``pyproject.toml``. The shared
cross-cutting flags live in ``_common.py`` and are consumed via
``parents=[build_common_parser()]``.
"""
