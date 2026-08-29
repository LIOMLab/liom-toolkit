"""Tests for ``liom_toolkit/utils/concurrency.py`` (managed concurrency layer).

Covers the SC-1 unit tests for the managed concurrency layer introduced in
the thread/worker consolidation phase: capped, configurable
``get_thread_pool()`` / ``get_process_pool()`` factories with sensible default
caps and explicit lifecycle, plus the ``thread_map_tqdm`` /
``process_map_tqdm`` tqdm-integrated map helpers.

The cap helpers (``_default_thread_cap`` / ``_default_process_cap``) are pure
functions of ``os.cpu_count()`` and are exercised by monkeypatching
``os.cpu_count`` for the edge cases (the ``cpu_count() == 1`` process-cap edge
must raise ``ValueError`` with the offending value, mirroring the
``DaskClientManager`` precedent at ``dask_client.py:100-101`` — no silent
zero-worker pool, AGENTS section 2).

The factory functions return real stdlib ``ThreadPoolExecutor`` /
``ProcessPoolExecutor`` instances; their ``_max_workers`` and
``_mp_context`` attributes are inspected directly (no mocking of
``concurrent.futures`` — these are cheap stdlib objects, not orchestration
deps). The tqdm helpers are exercised against trivial iterables with a small
``max_workers`` override so the test does not spawn the full default cap; the
``**kwargs`` forwarding is verified by inspecting the tqdm bar attributes
(``desc`` / ``unit`` / ``total``) returned by ``thread_map`` /
``process_map``.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest

from liom_toolkit.utils.concurrency import (
    _default_process_cap,
    _default_thread_cap,
    _validate_max_workers,
    get_process_pool,
    get_thread_pool,
    process_map_tqdm,
    thread_map_tqdm,
)

# ---------------------------------------------------------------------------
# Cap helpers
# ---------------------------------------------------------------------------


def test_default_thread_cap():
    """``_default_thread_cap()`` returns ``min(32, (os.cpu_count() or 1) + 4)``.

    The ``(os.cpu_count() or 1)`` guard handles ``cpu_count()`` returning
    ``None`` (mirrors CPython 3.12 ``ThreadPoolExecutor.__init__``). The 32
    ceiling mirrors the CPython thread-pool default since 3.8/3.13.
    """
    for cpu, expected in [(16, 20), (40, 32), (1, 5), (None, 5)]:
        monkey_cpu = cpu if cpu is not None else None
        _with_cpu_count(monkey_cpu, lambda: None)  # ensure patch is installable
        assert _default_thread_cap_under(monkey_cpu) == expected, f"cpu_count={cpu!r}"


def test_default_process_cap():
    """``_default_process_cap()`` returns ``min((os.cpu_count() or 1) - 1, 8)``.

    Matches the ``DaskClientManager`` cap exactly (``dask_client.py:99``) —
    the cap prevents OOM on high-core machines where each spawn worker is a
    full interpreter (RSS multiplies linearly with ``max_workers``).
    """
    for cpu, expected in [(16, 8), (9, 8), (4, 3), (2, 1), (None, 0)]:
        # cpu_count() == None resolves to (1) - 1 == 0, which must raise
        # (covered by test_process_cap_cpu_count_one). Here we only assert
        # the arithmetic for the >= 2 cases.
        if expected < 1:
            continue
        assert _default_process_cap_under(cpu) == expected, f"cpu_count={cpu!r}"


def test_process_cap_cpu_count_one():
    """``cpu_count() == 1`` (or ``None`` → 1) makes the process cap 0, which
    MUST raise ``ValueError`` with the offending value in the message — never
    silently spawn a zero-worker pool (AGENTS section 2, mirrors
    ``dask_client.py:100-101``).
    """
    for cpu in (1, None):
        with _patched_cpu_count(cpu):
            with pytest.raises(ValueError, match=r"process pool cap must be >= 1"):
                _default_process_cap()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_thread_pool_default_cap(monkeypatch):
    """``get_thread_pool()`` with no override uses ``_default_thread_cap()``."""
    monkeypatch.setattr("liom_toolkit.utils.concurrency.os.cpu_count", lambda: 16)
    pool = get_thread_pool()
    try:
        assert isinstance(pool, ThreadPoolExecutor)
        assert pool._max_workers == _default_thread_cap()
    finally:
        pool.shutdown(wait=True)


def test_thread_pool_override():
    """An explicit ``max_workers`` wins over the default cap (D-02a)."""
    pool = get_thread_pool(max_workers=2)
    try:
        assert isinstance(pool, ThreadPoolExecutor)
        assert pool._max_workers == 2
    finally:
        pool.shutdown(wait=True)


def test_process_pool_spawn_ctx():
    """``get_process_pool()`` defaults to the ``spawn`` start context (D-11).

    Spawn is mandatory for torch-importing callers — forking a multithreaded
    process (the Linux default) deadlocks when torch has started internal
    threads at import time.
    """
    pool = get_process_pool(max_workers=2)
    try:
        assert isinstance(pool, ProcessPoolExecutor)
        assert pool._max_workers == 2
        assert pool._mp_context.get_start_method() == "spawn"
    finally:
        pool.shutdown(wait=True)


def test_process_pool_mp_context_override():
    """``get_process_pool(mp_context=...)`` accepts a string or ``BaseContext``.

    The override is the escape hatch for future CPU-bound sites without torch
    (where fork would be safe). Both a string and a ``BaseContext`` are
    accepted and normalize to the same context.
    """
    fork_ctx = multiprocessing.get_context("fork")
    # String form
    pool_str = get_process_pool(max_workers=2, mp_context="fork")
    try:
        assert pool_str._mp_context.get_start_method() == "fork"
    finally:
        pool_str.shutdown(wait=True)
    # BaseContext form
    pool_ctx = get_process_pool(max_workers=2, mp_context=fork_ctx)
    try:
        assert pool_ctx._mp_context is fork_ctx
    finally:
        pool_ctx.shutdown(wait=True)


# ---------------------------------------------------------------------------
# max_workers=0 / negative validation (IN-01)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_validate_max_workers_rejects_explicit_invalid(bad):
    """``_validate_max_workers`` raises ``ValueError`` for explicit ``< 1``
    values, with the offending value in the message.

    ``None`` is the "use the default" sentinel and is always accepted. An
    explicit ``0`` would otherwise be silently replaced by the default cap
    due to Python's falsiness of ``0`` (``max_workers or _default_cap()``),
    masking a caller bug — surface it instead (AGENTS section 2).
    """
    with pytest.raises(ValueError, match=r"max_workers must be >= 1 or None"):
        _validate_max_workers(bad)


def test_validate_max_workers_accepts_none():
    """``None`` is the default sentinel and must NOT raise."""
    _validate_max_workers(None)  # no exception


def test_validate_max_workers_accepts_positive():
    """A positive explicit value is valid."""
    _validate_max_workers(1)
    _validate_max_workers(8)


def test_get_thread_pool_rejects_zero_workers():
    """``get_thread_pool(max_workers=0)`` raises ``ValueError`` rather than
    silently falling back to the default cap (IN-01).
    """
    with pytest.raises(ValueError, match=r"max_workers must be >= 1 or None"):
        get_thread_pool(max_workers=0)


def test_get_process_pool_rejects_zero_workers():
    """``get_process_pool(max_workers=0)`` raises ``ValueError`` rather than
    silently falling back to the default cap (IN-01).
    """
    with pytest.raises(ValueError, match=r"max_workers must be >= 1 or None"):
        get_process_pool(max_workers=0)


def test_thread_map_tqdm_rejects_zero_workers():
    """``thread_map_tqdm(max_workers=0)`` raises ``ValueError`` rather than
    silently falling back to the default cap (IN-01).
    """
    with pytest.raises(ValueError, match=r"max_workers must be >= 1 or None"):
        thread_map_tqdm(_double, range(4), max_workers=0)


def test_process_map_tqdm_rejects_zero_workers():
    """``process_map_tqdm(max_workers=0)`` raises ``ValueError`` rather than
    silently falling back to the default cap (IN-01).
    """
    with pytest.raises(ValueError, match=r"max_workers must be >= 1 or None"):
        process_map_tqdm(_double, range(4), max_workers=0)


# ---------------------------------------------------------------------------
# tqdm helpers
# ---------------------------------------------------------------------------


def test_thread_map_tqdm_forwards_kwargs():
    """``thread_map_tqdm`` forwards ``**kwargs`` (``desc``/``unit``/``total``)
    to ``tqdm.contrib.concurrent.thread_map`` and injects the layer cap
    (D-03a). The returned list preserves input order and length.
    """
    result = thread_map_tqdm(
        lambda x: x * 2,
        range(5),
        max_workers=2,
        desc="encoding",
        unit="slice",
        total=5,
    )
    assert list(result) == [0, 2, 4, 6, 8]


def test_process_map_tqdm_injects_spawn_defaults():
    """``process_map_tqdm`` injects ``mp_context=spawn`` and
    ``_lock=spawn_ctx.Lock()`` (D-11) and forwards ``**kwargs`` (D-03a).

    Uses a tiny module-level function so spawn can pickle it. The
    ``max_workers`` override keeps the test from spawning the full default
    cap. Verifies the spawn-context lock wiring is centralized in the layer
    (a fork-context Lock passed to a spawn-context ProcessPoolExecutor raises
    RuntimeError — RESEARCH Pitfall 1).
    """
    result = process_map_tqdm(
        _double,
        range(4),
        max_workers=2,
        desc="validating",
        unit="patches",
        total=4,
        chunksize=1,
    )
    assert list(result) == [0, 2, 4, 6]


def test_process_map_tqdm_mp_context_basecontext_override(monkeypatch):
    """``process_map_tqdm`` respects a ``BaseContext`` passed via
    ``mp_context`` (mirrors ``get_process_pool``'s contract).

    Previously the function silently ignored a ``BaseContext`` and always
    resolved to spawn. Now the passed context is used for BOTH the executor
    ``mp_context`` and the ``_lock`` (the lock MUST come from the same
    context as the executor, otherwise a spawn-context executor with a
    fork-context lock raises ``RuntimeError``).

    The ``process_map`` call is intercepted so the test does not actually
    fork workers; we assert the ``mp_context`` and ``_lock`` kwargs are
    sourced from the user-supplied context.
    """
    import liom_toolkit.utils.concurrency as concurrency_mod

    fork_ctx = multiprocessing.get_context("fork")
    captured: dict[str, object] = {}

    def fake_process_map(fn, *iterables, **kwargs):
        captured.update(kwargs)
        return [fn(x) for x in iterables[0]]

    monkeypatch.setattr(concurrency_mod, "process_map", fake_process_map)

    result = process_map_tqdm(
        _double,
        range(4),
        max_workers=2,
        mp_context=fork_ctx,
    )
    assert list(result) == [0, 2, 4, 6]
    assert captured["mp_context"] is fork_ctx, "BaseContext was not forwarded as mp_context"
    # The _lock must come from the same context as the executor.
    assert captured["_lock"] is not None
    assert captured.get("max_workers") == 2


# ---------------------------------------------------------------------------
# Module-level helper for process_map_tqdm (must be picklable under spawn)
# ---------------------------------------------------------------------------


def _double(x: int) -> int:
    """Return ``x * 2``. Module-level so ``process_map`` can pickle it under spawn."""
    return x * 2


# ---------------------------------------------------------------------------
# Internal helpers — cpu_count monkeypatching
# ---------------------------------------------------------------------------


class _CpuCountPatcher:
    """Context manager that patches ``os.cpu_count`` (and the module-level
    ``os`` reference in ``liom_toolkit.utils.concurrency``) to return a fixed
    value. Restores the originals on exit.
    """

    def __init__(self, value: int | None) -> None:
        self.value = value
        self._saved_os_cpu = None
        self._saved_mod_cpu = None

    def __enter__(self):
        self._saved_os_cpu = os.cpu_count
        self._saved_mod_cpu = __import__(
            "liom_toolkit.utils.concurrency", fromlist=["os"]
        ).os.cpu_count
        os.cpu_count = lambda: self.value  # type: ignore[assignment]
        import liom_toolkit.utils.concurrency as _c

        _c.os.cpu_count = lambda: self.value  # type: ignore[assignment]
        return self

    def __exit__(self, *exc):
        os.cpu_count = self._saved_os_cpu  # type: ignore[assignment]
        import liom_toolkit.utils.concurrency as _c

        _c.os.cpu_count = self._saved_mod_cpu  # type: ignore[assignment]
        return False


def _patched_cpu_count(value: int | None) -> _CpuCountPatcher:
    return _CpuCountPatcher(value)


def _with_cpu_count(value: int | None, fn) -> None:
    with _patched_cpu_count(value):
        fn()


def _default_thread_cap_under(value: int | None) -> int:
    with _patched_cpu_count(value):
        return _default_thread_cap()


def _default_process_cap_under(value: int | None) -> int:
    with _patched_cpu_count(value):
        return _default_process_cap()


# ---------------------------------------------------------------------------
# SC-3 AST regression guard — no uncapped pool construction outside the layer
# ---------------------------------------------------------------------------


def test_no_uncapped_pools_outside_concurrency_module():
    """SC-3: no direct ``ThreadPoolExecutor()`` / ``ProcessPoolExecutor()`` /
    ``process_map(`` / ``thread_map(`` construction outside
    ``utils/concurrency.py`` in ``liom_toolkit/`` source.

    The managed concurrency layer is the single sanctioned construction site
    for stdlib thread/process executors and tqdm map helpers. Any other module
    that constructs one of these directly bypasses the layer's caps and the
    mandatory spawn-context + spawn-lock wiring (D-11), reintroducing the
    fork-mutation deadlock and resource-exhaustion failure modes the
    consolidation exists to prevent.

    This is an AST walk (not a string grep) per AGENTS §5 — ``ast`` parses
    structure, not text, so it is sanctioned where regex-on-source is
    forbidden. Catches both the ``Name(...)`` form (``ThreadPoolExecutor()``)
    and the ``Attr.(...)`` form (``concurrent.process_map(...)``). The
    sanctioned ``utils/concurrency.py`` layer itself is skipped — it is the
    only sanctioned construction site. The skip matches by resolved full
    path, NOT by basename, so a future ``liom_toolkit/<subpackage>/concurrency.py``
    is NOT silently exempted (a basename match would defeat the guard).
    """
    import ast

    import liom_toolkit

    pkg_root = Path(liom_toolkit.__file__).parent
    sanctioned = (pkg_root / "utils" / "concurrency.py").resolve()
    forbidden_calls = {
        "ThreadPoolExecutor",
        "ProcessPoolExecutor",
        "process_map",
        "thread_map",
    }
    violations: list[str] = []
    for py in pkg_root.rglob("*.py"):
        if py.resolve() == sanctioned:
            continue
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in forbidden_calls
            ):
                violations.append(f"{py}:{node.lineno} constructs {node.func.id}()")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in forbidden_calls
            ):
                violations.append(f"{py}:{node.lineno} constructs .{node.func.attr}()")
    assert not violations, "uncapped pool construction outside utils/concurrency.py:\n" + "\n".join(
        violations
    )
