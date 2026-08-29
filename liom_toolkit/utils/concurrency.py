"""Managed concurrency layer — capped thread/process pool factories + tqdm-integrated map helpers.

This module is the single source of truth for stdlib ``concurrent.futures``
pool policy in the package. It replaces the two ad-hoc sites (the uncapped
``ThreadPoolExecutor`` in ``utils/io.py`` PNG-encode and the inline
``process_map`` in ``segmentation/vseg/dataset.py`` patch validation) with
factory functions and tqdm wrappers that inject type-specific default caps,
a mandatory spawn context for process pools, and the spawn-context lock that
tqdm's ``ensure_lock`` would otherwise source from the (fork) default context.

Design notes
------------
- **Factory functions, no singleton** (D-01): stdlib pools are cheap and
  just-in-time; a module-level singleton process pool would leak torch/ants
  worker state across unrelated call sites. The caller owns the
  ``with``/``close()`` lifecycle.
- **Type-specific caps** (D-02): the thread cap mirrors CPython's own
  ``ThreadPoolExecutor`` default (``min(32, cpu+4)`` — I/O-bound, GIL-released
  C encode); the process cap mirrors ``DaskClientManager``
  (``min(cpu-1, 8)`` — CPU-bound, each spawn worker is a full interpreter).
  The ``cpu_count() == 1`` edge (process cap → 0) raises ``ValueError`` with
  the offending value — never a silent zero-worker pool.
- **No env-var override** (D-02b): a hidden global tunable is a
  reproducibility hazard in a project whose dominant failure mode is silent
  data corruption. CPython 3.13's ``PYTHON_CPU_COUNT`` already provides a
  container-resource escape hatch at a lower layer.
- **Spawn context mandatory for process pools** (D-11): ``dataset.py`` imports
  torch at module top, torch starts internal threads at import time, and
  forking a multithreaded process on Linux (the default start method)
  deadlocks. The spawn-context ``_lock`` is also load-bearing — tqdm's
  ``ensure_lock`` would otherwise create a fork-context SemLock, and passing
  that to a spawn-context ``ProcessPoolExecutor`` raises ``RuntimeError``.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

from tqdm.contrib.concurrent import process_map, thread_map

# ``StartContext`` is the runtime type returned by
# ``multiprocessing.get_context`` and is the documented parameter type for
# ``ProcessPoolExecutor(mp_context=...)``. It is not exposed as a public
# attribute on the ``multiprocessing`` stub (ty cannot resolve it), so we
# alias it here once and use ``StartContext`` throughout.
StartContext = multiprocessing.context.BaseContext


def _default_thread_cap() -> int:
    """Return the default thread-pool cap: ``min(32, (os.cpu_count() or 1) + 4)``.

    The ``(os.cpu_count() or 1)`` guard handles ``cpu_count()`` returning
    ``None`` (mirrors CPython 3.12 ``ThreadPoolExecutor.__init__``). The 32
    ceiling mirrors the CPython thread-pool default since 3.8/3.13 — I/O-bound
    work where a thread costs ~8KB stack and contends on nothing the C code
    holds.

    Returns
    -------
    int
        The default thread-pool cap (``>= 5`` on any machine).
    """
    return min(32, (os.cpu_count() or 1) + 4)


def _default_process_cap() -> int:
    """Return the default process-pool cap: ``min((os.cpu_count() or 1) - 1, 8)``.

    Matches the ``DaskClientManager`` cap exactly (``dask_client.py:99``) —
    each spawn worker is a full interpreter whose RSS multiplies linearly with
    ``max_workers``, so the cap prevents OOM on high-core machines.

    Returns
    -------
    int
        The default process-pool cap (``>= 1``; raises before returning
        otherwise).

    Raises
    ------
    ValueError
        If the computed cap is less than 1 (covers ``cpu_count() == 1`` and
        ``cpu_count() is None`` → 0). The offending value is included in the
        message — never silently spawn a zero-worker pool (AGENTS section 2).
    """
    cap = min((os.cpu_count() or 1) - 1, 8)
    if cap < 1:
        raise ValueError(f"process pool cap must be >= 1, got {cap}")
    return cap


def get_thread_pool(max_workers: int | None = None) -> ThreadPoolExecutor:
    """Return a ``ThreadPoolExecutor`` with the layer's default cap applied.

    The caller owns the ``with``/``close()`` lifecycle (D-01 — no wrapper
    state, no singleton).

    Parameters
    ----------
    max_workers : int, optional
        Explicit override. ``None`` (the default) resolves to
        ``_default_thread_cap()``; an explicit value wins (D-02a).

    Returns
    -------
    ThreadPoolExecutor
        A configured stdlib thread pool. The caller is responsible for
        shutting it down (``with`` block or ``shutdown()``).
    """
    return ThreadPoolExecutor(max_workers=max_workers or _default_thread_cap())


def get_process_pool(
    max_workers: int | None = None,
    mp_context: str | StartContext = "spawn",
) -> ProcessPoolExecutor:
    """Return a ``ProcessPoolExecutor`` with the layer's default cap + spawn context.

    The caller owns the ``with``/``close()`` lifecycle (D-01 — no wrapper
    state, no singleton).

    Parameters
    ----------
    max_workers : int, optional
        Explicit override. ``None`` (the default) resolves to
        ``_default_process_cap()``; an explicit value wins (D-02a).
    mp_context : str or StartContext, optional
        Start context for the worker processes. Defaults to ``"spawn"``
        (D-11 — mandatory for torch-importing callers; forking a
        multithreaded process deadlocks on Linux). Accepts a string (resolved
        via ``multiprocessing.get_context``) or a ``BaseContext`` for
        fork-safe future sites.

    Returns
    -------
    ProcessPoolExecutor
        A configured stdlib process pool. The caller is responsible for
        shutting it down (``with`` block or ``shutdown()``).
    """
    ctx = multiprocessing.get_context(mp_context) if isinstance(mp_context, str) else mp_context
    return ProcessPoolExecutor(
        max_workers=max_workers or _default_process_cap(),
        mp_context=ctx,
    )


def thread_map_tqdm(
    fn: Any,
    *iterables: Any,
    max_workers: int | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Wrap ``tqdm.contrib.concurrent.thread_map`` with the layer's thread cap.

    Equivalent to ``thread_map(fn, *iterables, max_workers=cap, **kwargs)``.
    Forwards ``**kwargs`` (``desc``, ``unit``, ``total``, ``position``,
    ``leave``) to tqdm so callers keep the full tqdm ergonomics (D-03a).

    Parameters
    ----------
    fn : callable
        Function applied to each element of the iterables.
    *iterables
        One or more iterables passed to ``fn``.
    max_workers : int, optional
        Explicit override. ``None`` (the default) resolves to
        ``_default_thread_cap()``; an explicit value wins (D-02a).
    **kwargs
        Forwarded verbatim to ``tqdm.contrib.concurrent.thread_map``.

    Returns
    -------
    list
        Results in input order (``thread_map`` preserves order).
    """
    return thread_map(
        fn,
        *iterables,
        max_workers=max_workers or _default_thread_cap(),
        **kwargs,
    )


def process_map_tqdm(
    fn: Any,
    *iterables: Any,
    max_workers: int | None = None,
    mp_context: str | StartContext = "spawn",
    **kwargs: Any,
) -> list[Any]:
    """Wrap ``tqdm.contrib.concurrent.process_map`` with the layer's cap + spawn lock.

    Equivalent to ``process_map(fn, *iterables, max_workers=cap,
    mp_context=ctx, _lock=ctx.Lock(), **kwargs)``. The ``_lock`` is
    load-bearing (D-11): tqdm's ``ensure_lock`` would otherwise create a
    fork-context SemLock, and passing that to a spawn-context
    ``ProcessPoolExecutor`` raises ``RuntimeError``.

    Forwards ``**kwargs`` (``desc``, ``unit``, ``total``, ``position``,
    ``leave``, ``chunksize``) to tqdm (D-03a). ``chunksize`` has NO layer
    default — it is a per-call kwarg (chunksize is workload-specific,
    RESEARCH Open Q 2).

    Parameters
    ----------
    fn : callable
        Function applied to each element of the iterables. Must be picklable
        under spawn (module-level function, not a lambda/closure).
    *iterables
        One or more iterables passed to ``fn``.
    max_workers : int, optional
        Explicit override. ``None`` (the default) resolves to
        ``_default_process_cap()``; an explicit value wins (D-02a).
    mp_context : str or StartContext, optional
        Start context. Defaults to ``"spawn"`` (D-11). The ``_lock`` is
        sourced from the SAME context as ``mp_context`` — the lock and
        executor context must match (a fork-context lock passed to a
        spawn-context executor raises ``RuntimeError``).
    **kwargs
        Forwarded verbatim to ``tqdm.contrib.concurrent.process_map``.

    Returns
    -------
    list
        Results in input order (``process_map`` preserves order).
    """
    ctx = multiprocessing.get_context(mp_context) if isinstance(mp_context, str) else mp_context
    return process_map(
        fn,
        *iterables,
        max_workers=max_workers or _default_process_cap(),
        mp_context=ctx,
        _lock=ctx.Lock(),
        **kwargs,
    )
