"""Tests for ``liom_toolkit/utils/dask_client.py``.

Covers the DaskClientManager refactor: configurable ``n_workers`` (capped at
``min(cpu_count()-1, 8)``), ``close()`` lifecycle method, context-manager
support (``__enter__``/``__exit__``), and the ``Client(self.cluster)``
replacement for the fragile ``cluster.get_client()`` call.

Dask is mocked throughout (AGENTS.md §5 / MP-3): ``LocalCluster`` and
``Client`` are replaced with ``MagicMock`` per-test so no real Dask workers
are spawned (xdist + real ``LocalCluster`` deadlocks per the testing map).
``multiprocessing.cpu_count`` is monkeypatched for the cap-edge tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from liom_toolkit.utils.dask_client import DaskClientManager


def _install_dask_mocks(monkeypatch):
    """Patch ``LocalCluster`` and ``Client`` in the dask_client module with MagicMocks.

    Returns the two mocks so the test can inspect ``call_args``. Both are
    removed on test teardown by monkeypatch.
    """
    fake_cluster_cls = MagicMock(name="LocalCluster")
    fake_client_cls = MagicMock(name="Client")
    monkeypatch.setattr("liom_toolkit.utils.dask_client.LocalCluster", fake_cluster_cls)
    monkeypatch.setattr("liom_toolkit.utils.dask_client.Client", fake_client_cls)
    return fake_cluster_cls, fake_client_cls


def test_init_has_cluster_attr():
    """A fresh DaskClientManager has a ``cluster`` attribute initialized to None."""
    mgr = DaskClientManager()
    assert mgr.cluster is None
    assert mgr.client is None


def test_create_local_cluster_uses_client_not_get_client(monkeypatch):
    """``__create_local_cluster__`` builds the client via ``Client(self.cluster)``,
    not ``cluster.get_client()`` (MP-2 fix)."""
    fake_cluster_cls, fake_client_cls = _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()

    mgr.__create_local_cluster__()

    # The cluster attribute holds the LocalCluster instance.
    assert mgr.cluster is fake_cluster_cls.return_value
    # The client attribute holds the Client instance.
    assert mgr.client is fake_client_cls.return_value
    # Client was called with the cluster object as its sole argument —
    # NOT cluster.get_client().
    fake_client_cls.assert_called_once_with(mgr.cluster)


def test_n_workers_default_cap(monkeypatch):
    """``n_workers=None`` resolves to ``min(cpu_count()-1, 8)`` (integer arithmetic)."""
    fake_cluster_cls, _ = _install_dask_mocks(monkeypatch)

    for cpu, expected in [(16, 8), (9, 8), (4, 3)]:
        monkeypatch.setattr(
            "liom_toolkit.utils.dask_client.multiprocessing.cpu_count",
            lambda cpu=cpu: cpu,
        )
        mgr = DaskClientManager()
        mgr.__create_local_cluster__(None)
        kwargs = fake_cluster_cls.call_args.kwargs
        assert kwargs["n_workers"] == expected, f"cpu_count={cpu}"
        assert isinstance(kwargs["n_workers"], int), "n_workers must be int, not float"
        assert kwargs["dashboard_address"] is False, "dashboard must be disabled (teardown noise)"
        fake_cluster_cls.reset_mock()


def test_n_workers_explicit(monkeypatch):
    """An explicit ``n_workers`` is passed through to ``LocalCluster``."""
    fake_cluster_cls, _ = _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()

    mgr.__create_local_cluster__(2)

    assert fake_cluster_cls.call_args.kwargs["n_workers"] == 2
    assert fake_cluster_cls.call_args.kwargs["dashboard_address"] is False


def test_n_workers_zero_raises(monkeypatch):
    """``n_workers < 1`` raises ``ValueError`` with a clear message (never spawns 0 workers)."""
    _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()

    for bad in (0, -1):
        with pytest.raises(ValueError, match="n_workers must be >= 1"):
            mgr.__create_local_cluster__(bad)


def test_n_workers_cpu_count_one_raises(monkeypatch):
    """``cpu_count()=1`` → computed ``n_workers=0`` → ``ValueError`` (never silently 0 workers)."""
    _install_dask_mocks(monkeypatch)
    monkeypatch.setattr("liom_toolkit.utils.dask_client.multiprocessing.cpu_count", lambda: 1)
    mgr = DaskClientManager()

    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        mgr.__create_local_cluster__(None)


def test_close_releases_client_and_cluster(monkeypatch):
    """``close()`` calls ``client.close()`` then ``cluster.close()``, then nulls both."""
    _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()
    mgr.__create_local_cluster__()
    client_mock = mgr.client
    cluster_mock = mgr.cluster

    mgr.close()

    client_mock.close.assert_called_once()
    cluster_mock.close.assert_called_once()
    assert mgr.client is None
    assert mgr.cluster is None


def test_close_idempotent(monkeypatch):
    """``close()`` called twice is a no-op on the second call (no raise)."""
    _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()
    mgr.__create_local_cluster__()

    mgr.close()
    # Second call finds client=None, cluster=None — must not raise.
    mgr.close()
    assert mgr.client is None
    assert mgr.cluster is None


def test_close_without_create():
    """``close()`` on a fresh manager (never created a cluster) is a no-op, no raise."""
    mgr = DaskClientManager()
    mgr.close()
    assert mgr.client is None
    assert mgr.cluster is None


def test_close_client_raises_still_closes_cluster(monkeypatch):
    """If ``client.close()`` raises, ``cluster.close()`` is still called and
    both fields are cleared. Without the try/finally, a client.close() error
    would leak the cluster and leave the manager in a dirty state.

    The RuntimeError from ``client.close()`` propagates to the caller (the
    try/finally ensures cluster cleanup runs, not that the error is swallowed).
    """
    fake_cluster_cls, fake_client_cls = _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()
    mgr.__create_local_cluster__()
    client_mock = mgr.client
    cluster_mock = mgr.cluster
    client_mock.close.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        mgr.close()

    client_mock.close.assert_called_once()
    cluster_mock.close.assert_called_once()
    assert mgr.client is None
    assert mgr.cluster is None


def test_context_manager(monkeypatch):
    """``with DaskClientManager() as mgr`` yields the manager and calls ``close()`` on exit."""
    _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()

    with mgr as m:
        assert m is mgr
        m.set_client("")
        assert m.client is not None

    assert mgr.client is None
    assert mgr.cluster is None


def test_context_manager_exception_calls_close(monkeypatch):
    """An exception inside the with-block still triggers ``close()`` (finally semantics)."""
    _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()
    mgr.set_client("")
    assert mgr.client is not None

    with pytest.raises(RuntimeError, match="boom"), mgr:
        raise RuntimeError("boom")

    assert mgr.client is None
    assert mgr.cluster is None


def test_set_client_threads_n_workers(monkeypatch):
    """``set_client("", n_workers=3)`` creates a local cluster with ``n_workers=3``."""
    fake_cluster_cls, _ = _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()

    mgr.set_client("", n_workers=3)

    assert fake_cluster_cls.call_args.kwargs["n_workers"] == 3


def test_get_client_threads_n_workers(monkeypatch):
    """``get_client("", n_workers=3)`` creates a local cluster and returns the client."""
    fake_cluster_cls, fake_client_cls = _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()

    client = mgr.get_client("", n_workers=3)

    assert fake_cluster_cls.call_args.kwargs["n_workers"] == 3
    assert client is fake_client_cls.return_value


def test_set_client_threads_per_worker_default_is_one(monkeypatch):
    """``set_client("", n_workers=3)`` defaults ``threads_per_worker`` to 1
    (preserves the pre-hardening behavior — D-04c)."""
    fake_cluster_cls, _ = _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()

    mgr.set_client("", n_workers=3)

    assert fake_cluster_cls.call_args.kwargs["threads_per_worker"] == 1


def test_set_client_threads_per_worker_plumbed(monkeypatch):
    """``set_client("", n_workers=3, threads_per_worker=2)`` threads the param
    through to ``LocalCluster`` (D-04c)."""
    fake_cluster_cls, _ = _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()

    mgr.set_client("", n_workers=3, threads_per_worker=2)

    assert fake_cluster_cls.call_args.kwargs["threads_per_worker"] == 2


def test_get_client_threads_per_worker_plumbed(monkeypatch):
    """``get_client("", n_workers=3, threads_per_worker=2)`` threads the param
    through to ``LocalCluster`` on the empty-address path (D-04c)."""
    fake_cluster_cls, _ = _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()

    mgr.get_client("", n_workers=3, threads_per_worker=2)

    assert fake_cluster_cls.call_args.kwargs["threads_per_worker"] == 2


def test_set_client_wait_for_workers_called(monkeypatch):
    """``set_client(address, min_workers=2, timeout=30)`` calls
    ``client.wait_for_workers(2, timeout=30)`` on the non-empty-address path
    (D-04a readiness check)."""
    _, fake_client_cls = _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()

    mgr.set_client("127.0.0.1:8786", min_workers=2, timeout=30)

    fake_client_cls.return_value.wait_for_workers.assert_called_once_with(2, timeout=30)


def test_set_client_min_workers_defaults_to_n_workers(monkeypatch):
    """When ``min_workers`` is unset but ``n_workers`` is supplied,
    ``min_workers`` defaults to ``n_workers`` so the existing ``--n-workers``
    CLI flag drives readiness with no new flag (D-04a)."""
    _, fake_client_cls = _install_dask_mocks(monkeypatch)
    mgr = DaskClientManager()

    mgr.set_client("127.0.0.1:8786", n_workers=4)

    fake_client_cls.return_value.wait_for_workers.assert_called_once_with(4, timeout=None)


def test_set_client_wait_for_workers_timeout_re_raises_runtimeerror(monkeypatch):
    """A ``wait_for_workers`` ``TimeoutError`` is re-raised as ``RuntimeError``
    naming the offending address (D-04a / AGENTS §2 — raise with the offending
    value, never silently fall back)."""
    _, fake_client_cls = _install_dask_mocks(monkeypatch)
    fake_client_cls.return_value.wait_for_workers.side_effect = TimeoutError("nope")
    mgr = DaskClientManager()

    with pytest.raises(RuntimeError, match="connect to Dask scheduler") as exc:
        mgr.set_client("127.0.0.1:8786", min_workers=1, timeout=5)

    assert "127.0.0.1:8786" in str(exc.value)
    # No half-connected client leaks after the failure (T-13-08).
    assert mgr.client is None


def test_set_client_client_oserror_re_raises_runtimeerror(monkeypatch):
    """When ``Client(address)`` itself raises ``OSError`` (e.g. unreachable
    address), it is re-raised as ``RuntimeError`` naming the offending address
    (D-04a — verified: ``Client('127.0.0.1:1')`` raises ``OSError`` directly)."""
    _, fake_client_cls = _install_dask_mocks(monkeypatch)
    fake_client_cls.side_effect = OSError("connection refused")
    mgr = DaskClientManager()

    with pytest.raises(RuntimeError, match="connect to Dask scheduler") as exc:
        mgr.set_client("127.0.0.1:1", min_workers=1, timeout=5)

    assert "127.0.0.1:1" in str(exc.value)
    assert mgr.client is None


def test_set_client_no_fallback_to_local_cluster(monkeypatch):
    """On a failed remote connect, NO ``LocalCluster`` is constructed — the
    no-silent-fallback boundary (D-04a / AGENTS §2 / T-13-06)."""
    fake_cluster_cls, fake_client_cls = _install_dask_mocks(monkeypatch)
    fake_client_cls.side_effect = OSError("connection refused")
    mgr = DaskClientManager()

    with pytest.raises(RuntimeError):
        mgr.set_client("127.0.0.1:1", min_workers=1, timeout=5)

    fake_cluster_cls.assert_not_called()
    assert mgr.cluster is None
    assert mgr.client is None
