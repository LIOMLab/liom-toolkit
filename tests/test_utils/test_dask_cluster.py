"""Tests for ``liom_toolkit/utils/dask_cluster.py``.

Covers the ``create_slurm_cluster`` factory's lazy-import guard (D-04 /
D-04b) and the SC-5 ``LocalCluster(processes=True)`` proxy that verifies the
hardened ``set_client(address, min_workers=, timeout=)`` path end-to-end
(D-04d).

The unit tests (ImportError guard, module-imports-without-extra) use
``monkeypatch.setitem(sys.modules, ...)`` to force the absence of
``dask_jobqueue`` — no real SLURM cluster is spawned. The two integration
tests (``@pytest.mark.slow``) spin a real ``LocalCluster(processes=True)``
as the SC-5 proxy for a remote scheduler; dask is core so no
``importorskip`` is needed (AGENTS.md §5).

The ``create_slurm_cluster`` happy-path tests mock ``SLURMCluster`` and
``dask.distributed.Client`` (no real SLURM scheduler) to verify the
factory's kwargs None-filtering, ``scale(jobs=N)`` idiom,
``wait_for_workers`` readiness call, the ``(cluster, client)`` return, and
that ``TimeoutError`` propagates unwrapped (no silent fallback — the caller
decides retry/abort, AGENTS §2).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


def test_create_slurm_cluster_missing_extra(monkeypatch):
    """``create_slurm_cluster`` raises ``ImportError`` with an install hint
    naming the ``[dask-cluster]`` extra when ``dask_jobqueue`` is absent
    (D-04 / AGENTS §8 — degrade gracefully with a user-facing message)."""
    # Force ImportError on `import dask_jobqueue` inside the function.
    monkeypatch.setitem(sys.modules, "dask_jobqueue", None)
    from liom_toolkit.utils.dask_cluster import create_slurm_cluster

    with pytest.raises(ImportError, match=r"install liom-toolkit\[dask-cluster\]"):
        create_slurm_cluster(
            queue="gpu",
            cores=1,
            memory="1GB",
            walltime="00:10:00",
            n_workers=1,
        )


def test_dask_cluster_module_imports_without_extra(monkeypatch):
    """``liom_toolkit.utils.dask_cluster`` imports cleanly without
    ``dask_jobqueue`` installed — the heavy import is function-scope lazy,
    not module-top (AGENTS §8/§9)."""
    monkeypatch.setitem(sys.modules, "dask_jobqueue", None)
    # Drop any cached import so the module re-executes under the patched
    # sys.modules.
    sys.modules.pop("liom_toolkit.utils.dask_cluster", None)
    import liom_toolkit.utils.dask_cluster as mod

    assert hasattr(mod, "create_slurm_cluster")


@pytest.mark.slow
def test_set_client_address_wait_for_workers():
    """SC-5 proxy: the hardened ``set_client(address, min_workers=,
    timeout=)`` path connects to a ``LocalCluster(processes=True)`` and
    waits for 2 workers — verifies the real ``wait_for_workers`` readiness
    path end-to-end (D-04d).

    A ``LocalCluster(processes=True)`` is the closest in-process proxy for a
    remote SLURM scheduler: it spawns real worker subprocesses that
    ``wait_for_workers`` must actually wait for, exercising the same
    ``Client(address)`` → ``wait_for_workers`` code path the SLURM factory
    uses. ``dashboard_address=False`` avoids the bokeh teardown noise.
    """
    from dask.distributed import LocalCluster

    from liom_toolkit.utils.dask_client import DaskClientManager

    cluster = LocalCluster(processes=True, n_workers=2, dashboard_address=False)
    try:
        mgr = DaskClientManager()
        mgr.set_client(cluster.scheduler_address, min_workers=2, timeout=30)
        # 2 workers came up — the readiness check passed and the client sees
        # both of them.
        assert len(mgr.client.ncores()) == 2
        mgr.close()
    finally:
        cluster.close()


@pytest.mark.slow
def test_set_client_bad_address_no_fallback():
    """``set_client`` with a bad address raises ``RuntimeError`` and does NOT
    fall back to a ``LocalCluster`` (D-04a / AGENTS §2 — a wrong cluster
    returning plausible-looking results is the dominant failure mode for
    this pipeline).

    ``127.0.0.1:1`` is reserved/unroutable, so ``Client('127.0.0.1:1')``
    raises ``OSError`` directly; the hardened path re-raises it as
    ``RuntimeError`` naming the offending address. The boundary check
    (``mgr.cluster is None``) proves no silent ``LocalCluster`` was
    constructed.
    """
    from liom_toolkit.utils.dask_client import DaskClientManager

    mgr = DaskClientManager()
    # timeout=2 bounds the connect attempt (passed through to Client(address)
    # as well as wait_for_workers) so the test fails fast instead of waiting
    # dask's default 30s connect timeout.
    with pytest.raises(RuntimeError, match="connect to Dask scheduler"):
        mgr.set_client("127.0.0.1:1", min_workers=1, timeout=2)
    # No silent LocalCluster fallback (T-13-06 / D-04a boundary).
    assert mgr.cluster is None
    assert mgr.client is None


# --- create_slurm_cluster happy-path (mocked SLURMCluster + Client) ---


def _install_fake_slurm(monkeypatch, *, wait_side_effect=None):
    """Inject a fake ``dask_jobqueue.SLURMCluster`` and fake ``Client``.

    Returns ``(fake_cluster_cls, fake_client_cls)`` so the test can inspect
    construction kwargs, ``scale`` calls, and ``wait_for_workers`` calls. The
    fake ``SLURMCluster`` records the kwargs it was built with; the fake
    ``Client`` records the cluster it wrapped and the ``wait_for_workers``
    args. No real SLURM scheduler or dask worker process is spawned — this
    exercises the factory's *plumbing* (kwargs filtering, scale idiom,
    readiness call, return tuple), not dask-jobqueue's sbatch submission.
    """
    import types

    captured: dict = {}

    class _FakeCluster:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.scale = MagicMock(name="scale")

        def close(self):
            pass

    class _FakeClient:
        def __init__(self, cluster):
            captured["wrapped_cluster"] = cluster
            self.wait_for_workers = MagicMock(name="wait_for_workers")
            if wait_side_effect is not None:
                self.wait_for_workers.side_effect = wait_side_effect

        def close(self):
            pass

    fake_mod = types.ModuleType("dask_jobqueue")
    fake_mod.SLURMCluster = _FakeCluster
    monkeypatch.setitem(sys.modules, "dask_jobqueue", fake_mod)
    # Client is imported as `from dask.distributed import Client` inside the
    # factory — patch the attribute on the real dask.distributed module so
    # the function-scope import binds to the fake.
    import dask.distributed as dd

    monkeypatch.setattr(dd, "Client", _FakeClient)
    return _FakeCluster, _FakeClient, captured


def test_create_slurm_cluster_happy_path(monkeypatch):
    """``create_slurm_cluster`` builds SLURMCluster with None kwargs omitted,
    scales via ``scale(jobs=N)`` (the SLURM idiom, NOT ``n_workers=N``), wraps
    a Client, blocks on ``wait_for_workers(N, timeout=...)``, and returns
    ``(cluster, client)`` which the caller owns (D-04 / D-04c).

    With ``account`` and ``processes`` left ``None`` they are OMITTED from the
    SLURMCluster kwargs (dask-jobqueue rejects ``None`` for some kwargs). The
    factory returns the cluster so the caller can ``close()`` it on failure.
    """
    _, _, captured = _install_fake_slurm(monkeypatch, wait_side_effect=None)
    from liom_toolkit.utils.dask_cluster import create_slurm_cluster

    cluster, client = create_slurm_cluster(
        queue="gpu",
        cores=8,
        memory="16GB",
        walltime="01:00:00",
        n_workers=4,
        timeout=30,
    )

    # None-valued optional kwargs are omitted (dask-jobqueue rejects None).
    kwargs = captured["kwargs"]
    assert "queue" in kwargs and kwargs["queue"] == "gpu"
    assert kwargs["cores"] == 8
    assert kwargs["memory"] == "16GB"
    assert kwargs["walltime"] == "01:00:00"
    assert "account" not in kwargs, "None account must be omitted"
    assert "processes" not in kwargs, "None processes must be omitted"
    # job_extra_directives is normalized to a list.
    assert kwargs["job_extra_directives"] == []

    # SLURM idiom: scale(jobs=N), NOT scale(n_workers=N).
    cluster.scale.assert_called_once_with(jobs=4)

    # Client wraps the cluster; wait_for_workers called with (N, timeout=).
    assert captured["wrapped_cluster"] is cluster
    client.wait_for_workers.assert_called_once_with(4, timeout=30)

    # Factory returns (cluster, client) — caller owns cluster.close().
    assert cluster is not None
    assert client is not None


def test_create_slurm_cluster_includes_account_processes(monkeypatch):
    """When ``account`` and ``processes`` are supplied they ARE passed to
    SLURMCluster (not omitted) — the None-omission is conditional, not
    unconditional (D-04c)."""
    _, _, captured = _install_fake_slurm(monkeypatch)
    from liom_toolkit.utils.dask_cluster import create_slurm_cluster

    create_slurm_cluster(
        queue="gpu",
        cores=8,
        memory="16GB",
        walltime="01:00:00",
        n_workers=2,
        account="def-user",
        processes=2,
        job_extra_directives=("--exclusive",),
    )

    kwargs = captured["kwargs"]
    assert kwargs["account"] == "def-user"
    assert kwargs["processes"] == 2
    # tuple job_extra_directives normalized to list for dask-jobqueue.
    assert kwargs["job_extra_directives"] == ["--exclusive"]


def test_create_slurm_cluster_timeout_propagates_unwrapped(monkeypatch):
    """A ``wait_for_workers`` ``TimeoutError`` propagates UNWRAPPED — the
    factory does not convert it to ``RuntimeError`` or swallow it (AGENTS §2:
    no silent fallback; the caller decides retry/abort). Contrast with
    ``DaskClientManager.__connect_to_cluster__`` which wraps for its
    singleton-UX contract; the factory is a lower-level primitive."""
    _, _, _ = _install_fake_slurm(
        monkeypatch, wait_side_effect=TimeoutError("workers did not come up")
    )
    from liom_toolkit.utils.dask_cluster import create_slurm_cluster

    with pytest.raises(TimeoutError, match="workers did not come up"):
        create_slurm_cluster(
            queue="gpu",
            cores=1,
            memory="1GB",
            walltime="00:10:00",
            n_workers=1,
            timeout=5,
        )
