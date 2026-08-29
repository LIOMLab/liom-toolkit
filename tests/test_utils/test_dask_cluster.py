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
"""

from __future__ import annotations

import sys

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
    with pytest.raises(RuntimeError, match="connect to Dask scheduler"):
        mgr.set_client("127.0.0.1:1", min_workers=1, timeout=5)
    # No silent LocalCluster fallback (T-13-06 / D-04a boundary).
    assert mgr.cluster is None
    assert mgr.client is None
