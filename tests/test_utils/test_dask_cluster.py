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
