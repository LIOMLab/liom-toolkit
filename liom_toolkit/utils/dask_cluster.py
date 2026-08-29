"""SLURM cluster factory for ``dask-jobqueue`` (lazy-imported).

The ``create_slurm_cluster`` factory builds a ``dask_jobqueue.SLURMCluster``,
scales it to ``n_workers`` jobs, wraps a ``dask.distributed.Client`` around it,
and blocks on ``wait_for_workers`` for readiness. ``dask_jobqueue`` is an
optional extra (``liom-toolkit[dask-cluster]``) — it is imported
**function-scope lazy** so this module imports cleanly without the extra
installed (AGENTS §8/§9). Heavy types live under ``TYPE_CHECKING``.

``threads_per_worker`` is NOT a parameter here: ``SLURMCluster`` does not
accept it directly. The per-worker thread count is expressed via the
``cores``/``processes`` ratio (``threads_per_worker = cores // processes``) —
the dask-jobqueue-native model (D-04c). Callers control it by choosing
``cores`` and ``processes``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster


def create_slurm_cluster(
    queue: str,
    cores: int,
    memory: str,
    walltime: str,
    *,
    n_workers: int,
    account: str | None = None,
    processes: int | None = None,
    job_extra_directives: tuple[str, ...] = (),
    timeout: float | None = None,
) -> tuple[SLURMCluster, Client]:
    """Build a SLURM-backed Dask cluster and wait for ``n_workers`` to come up.

    Submits ``n_workers`` SLURM jobs via ``dask_jobqueue.SLURMCluster`` +
    ``cluster.scale(jobs=n_workers)``, wraps a ``Client`` around the cluster,
    and blocks on ``client.wait_for_workers(n_workers, timeout=timeout)`` for
    readiness. The caller owns the returned ``cluster`` and is responsible for
    ``cluster.close()`` (this is a factory, NOT a singleton — contrast with
    ``DaskClientManager``).

    Parameters
    ----------
    queue : str
        SLURM partition/queue name.
    cores : int
        Total cores per SLURM job. Together with ``processes`` this implies
        ``threads_per_worker = cores // processes`` (D-04c — ``SLURMCluster``
        does not accept ``threads_per_worker`` directly).
    memory : str
        Memory per SLURM job (e.g. ``"16GB"``).
    walltime : str
        SLURM walltime (e.g. ``"01:00:00"``).
    n_workers : int
        Number of SLURM jobs to submit and wait for.
    account : str, optional
        SLURM account/charge ID. ``None`` omits the kwarg (dask-jobqueue
        rejects ``None`` for some kwargs).
    processes : int, optional
        Number of Dask worker processes per SLURM job. ``None`` lets
        dask-jobqueue default it (typically 1).
    job_extra_directives : tuple of str, default ()
        Extra ``#SBATCH`` directives passed to the job script.
    timeout : float, optional
        Readiness timeout in seconds for ``wait_for_workers``. ``None`` means
        dask's default. A timeout / connection error propagates as
        ``TimeoutError`` / ``OSError`` — the caller decides whether to retry
        or abort (no silent fallback, AGENTS §2).

    Returns
    -------
    tuple[Any, Any]
        ``(cluster, client)`` — the ``SLURMCluster`` and the ``Client``
        connected to it. The caller owns ``cluster.close()``.

    Raises
    ------
    ImportError
        If ``dask_jobqueue`` is not installed (the ``[dask-cluster]`` extra).
        The message names the install command.

    Notes
    -----
    ``wait_for_workers`` / ``Client`` may raise ``TimeoutError`` or ``OSError``
    if the cluster does not reach ``n_workers`` within ``timeout`` or the
    scheduler is unreachable. These propagate unwrapped — the caller is in a
    better position to decide retry/abort semantics (no silent fallback,
    AGENTS §2).
    """
    try:
        from dask_jobqueue import SLURMCluster
    except ImportError as e:
        raise ImportError("install liom-toolkit[dask-cluster]") from e
    from dask.distributed import Client

    # Build kwargs, omitting None values — dask_jobqueue rejects None for some
    # kwargs (e.g. account, processes).
    kwargs: dict[str, object] = {
        "queue": queue,
        "cores": cores,
        "memory": memory,
        "walltime": walltime,
        "job_extra_directives": list(job_extra_directives),
    }
    if account is not None:
        kwargs["account"] = account
    if processes is not None:
        kwargs["processes"] = processes

    cluster = SLURMCluster(**kwargs)
    # SLURM idiom: scale(jobs=N) submits N SLURM jobs (NOT scale(n_workers=N),
    # which scales by worker count and can mis-subdivide across processes).
    cluster.scale(jobs=n_workers)
    client = Client(cluster)
    # Readiness — no silent fallback. Let TimeoutError/OSError propagate; the
    # caller decides retry/abort. On failure the cluster + client are closed
    # here before re-raising — otherwise the caller never receives a reference
    # to them (the function raises before returning the tuple) and the SLURM
    # jobs stay submitted while the dask objects leak (open sockets, worker
    # processes). The cleanup closes use contextlib.suppress with the specific
    # exception types close() raises on a broken cluster (AGENTS §2: no bare
    # except / except Exception).
    try:
        client.wait_for_workers(n_workers, timeout=timeout)
    except (TimeoutError, OSError):
        with contextlib.suppress(OSError, RuntimeError):
            client.close()
        with contextlib.suppress(OSError, RuntimeError):
            cluster.close()
        raise
    return cluster, client
