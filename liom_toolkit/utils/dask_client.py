"""Dask distributed client manager (singleton)."""

from __future__ import annotations

import contextlib
import multiprocessing

from dask.distributed import Client, LocalCluster


class DaskClientManager:
    """Singleton manager for a Dask distributed client (local cluster or remote).

    Supports use as a context manager so the client and cluster are reliably
    released on exit::

        with DaskClientManager() as mgr:
            mgr.set_client(n_workers=4)
            ...

    The ``n_workers`` parameter (on ``set_client`` / ``get_client``) defaults
    to ``min(cpu_count() - 1, 8)`` when ``None`` — the cap prevents unbounded
    worker spawning on high-core machines. An explicit ``n_workers < 1``
    raises ``ValueError`` rather than silently spawning zero workers.
    """

    def __init__(self) -> None:
        self.client: Client | None = None
        self.cluster: LocalCluster | None = None

    def get_client(
        self,
        address: str = "",
        n_workers: int | None = None,
        threads_per_worker: int = 1,
    ) -> Client:
        """Get the client to a local cluster or a cluster.

        Implicitly sets the client when not yet initialized. Defaults to a
        local cluster, but will connect when given an address.

        Parameters
        ----------
        address : str
            The address of the cluster.
        n_workers : int, optional
            Number of workers for the local cluster. Ignored when connecting
            to a remote scheduler (``address`` non-empty). Defaults to
            ``min(cpu_count() - 1, 8)`` when ``None``. When ``min_workers`` is
            unset, this also drives the readiness check on the remote path.
        threads_per_worker : int, default 1
            Threads per worker for the local cluster. Ignored on the remote
            path (the remote scheduler controls its own thread count). Default
            ``1`` preserves the pre-hardening behavior.

        Returns
        -------
        Client
            The dask distributed client.

        Raises
        ------
        RuntimeError
            If the client could not be initialized after the setup attempt,
            or if connecting to a remote scheduler times out / is unreachable
            (the offending address is included in the message — never falls
            back to a ``LocalCluster``).
        """
        if self.client is None and address == "":
            self.__create_local_cluster__(n_workers, threads_per_worker)
        elif self.client is None and address != "":
            self.__connect_to_cluster__(address, min_workers=n_workers)
        if self.client is None:
            raise RuntimeError("Dask client was not initialized")
        return self.client

    def set_client(
        self,
        address: str = "",
        n_workers: int | None = None,
        min_workers: int | None = None,
        timeout: float | None = None,
        threads_per_worker: int = 1,
    ) -> None:
        """Set the client to a local cluster or a cluster. Explicit function.

        Parameters
        ----------
        address : str
            The address of the cluster. Empty string constructs a
            ``LocalCluster``; a non-empty string connects to a remote
            scheduler and **never** falls back to a ``LocalCluster`` on
            failure (AGENTS §2 — raise with the offending address instead).
        n_workers : int, optional
            Number of workers for the local cluster. Ignored when connecting
            to a remote scheduler (``address`` non-empty). Defaults to
            ``min(cpu_count() - 1, 8)`` when ``None``.
        min_workers : int, optional
            Readiness threshold for the remote path: ``wait_for_workers`` is
            called with this many workers. When ``None`` and ``n_workers`` is
            supplied, defaults to ``n_workers`` so the existing ``--n-workers``
            CLI flag drives readiness with no new flag. Ignored on the
            empty-address (local) path.
        timeout : float, optional
            Timeout in seconds for the remote ``wait_for_workers`` readiness
            check. ``None`` means dask's default. Ignored on the local path.
        threads_per_worker : int, default 1
            Threads per worker for the local cluster. Ignored on the remote
            path. Default ``1`` preserves the pre-hardening behavior.

        Notes
        -----
        On the local path, ``ValueError`` propagates from
        ``__create_local_cluster__`` if ``n_workers`` resolves to less than 1.
        On the remote path, ``RuntimeError`` propagates from
        ``__connect_to_cluster__`` if the scheduler is unreachable or does not
        reach ``min_workers`` within ``timeout`` — the offending address is
        included in the message and no ``LocalCluster`` fallback is constructed
        (AGENTS §2).
        """
        if self.client is None and address == "":
            self.__create_local_cluster__(n_workers, threads_per_worker)
        elif self.client is None and address != "":
            if min_workers is None:
                min_workers = n_workers
            self.__connect_to_cluster__(address, min_workers=min_workers, timeout=timeout)

    def __create_local_cluster__(
        self,
        n_workers: int | None = None,
        threads_per_worker: int = 1,
    ) -> None:
        """Create a local cluster.

        Parameters
        ----------
        n_workers : int, optional
            Number of workers. Defaults to ``min(cpu_count() - 1, 8)`` when
            ``None`` (the cap prevents OOM on high-core machines). An explicit
            value less than 1 raises ``ValueError``.
        threads_per_worker : int, default 1
            Threads per worker. Default ``1`` preserves the pre-hardening
            behavior.

        Raises
        ------
        ValueError
            If ``n_workers`` resolves to a value less than 1 (covers the
            ``cpu_count() == 1`` edge where ``cpu_count() - 1 == 0``).
        """
        if self.client is not None:
            return
        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count() - 1, 8)
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        # dashboard_address=False disables the bokeh dashboard. The dashboard's
        # TornadoServerApplication.stop() raises "Cannot synchronously wait on a
        # running event loop" during synchronous teardown (bokeh 3.x +
        # distributed 2026.x), which propagates as a RuntimeError out of
        # Client.close(). The dashboard is a development/debugging web UI and is
        # not needed for the library's use case.
        self.cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            dashboard_address=False,
        )
        self.client = Client(self.cluster)

    def __connect_to_cluster__(
        self,
        address: str,
        *,
        min_workers: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """Connect to a remote cluster and wait for worker readiness.

        Parameters
        ----------
        address : str
            The address of the cluster scheduler.
        min_workers : int, optional
            When not ``None``, call ``client.wait_for_workers(min_workers,
            timeout=timeout)`` after connecting to block until at least this
            many workers are up. ``None`` skips the readiness check (preserves
            the pre-hardening behavior for callers that don't pass it).
        timeout : float, optional
            Timeout in seconds for the readiness check. ``None`` means dask's
            default.

        Raises
        ------
        RuntimeError
            If ``Client(address)`` raises ``OSError`` (unreachable address) or
            ``wait_for_workers`` raises ``TimeoutError`` / ``OSError`` (not
            enough workers came up in time). The offending address and the
            actual worker count are included in the message. The half-connected
            client is closed and cleared so no leaked state persists.

        Notes
        -----
        This method **never** falls back to a ``LocalCluster`` on failure
        (AGENTS §2 — a wrong cluster returning plausible-looking results is
        the dominant failure mode for this pipeline). A non-empty ``address``
        is a hard contract: either it connects and reaches ``min_workers``, or
        it raises.
        """
        if self.client is not None:
            return
        try:
            # Pass `timeout` to Client(address) too, not just wait_for_workers.
            # Without it, an unreachable scheduler blocks for dask's default
            # 30s connect timeout regardless of the caller's `timeout` — the
            # `timeout` arg is meant to bound the whole connect+ready attempt,
            # and a 30s wait when the user asked for 5s is a latent UX bug.
            # `Client(timeout=None)` raises ValueError, so pass it only when set.
            if timeout is not None:
                self.client = Client(address, timeout=timeout)
            else:
                self.client = Client(address)
            if min_workers is not None:
                self.client.wait_for_workers(min_workers, timeout=timeout)
        except (TimeoutError, OSError) as e:
            # Defensive worker-count for the message; do NOT swallow if this
            # also raises — just format around it. ncores() on a broken client
            # realistically raises OSError (transport) or RuntimeError (state).
            actual: str
            if self.client is None:
                actual = "0"
            else:
                try:
                    actual = str(len(self.client.ncores()))
                except (OSError, RuntimeError):
                    actual = "unknown"
            # Clean up any half-connected client so no leaked state persists.
            # close() failures on a broken client are typically OSError /
            # RuntimeError — suppress those specifically (AGENTS §2: no bare
            # except / except Exception). The original RuntimeError below is
            # the one that must surface; the cleanup must not mask it.
            if self.client is not None:
                with contextlib.suppress(OSError, RuntimeError):
                    self.client.close()
                self.client = None
            raise RuntimeError(
                f"connect to Dask scheduler {address!r} failed: {e}; {actual} workers came up"
            ) from e

    def close(self) -> None:
        """Close the client and cluster, releasing worker processes.

        Idempotent: safe to call when no client/cluster was ever created, and
        safe to call more than once. The client is closed before the cluster
        so in-flight tasks are drained before the scheduler shuts down. If
        ``client.close()`` raises, the cluster is still closed (via finally)
        and both fields are cleared, so the manager is left in a clean state
        regardless of teardown errors.
        """
        if self.client is None and self.cluster is None:
            return
        try:
            if self.client is not None:
                self.client.close()
        finally:
            if self.cluster is not None:
                self.cluster.close()
            self.client = None
            self.cluster = None

    def __enter__(self) -> DaskClientManager:
        """Enter the context manager.

        Returns
        -------
        DaskClientManager
            ``self``, so the caller can access the manager inside the block.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Exit the context manager, closing the client and cluster.

        Called on both normal exit and exception. Does not suppress
        exceptions — ``*exc_info`` is captured only so the signature matches
        the context-manager protocol.
        """
        self.close()


# Create a global Dask client manager. Can be interpreted as a singleton.
dask_client_manager = DaskClientManager()
