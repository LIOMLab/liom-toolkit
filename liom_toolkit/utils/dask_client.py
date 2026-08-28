"""Dask distributed client manager (singleton)."""

from __future__ import annotations

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

    def get_client(self, address: str = "", n_workers: int | None = None) -> Client:
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
            ``min(cpu_count() - 1, 8)`` when ``None``.

        Returns
        -------
        Client
            The dask distributed client.

        Raises
        ------
        RuntimeError
            If the client could not be initialized after the setup attempt.
        """
        if self.client is None and address == "":
            self.__create_local_cluster__(n_workers)
        elif self.client is None and address != "":
            self.__connect_to_cluster__(address)
        if self.client is None:
            raise RuntimeError("Dask client was not initialized")
        return self.client

    def set_client(self, address: str = "", n_workers: int | None = None) -> None:
        """Set the client to a local cluster or a cluster. Explicit function.

        Parameters
        ----------
        address : str
            The address of the cluster.
        n_workers : int, optional
            Number of workers for the local cluster. Ignored when connecting
            to a remote scheduler (``address`` non-empty). Defaults to
            ``min(cpu_count() - 1, 8)`` when ``None``.
        """
        if self.client is None and address == "":
            self.__create_local_cluster__(n_workers)
        elif self.client is None and address != "":
            self.__connect_to_cluster__(address)

    def __create_local_cluster__(self, n_workers: int | None = None) -> None:
        """Create a local cluster.

        Parameters
        ----------
        n_workers : int, optional
            Number of workers. Defaults to ``min(cpu_count() - 1, 8)`` when
            ``None`` (the cap prevents OOM on high-core machines). An explicit
            value less than 1 raises ``ValueError``.

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
            n_workers=n_workers, threads_per_worker=1, dashboard_address=False
        )
        self.client = Client(self.cluster)

    def __connect_to_cluster__(self, address: str) -> None:
        """Connect to a cluster.

        Parameters
        ----------
        address : str
            The address of the cluster.
        """
        if self.client is not None:
            return
        self.client = Client(address)

    def close(self) -> None:
        """Close the client and cluster, releasing worker processes.

        Idempotent: safe to call when no client/cluster was ever created, and
        safe to call more than once. The client is closed before the cluster
        so in-flight tasks are drained before the scheduler shuts down.
        """
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.cluster is not None:
            self.cluster.close()
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
