"""Dask distributed client manager (singleton)."""

from __future__ import annotations

import multiprocessing

from dask.distributed import Client, LocalCluster


class DaskClientManager:
    """Singleton manager for a Dask distributed client (local cluster or remote)."""

    def __init__(self) -> None:
        self.client: Client | None = None

    def get_client(self, address: str = "") -> Client:
        """Get the client to a local cluster or a cluster.

        Implicitly sets the client when not yet initialized. Defaults to a
        local cluster, but will connect when given an address.

        Parameters
        ----------
        address : str
            The address of the cluster.

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
            self.__create_local_cluster__()
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
            ``cpu_count() - 1`` when ``None`` and a local cluster is created.
        """
        if self.client is None and address == "":
            self.__create_local_cluster__(n_workers)
        elif self.client is None and address != "":
            self.__connect_to_cluster__(address)

    def __create_local_cluster__(self, n_workers: int | None = None) -> None:
        """Create a local cluster with the number of cores - 1.

        Parameters
        ----------
        n_workers : int, optional
            Number of workers. Defaults to ``cpu_count() - 1`` when ``None``.
        """
        if self.client is not None:
            return
        if n_workers is None:
            n_workers = multiprocessing.cpu_count() - 1
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        self.client = cluster.get_client()

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


# Create a global Dask client manager. Can be interpreted as a singleton.
dask_client_manager = DaskClientManager()
