import multiprocessing

from dask.distributed import LocalCluster, Client


class DaskClientManager:
    def __init__(self):
        self.client = None

    def get_client(self, address=""):
        """
        Get the client to a local cluster or a cluster. Implicitly sets the client when not yet initialized. Defaults to
        local cluster, but will connect when given an address.

        :param address: The address of the cluster
        :type address: str
        :return: The client
        """
        if self.client is None and address == "":
            self.__create_local_cluster__()
        elif self.client is None and address != "":
            self.__connect_to_cluster__(address)
        return self.client

    def set_client(self, address=""):
        """
        Set the client to a local cluster or a cluster. Explicit function.

        :param address: The address of the cluster
        :type address: str
        """
        if self.client is None and address == "":
            self.__create_local_cluster__()
        elif self.client is None and address != "":
            self.__connect_to_cluster__(address)

    def __create_local_cluster__(self):
        """
        Create a local cluster with the number of cores - 1
        """
        if self.client is not None:
            return
        cores = multiprocessing.cpu_count() - 1
        cluster = LocalCluster(n_workers=cores, threads_per_worker=1)
        self.client = cluster.get_client()

    def __connect_to_cluster__(self, address):
        """
        Connect to a cluster

        :param address: The address of the cluster
        :type address: str
        """
        if self.client is not None:
            return
        self.client = Client(address)


# Create a global Dask client manager. Can be interpreted as a singleton.
dask_client_manager = DaskClientManager()
