import multiprocessing

from dask.distributed import LocalCluster, Client


class DaskClientManager:
    def __init__(self):
        self.client = None

    def get_client(self, address=""):
        if self.client is None and address == "":
            self.__create_local_cluster__()
        elif self.client is None and address != "":
            self.__connect_to_cluster__(address)
        return self.client

    def create_local_cluster(self):
    def __create_local_cluster__(self):
        """
        Create a local cluster with the number of cores - 1

        :return:
        """
        cores = multiprocessing.cpu_count() - 1
        cluster = LocalCluster(n_workers=cores, threads_per_worker=1)
        return cluster.get_client()

    def __connect_to_cluster__(self, address):
        """
        Connect to a cluster

        :param address: The address of the cluster
        :return: The client
        """
        return Client(address)


dask_client_manager = DaskClientManager()
