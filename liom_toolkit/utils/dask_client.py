import multiprocessing

from dask.distributed import LocalCluster, Client


class DaskClientManager:
    def __init__(self):
        self.client = None

    def get_client(self, address=""):
        if self.client is None and address == "":
            self.client = self.create_local_cluster()
        elif self.client is None and address != "":
            self.client = self.connect_to_cluster(address)
        return self.client

    def create_local_cluster(self):
        """
        Create a local cluster with the number of cores - 1

        :return:
        """
        cores = multiprocessing.cpu_count() - 1
        cluster = LocalCluster(n_workers=cores, threads_per_worker=1)
        return cluster.get_client()

    def connect_to_cluster(self, address):
        """
        Connect to a cluster

        :param address: The address of the cluster
        :return: The client
        """
        return Client(address)


dask_client_manager = DaskClientManager()
