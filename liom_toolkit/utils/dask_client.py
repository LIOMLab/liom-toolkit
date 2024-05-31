import multiprocessing

from dask.distributed import LocalCluster, Client

global client


def get_client(address=""):
    if client is None and address == "":
        create_local_cluster()
    elif client is None and address != "":
        connect_to_cluster(address)
    return client


def create_local_cluster():
    """
    Create a local cluster with the number of cores - 1

    :return:
    """
    cores = multiprocessing.cpu_count() - 1
    cluster = LocalCluster(n_workers=cores, threads_per_worker=1)
    global client
    client = cluster.get_client()


def connect_to_cluster(address):
    """
    Connect to a cluster

    :param address: The address of the cluster
    :return: The client
    """
    global client
    client = Client(address)
