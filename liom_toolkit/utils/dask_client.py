import multiprocessing

from dask.distributed import LocalCluster

# Setup default client for Dask. This is a global variable that can be used in any module.
# This client is a local cluster that will use all threads available minus 1.
cores = multiprocessing.cpu_count() - 1
cluster = LocalCluster(n_workers=cores, threads_per_worker=1)
client = cluster.get_client()
