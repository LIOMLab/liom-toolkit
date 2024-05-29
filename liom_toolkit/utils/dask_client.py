from dask.distributed import LocalCluster

# Setup default client for Dask. This is a global variable that can be used in any module.
# This client is a local cluster that will use all threads available.
cluster = LocalCluster()
client = cluster.get_client()
