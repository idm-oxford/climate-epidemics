"""
Script to start a Dask distributed cluster to be used in the application.

As per Panel recommendation (https://panel.holoviz.org/how_to/concurrency/dask.html),
this script should be used to start the cluster separately from the application.
"""

from dask.distributed import LocalCluster

DASK_SCHEDULER_PORT = 64719
DASK_SCHEDULER_ADDRESS = f"tcp://127.0.0.1:{DASK_SCHEDULER_PORT}"

if __name__ == "__main__":
    cluster = LocalCluster(
        scheduler_port=DASK_SCHEDULER_PORT,
    )
    print(
        f"Dask cluster started. Check the Dask dashboard at {cluster.dashboard_link}."
    )
    input()
