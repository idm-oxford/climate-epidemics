"""
Script to start a Dask distributed cluster to be used in the application.

As per Panel recommendation (https://panel.holoviz.org/how_to/concurrency/dask.html),
this script should be used to start the cluster separately from the application
(in a separate terminal, run `python -m climepi.app.cluster`).
"""

from dask.distributed import LocalCluster

from climepi.app import DASK_SCHEDULER_PORT

if __name__ == "__main__":
    cluster = LocalCluster(scheduler_port=DASK_SCHEDULER_PORT)
    print(
        f"Dask cluster started. Check the Dask dashboard at {cluster.dashboard_link}."
    )
    input()
