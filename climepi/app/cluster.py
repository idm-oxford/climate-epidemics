"""
Script to start a Dask local cluster to be used in the application.

As per Panel recommendation (https://panel.holoviz.org/how_to/concurrency/dask.html),
this script should be used to start the cluster before starting the app.
"""

import time

from dask.distributed import LocalCluster

from climepi.app._app_construction import get_logger
from climepi.app._dask_port_address import DASK_SCHEDULER_PORT


def _start_cluster():
    logger = get_logger(name="cluster")
    cluster = LocalCluster(scheduler_port=DASK_SCHEDULER_PORT)
    logger.info(
        "Dask local cluster started. Check the Dask dashboard at %s",
        cluster.dashboard_link,
    )
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    _start_cluster()  # pragma: no cover
