"""Subpackage providing a browser-based front-end application."""

from climepi.app._app_construction import get_app  # noqa

DASK_SCHEDULER_PORT = 64719
DASK_SCHEDULER_ADDRESS = f"tcp://127.0.0.1:{DASK_SCHEDULER_PORT}"
