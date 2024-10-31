"""Entry point for the application. Run with ``python -m climepi.app``."""

import argparse

from climepi.app._app_construction import run_app

parser = argparse.ArgumentParser(description="Run the climepi app locally.")
parser.add_argument(
    "--dask-distributed",
    action="store_true",
    default=False,
    help="Use the Dask distributed scheduler (recommended if using multiple app "
    "instances simultaneously). If not specified, the Dask thread-based scheduler "
    "will be used.",
)

args = parser.parse_args()
run_app(dask_distributed=args.dask_distributed)
