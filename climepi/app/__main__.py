"""Entry point for the application. Run with ``python -m climepi.app``."""

import argparse

from climepi.app import run_app

parser = argparse.ArgumentParser(description="Run the climepi app locally.")
parser.add_argument(
    "--dask-distributed",
    action="store_true",
    default=False,
    help="Whether to use the Dask distributed scheduler. Default is False. If True "
    "(recommended), a Dask local cluster should first be started (from a separate "
    "terminal) by running ``python -m climepi.app.cluster``.",
)

args = parser.parse_args()
dask_distributed = args.dask_distributed
run_app(dask_distributed=dask_distributed)
