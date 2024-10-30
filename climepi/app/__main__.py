"""Entry point for the application. Run with ``python -m climepi.app``."""

import argparse

import panel as pn
from dask.distributed import Client

from climepi.app import DASK_SCHEDULER_ADDRESS, get_app


def run_app(
    dask_distributed=False,
    clim_dataset_example_base_dir=None,
    clim_dataset_example_names=None,
    epi_model_example_names=None,
    enable_custom_epi_model=True,
):
    """
    Run the climepi `Panel` app locally in a browser.

    Parameters
    ----------
    dask_distributed: bool
        Whether to use the Dask distributed scheduler. Default is False. If True
        (recommended), a Dask local cluster should first be started (from a separate
        terminal) by running ``python -m climepi.app.cluster``.
    clim_dataset_example_base_dir: str or pathlib.Path
        Base directory for the example climate datasets, optional. If None, the datasets
        will be downloaded to and accessed from the OS cache.
    clim_dataset_example_names: list of str
        List of example names for climate datasets, optional. If None, the default list
        in climdata.EXAMPLE_NAMES is used.
    epi_model_example_names: list of str
        List of example names for epidemiological models, optional. If None, the default
        list in epimod.EXAMPLE_NAMES is used.
    clim_dataset_example_base_dir: str or pathlib.Path
        Base directory for the example climate datasets, optional. If None, the datasets
        will be downloaded to and accessed from the OS cache.
    clim_dataset_example_names: list of str
        List of example names for climate datasets, optional. If None, the default list
        in climdata.EXAMPLE_NAMES is used.
    epi_model_example_names: list of str
        List of example names for epidemiological models, optional. If None, the default
        list in epimod.EXAMPLE_NAMES is used.
    enable_custom_epi_model: bool
        Whether to enable the option to specify a custom temperature range in which
        transmission can occur. Default is True.

    Returns
    -------
    None
    """
    if dask_distributed:
        try:
            client = Client(DASK_SCHEDULER_ADDRESS)
            print(f"Client connected to Dask local cluster ({client.dashboard_link}).")
        except OSError as e:
            raise OSError(
                "Could not connect to Dask local cluster. Start the cluster by running "
                "`python -m climepi.app.cluster` in a separate terminal."
            ) from e
    else:
        print("Running using Dask thread-based scheduler.")

    def _get_app():
        return get_app(
            clim_dataset_example_base_dir=clim_dataset_example_base_dir,
            clim_dataset_example_names=clim_dataset_example_names,
            epi_model_example_names=epi_model_example_names,
            enable_custom_epi_model=enable_custom_epi_model,
        )

    pn.serve({"/climepi_app": _get_app})


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
run_app(dask_distributed=args.dask_distributed)
