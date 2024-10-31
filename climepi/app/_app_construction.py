"""Module defining the layout of the climepi app and providing a method to run it."""

import atexit
import logging
import signal
import sys

import panel as pn
from dask.distributed import Client, LocalCluster

from climepi.app._app_classes_methods import Controller

DASK_SCHEDULER_PORT = 64719
DASK_SCHEDULER_ADDRESS = f"tcp://127.0.0.1:{DASK_SCHEDULER_PORT}"


@pn.cache()
def _get_logger(name):
    # Set up logger (see https://panel.holoviz.org/how_to/logging/index.html)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setStream(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def _setup(dask_distributed=None):
    # Setup actions

    logger = _get_logger(name="setup")
    logger.info("Setting up the app. Press Ctrl+C to stop the `Panel` server")

    # Set up the Dask cluster if needed

    if dask_distributed:
        cluster = LocalCluster(scheduler_port=DASK_SCHEDULER_PORT)
        logger.info(
            "Dask local cluster started with %s workers. "
            "Check the Dask dashboard at %s",
            len(cluster.workers),
            cluster.dashboard_link,
        )
        return cluster
    logger.info("Using the Dask thread-based scheduler")

    pn.state.cache["controllers"] = {}
    return None


def _session_created(dask_distributed=None):
    # Set up the session (connect to the Dask cluster if needed)

    logger = _get_logger(name="session")
    logger.info("Session created")
    if dask_distributed:
        try:
            client = Client(DASK_SCHEDULER_ADDRESS)
            logger.info(
                "Client connected to Dask local cluster (%s)", client.dashboard_link
            )
        except OSError as e:
            raise OSError("Client could not connect to Dask local cluster") from e


def _layout(
    clim_dataset_example_base_dir=None,
    clim_dataset_example_names=None,
    epi_model_example_names=None,
    enable_custom_epi_model=None,
):
    # Define the layout of the app

    template = pn.template.BootstrapTemplate(title="climepi app")

    controller = Controller(
        clim_dataset_example_base_dir=clim_dataset_example_base_dir,
        clim_dataset_example_names=clim_dataset_example_names,
        epi_model_example_names=epi_model_example_names,
        enable_custom_epi_model=enable_custom_epi_model,
    )

    data_controls = controller.data_controls
    template.sidebar.append(data_controls)

    clim_plot_controls = controller.clim_plot_controls
    epi_plot_controls = controller.epi_plot_controls
    clim_plot_view = controller.clim_plot_view
    epi_model_plot_view = controller.epi_model_plot_view
    epi_plot_view = controller.epi_plot_view

    template.main.append(
        pn.Tabs(
            ("Climate data", pn.Row(clim_plot_controls, clim_plot_view)),
            ("Epidemiological model", pn.Row(epi_model_plot_view)),
            ("Epidemiological projections", pn.Row(epi_plot_controls, epi_plot_view)),
        )
    )

    return template, controller


def _session_destroyed(controller):
    # Clean up the session

    controller.cleanup_temp_file()
    logger = _get_logger(name="session")
    logger.info("Session cleaned up successfully (deleted temporary file(s))")


def _session(
    clim_dataset_example_base_dir=None,
    clim_dataset_example_names=None,
    epi_model_example_names=None,
    enable_custom_epi_model=None,
):
    # Get the template and controller

    template, controller = _layout(
        clim_dataset_example_base_dir=clim_dataset_example_base_dir,
        clim_dataset_example_names=clim_dataset_example_names,
        epi_model_example_names=epi_model_example_names,
        enable_custom_epi_model=enable_custom_epi_model,
    )

    session_id = pn.state.curdoc.session_context.id
    pn.state.cache["controllers"][session_id] = controller

    # Ensure temp files are cleaned up both when a session is closed

    def _cleanup_session(session_context):
        _session_destroyed(controller)
        pn.state.cache["controllers"].pop(session_id)

    pn.state.on_session_destroyed(_cleanup_session)

    return template


def _stop(panel_server, dask_cluster):
    # Stop the app and the Dask cluster (if used)

    logger = _get_logger(name="stop")
    logger.info("Cleaning up sessions")
    for controller in pn.state.cache["controllers"].values():
        _session_destroyed(controller)
    panel_server.stop()
    logger.info("Panel server stopped")
    if dask_cluster:
        dask_cluster.close()
        logger.info("Dask local cluster stopped")
    sys.exit(0)


def run_app(
    dask_distributed=False,
    clim_dataset_example_base_dir=None,
    clim_dataset_example_names=None,
    epi_model_example_names=None,
    enable_custom_epi_model=True,
    **kwargs,
):
    """
    Run the climepi `Panel` app locally in a browser.

    Parameters
    ----------
    dask_distributed: bool
        Whether to use the Dask distributed scheduler (recommended if using multiple app
        instances simultaneously). If False, the Dask thread-based scheduler will be
        used. Default is False.
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
    **kwargs
        Additional keyword arguments to pass to `pn.serve`.

    Returns
    -------
    None
    """
    dask_cluster = _setup(dask_distributed=dask_distributed)

    def session_created(session_context):
        _session_created(dask_distributed=dask_distributed)

    pn.state.on_session_created(session_created)

    def session():
        return _session(
            clim_dataset_example_base_dir=clim_dataset_example_base_dir,
            clim_dataset_example_names=clim_dataset_example_names,
            epi_model_example_names=epi_model_example_names,
            enable_custom_epi_model=enable_custom_epi_model,
        )

    panel_server = pn.serve({"/climepi_app": session}, start=False, **kwargs)

    def stop(signum, frame):
        _stop(panel_server, dask_cluster)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    panel_server.io_loop.start()
