"""Module defining the layout of the climepi app and providing a method to run it."""

import logging
import signal
import sys

import panel as pn
from dask.distributed import Client

from climepi.app._app_classes_methods import Controller
from climepi.app._dask_port_address import DASK_SCHEDULER_ADDRESS


def run_app(
    clim_dataset_example_base_dir=None,
    clim_dataset_example_names=None,
    epi_model_example_names=None,
    enable_custom_epi_model=True,
    dask_distributed=False,
    **kwargs,
):
    """
    Run the climepi `Panel` app locally in a browser.

    Parameters
    ----------
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
    dask_distributed: bool
        Whether to use the Dask distributed scheduler. Default is False. If False, the
        Dask single-machine scheduler using threads will be used. To use the distributed
        scheduler, a Dask local cluster must be started from a separate terminal by
        running ``python -m climepi.app.cluster`` before starting the app.
    **kwargs
        Additional keyword arguments to pass to `pn.serve`.

    Returns
    -------
    None
    """
    logger = get_logger(name="setup")
    logger.info("Setting up the app")

    _setup_dask(dask_distributed=dask_distributed)

    def session_created(session_context):
        logger_session = get_logger(name="session")
        logger_session.info("Session created")

    pn.state.on_session_created(session_created)

    def session():
        return _session(
            clim_dataset_example_base_dir=clim_dataset_example_base_dir,
            clim_dataset_example_names=clim_dataset_example_names,
            epi_model_example_names=epi_model_example_names,
            enable_custom_epi_model=enable_custom_epi_model,
        )

    server = pn.serve({"/climepi_app": session}, start=False, **kwargs)

    _set_shutdown()

    logger.info("Set-up complete. Press Ctrl+C to stop the app")

    server.start()
    server.io_loop.start()


@pn.cache()
def get_logger(name):
    """
    Set up logger (see https://panel.holoviz.org/how_to/logging/index.html).

    Parameters
    ----------
    name: str
        Name of the logger.

    Returns
    -------
    logging.Logger
        Logger with the specified name.
    """
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


def _setup_dask(dask_distributed):
    logger = get_logger(name="setup")
    if dask_distributed:
        try:
            client = Client(DASK_SCHEDULER_ADDRESS)
            pn.state.cache["dask_client"] = client
            logger.info("Client connected to Dask local cluster")
        except OSError as e:
            raise OSError(
                "Error connecting to Dask local cluster. Note that the cluster should "
                "be started separately by running `python -m climepi.app.cluster` (in "
                "a separate terminal) before starting the app."
            ) from e
    else:
        logger.info("Using the Dask thread-based scheduler")


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
    pn.state.cache["controllers"] = {
        **pn.state.cache.get("controllers", {}),
        session_id: controller,
    }

    # Ensure temp files are cleaned up both when a session is closed

    def _cleanup_session(session_context):
        _session_destroyed(session_id=session_context.id)

    pn.state.on_session_destroyed(_cleanup_session)

    return template


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


def _session_destroyed(session_id):
    # Clean up the session
    controller = pn.state.cache["controllers"].pop(session_id)
    controller.cleanup_temp_file()
    logger = get_logger(name="session")
    logger.info("Session cleaned up successfully (deleted temporary file(s))")


def _set_shutdown():
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint(signum, frame):
        logger = get_logger(name="stop")
        logger.info("Cleaning up sessions")
        session_ids = list(pn.state.cache["controllers"].keys())
        for session_id in session_ids:
            _session_destroyed(session_id=session_id)
        if "dask_client" in pn.state.cache:
            logger.info("Closing Dask client")
            client = pn.state.cache["dask_client"]
            client.cancel(client.futures, force=True)
            client.close()
        logger.info("Stopping app")
        original_sigint(signum, frame)

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)
