"""Module defining the layout of the climepi app and providing a method to run it."""

import panel as pn

from climepi.app._app_classes_methods import Controller


def get_app(
    clim_dataset_example_base_dir=None,
    clim_dataset_example_names=None,
    epi_model_example_names=None,
    enable_custom_epi_model=True,
):
    """
    Get a `Panel` template object defining the layout of the climepi app.

    Parameters
    ----------
    clim_dataset_example_base_dir: str or pathlib.Path
        Base directory for the example climate datasets, optional. If None, the datasets
        will be downloaded to and accessed from the OS cache.
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
    """
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

    def _cleanup_temp_file(session_context):
        controller.cleanup_temp_file()

    pn.state.on_session_destroyed(_cleanup_temp_file)

    return template


def run_app(
    clim_dataset_example_base_dir=None,
    clim_dataset_example_names=None,
    epi_model_example_names=None,
    enable_custom_epi_model=True,
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

    def _get_app():
        return get_app(
            clim_dataset_example_base_dir=clim_dataset_example_base_dir,
            clim_dataset_example_names=clim_dataset_example_names,
            epi_model_example_names=epi_model_example_names,
            enable_custom_epi_model=enable_custom_epi_model,
        )

    pn.serve({"/climepi_app": _get_app})
