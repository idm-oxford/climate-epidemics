"""
Module defining the layout of the climepi app and providing a method to run it.
"""

import panel as pn

from climepi.app._app_classes_methods import Controller


def _get_app():
    # Returns a Panel template object defining the layout of the app.
    template = pn.template.BootstrapTemplate(title="climepi app")

    controller = Controller()

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
    return template


def run_app():
    """
    Method to run the climepi `Panel` app locally in a browser.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    template = _get_app()
    pn.serve(template)
