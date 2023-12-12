import panel as pn

from climepi import app


def _get_app():
    template = pn.template.BootstrapTemplate(title="climepi app")

    controller = app.Controller()

    data_controls = controller.data_controls
    template.sidebar.append(data_controls)

    clim_plot_controls = controller.clim_plot_controls
    epi_plot_controls = controller.epi_plot_controls
    clim_plot_view = controller.clim_plot_view
    epi_plot_view = controller.epi_plot_view

    template.main.append(
        pn.Tabs(
            ("Climate data", pn.Row(clim_plot_controls, clim_plot_view)),
            ("Epidemiological model results", pn.Row(epi_plot_controls, epi_plot_view)),
        )
    )
    return template


def run_app():
    template = _get_app()
    pn.serve(template)
