import panel as pn

import climepi.app

pn.extension(template="bootstrap")

pn.Row().servable(title="climepi app", target="main")

controller = climepi.app.Controller()

data_controls = controller.data_controls
data_controls.servable(target="sidebar")

clim_plot_controls = controller.clim_plot_controls
epi_plot_controls = controller.epi_plot_controls
clim_plot_view = controller.clim_plot_view
epi_plot_view = controller.epi_plot_view

pn.Tabs(
    ("Climate data", pn.Row(clim_plot_controls, clim_plot_view)),
    ("Epidemiological model results", pn.Row(epi_plot_controls, epi_plot_view)),
).servable(target="main")
