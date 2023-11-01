import panel as pn

import climepi.app

pn.extension(template="bootstrap")

data_controller = climepi.app.DataController()
# main_controller.clim_data_load_initiator = True
sidebar_controls = data_controller.sidebar_controls

sidebar_controls.servable(target="sidebar")

clim_plot_controls = data_controller.clim_plot_controls
epi_plot_controls = data_controller.epi_plot_controls

clim_plot_view = data_controller.clim_plot_view
epi_plot_view = data_controller.epi_plot_view

pn.Tabs(
    ("Climate data", pn.Row(clim_plot_controls, clim_plot_view)),
    ("Epidemiological model results", pn.Row(epi_plot_controls, epi_plot_view)),
).servable(target="main")
