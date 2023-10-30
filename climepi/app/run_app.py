import panel as pn

import climepi.app

pn.extension(template="bootstrap")

main_controller = climepi.app.MainController()
main_controller.clim_data_load_initiator = True
sidebar_controls = pn.Column(main_controller.sidebar_controls)

sidebar_controls.servable(target="sidebar")

clim_visualizer = main_controller.clim_visualizer
clim_plot_controls = pn.Column(clim_visualizer.controls)
epi_visualizer = main_controller.epi_visualizer
epi_plot_controls = pn.Column(epi_visualizer.controls)

pn.Tabs(
    ("Climate data", clim_plot_controls), ("Epidemic model output", epi_plot_controls)
).servable(target="main")
