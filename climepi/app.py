import panel as pn
import param

import climepi
import climepi.climdata.cesm as cesm
import climepi.epimod
import climepi.epimod.ecolniche as ecolniche

pn.extension(template="bootstrap")

# Sidebar widgets to setup data and model


def load_clim_data(clim_ds_choice):
    """Load climate data from the data source."""
    if clim_ds_choice in ["CESM2", "Also CESM2"]:
        return cesm.load_example_data()
    raise ValueError(f"Unknown climate dataset: {clim_ds_choice}")


def get_epi_model(epi_model_choice):
    """Get the epidemiological model."""
    if epi_model_choice in ["Kaye ecological niche", "Also Kaye ecological niche"]:
        return ecolniche.import_kaye_model()
    raise ValueError(f"Unknown epidemiological model: {epi_model_choice}")


class DataController(param.Parameterized):
    """Controller parameters for the dashboard side panel."""

    clim_ds_choice = param.ObjectSelector(
        default="CESM2", objects=["CESM2", "Also CESM2"]
    )
    clim_data_load_initiator = param.Boolean(default=False)
    clim_data_load_status = param.String(default="Data not loaded")
    epi_model_choice = param.ObjectSelector(
        default="Kaye ecological niche",
        objects=["Kaye ecological niche", "Also Kaye ecological niche"],
    )
    epi_model_run_initiator = param.Boolean(default=False)
    epi_model_run_status = param.String(default="Model has not been run")

    def __init__(self, **params):
        super().__init__(**params)
        self.ds_clim = None
        self.ds_epi = None
        sb_widgets = {
            "clim_ds_choice": {"name": "Climate dataset"},
            "clim_data_load_initiator": pn.widgets.Button(name="Load data"),
            "clim_data_load_status": {
                "widget_type": pn.widgets.StaticText,
                "name": "",
            },
            "epi_model_choice": {"name": "Epidemiological model"},
            "epi_model_run_initiator": pn.widgets.Button(name="Run model"),
            "epi_model_run_status": {
                "widget_type": pn.widgets.StaticText,
                "name": "",
            },
        }
        self.controls = pn.Param(self, widgets=sb_widgets, show_name=False)

    @param.depends("clim_data_load_initiator", watch=True)
    def _load_clim_data(self):
        """Load data from the data source."""
        if not self.clim_data_load_initiator:
            return
        self.clim_data_load_status = "Loading data..."
        self.ds_clim = load_clim_data(self.clim_ds_choice)
        self.clim_data_load_status = "Data loaded"
        self.epi_model_run_status = "Model has not been run"

    @param.depends("epi_model_run_initiator", watch=True)
    def _run_epi_model(self):
        """Setup and run the epidemiological model."""
        if not self.epi_model_run_initiator:
            return
        if self.ds_clim is None:
            self.epi_model_run_status = "Need to load climate data"
            self.epi_model_run_initiator = False
            return
        self.epi_model_run_status = "Running model..."
        epi_model = get_epi_model(self.epi_model_choice)
        self.ds_clim.epimod.model = epi_model
        self.ds_epi = self.ds_clim.epimod.run_model()
        self.epi_model_run_status = "Model run complete"

    @param.depends("clim_ds_choice", watch=True)
    def _revert_clim_data_load(self):
        """Revert the climate data load."""
        self.ds_clim = None
        self.clim_data_load_initiator = False
        self.clim_data_load_status = "Data not loaded"

    @param.depends("clim_ds_choice", "epi_model_choice", watch=True)
    def _revert_epi_model_run_(self):
        """Revert the epi model run."""
        self.ds_epi = None
        self.epi_model_run_initiator = False
        self.epi_model_run_status = "Model has not been run"


class DataVisualizer(param.Parameterized):
    """Data visualizer for the dashboard."""

    pass


data_controller = DataController()
pn.Column(data_controller.controls).servable(target="sidebar")


# class DataVisualizer(param.Parameterized):
#     """Data visualizer for the dashboard."""

#     pass
