import panel as pn
import param

import climepi
import climepi.climdata.cesm as cesm
import climepi.epimod
import climepi.epimod.ecolniche as ecolniche


@pn.cache
def get_clim_visualizer(clim_ds_name):
    """Load climate data from the data source."""
    if clim_ds_name in ["CESM2", "Also CESM2"]:
        ds_clim = cesm.load_example_data()
        base_modes_clim = {
            "type": "climate",
            "spatial": "global",
            "temporal": "monthly",
            "ensemble": "ensemble",
        }
        clim_visualizer = DataVisualizer(ds_clim, base_modes_clim)
        return clim_visualizer
    raise ValueError(f"Unknown climate dataset: {clim_ds_name}")


@pn.cache
def get_epi_visualizer(clim_ds_name, epi_model_name):
    """Get and run the epidemiological model."""
    clim_visualizer = get_clim_visualizer(clim_ds_name)
    if clim_ds_name in ["CESM2", "Also CESM2"] and epi_model_name in [
        "Kaye ecological niche",
        "Also Kaye ecological niche",
    ]:
        epi_model = ecolniche.import_kaye_model()
        base_modes_epi = {
            "type": "climate",
            "spatial": "global",
            "temporal": "monthly",
            "ensemble": "ensemble",
        }
    else:
        raise ValueError(f"Unknown epidemiological model: {epi_model_name}")
    ds_clim = clim_visualizer.ds_base
    ds_clim.epimod.model = epi_model
    ds_epi = ds_clim.epimod.run_model()
    epi_visualizer = DataVisualizer(ds_epi, base_modes_epi)
    return epi_visualizer


class MainController(param.Parameterized):
    """Controller parameters for the dashboard side panel."""

    clim_ds_name = param.ObjectSelector(
        default="CESM2", objects=["CESM2", "Also CESM2"]
    )
    clim_data_load_initiator = param.Boolean(default=False)
    clim_data_load_status = param.String(default="Data not loaded")
    epi_model_name = param.ObjectSelector(
        default="Kaye ecological niche",
        objects=["Kaye ecological niche", "Also Kaye ecological niche"],
    )
    epi_model_run_initiator = param.Boolean(default=False)
    epi_model_run_status = param.String(default="Model has not been run")

    def __init__(self, **params):
        super().__init__(**params)
        sidebar_widgets = {
            "clim_ds_name": {"name": "Climate dataset"},
            "clim_data_load_initiator": pn.widgets.Button(name="Load data"),
            "clim_data_load_status": {
                "widget_type": pn.widgets.StaticText,
                "name": "",
            },
            "epi_model_name": {"name": "Epidemiological model"},
            "epi_model_run_initiator": pn.widgets.Button(name="Run model"),
            "epi_model_run_status": {
                "widget_type": pn.widgets.StaticText,
                "name": "",
            },
        }
        self.sidebar_controls = pn.Param(self, widgets=sidebar_widgets, show_name=False)
        self.clim_visualizer = EmptyVisualizer("climate")
        self.epi_visualizer = EmptyVisualizer("epidemic")

    @param.depends("clim_data_load_initiator", watch=True)
    def _load_clim_data(self):
        """Load data from the data source."""
        if not self.clim_data_load_initiator:
            return
        self.clim_data_load_status = "Loading data..."
        self.clim_visualizer = get_clim_visualizer(self.clim_ds_name)
        self.clim_data_load_status = "Data loaded"
        self.epi_model_run_status = "Model has not been run"

    @param.depends("epi_model_run_initiator", watch=True)
    def _run_epi_model(self):
        """Setup and run the epidemiological model."""
        if not self.epi_model_run_initiator:
            return
        if not self.clim_data_load_initiator:
            self.epi_model_run_status = "Need to load climate data"
            self.epi_model_run_initiator = False
            return
        self.epi_model_run_status = "Running model..."
        self.epi_visualizer = get_epi_visualizer(self.clim_ds_name, self.epi_model_name)
        self.epi_model_run_status = "Model run complete"

    @param.depends("clim_ds_name", watch=True)
    def _revert_clim_data_load(self):
        """Revert the climate data load."""
        self.clim_visualizer = EmptyVisualizer("climate")
        self.clim_data_load_initiator = False
        self.clim_data_load_status = "Data not loaded"

    @param.depends("clim_ds_name", "epi_model_name", watch=True)
    def _revert_epi_model_run_(self):
        """Revert the epi model run."""
        self.epi_visualizer = EmptyVisualizer("epidemic")
        self.epi_model_run_initiator = False
        self.epi_model_run_status = "Model has not been run"


class DataVisualizer(param.Parameterized):
    """Data visualizer for the dashboard."""

    data_var = param.ObjectSelector()
    plot_type = param.ObjectSelector()
    location = param.ObjectSelector()
    temporal_mode = param.ObjectSelector()
    year_range = param.Range()
    ensemble_mode = param.ObjectSelector()
    realization = param.Integer()

    def __init__(self, ds_in, base_modes, **params):
        # base_modes should be a dictionary with the following keys and currently
        # supported values:
        #     base_modes = {
        #         "type": "climate" or "epidemic",
        #         "spatial": "global",
        #         "temporal": "monthly" or "annual",
        #         "ensemble": "ensemble",
        #     }
        super().__init__(**params)
        self.ds_base = ds_in
        self._base_modes = base_modes
        self._ds_dict = None
        self._fill_ds_dict()
        self._initialise_fixed_param_choices()
        self._update_variable_param_choices()
        self.controls = None
        self._update_controls()

    def _fill_ds_dict(self):
        ds_base = self.ds_base
        ds_dict = {}
        # if (self._base_modes["temporal"] == "monthly") and (
        #     self._base_modes["ensemble"] == "ensemble"
        # ):
        #     ds_monthly_ensemble = ds_base
        #     ds_monthly_ensemble_stats = ds_base.climepi.ensemble_stats()
        #     ds_annual_ensemble = ds_base.climepi.annual_mean()
        #     ds_annual_ensemble_stats = ds_annual_ensemble.climepi.ensemble_stats()
        #     ds_dict["monthly"] = {
        #         "ensemble": ds_monthly_ensemble,
        #         "ensemble_stats": ds_monthly_ensemble_stats,
        #     }
        #     ds_dict["annual"] = {
        #         "ensemble": ds_annual_ensemble,
        #         "ensemble_stats": ds_annual_ensemble_stats,
        #     }
        # if (self._base_modes["temporal"] == "annual") and (
        #     self._base_modes["ensemble"] == "ensemble"
        # ):
        #     ds_annual_ensemble = ds_base
        #     ds_annual_ensemble_stats = ds_annual_ensemble.climepi.ensemble_stats()
        #     ds_dict["annual"] = {
        #         "ensemble": ds_annual_ensemble,
        #         "ensemble_stats": ds_annual_ensemble_stats,
        #     }
        # else:
        #     raise NotImplementedError(
        #         "Only monthly and annual ensembles are currently supported"
        #     )
        self._ds_dict = ds_dict

    def _initialise_fixed_param_choices(self):
        # Data variable choices
        if self._base_modes["type"] == "climate":
            data_var_choices = ["temperature", "precipitation"]
        elif self._base_modes["type"] == "epidemic":
            data_var_choices = ["months_suitable"]
        else:
            raise ValueError("Only climate and epidemic data types are defined")
        self.param.data_var.objects = data_var_choices
        self.param.data_var.default = data_var_choices[0]
        self.data_var = data_var_choices[0]
        # Plot type choices
        if self._base_modes["spatial"] == "global":
            plot_type_choices = ["time_series", "map"]
        else:
            raise NotImplementedError("Only global spatial mode is currently supported")
        self.param.plot_type.objects = plot_type_choices
        self.param.plot_type.default = plot_type_choices[0]
        self.plot_type = plot_type_choices[0]
        # Location choices
        if self._base_modes["spatial"] == "global":
            location_choices = ["Miami", "Cape Town"]
        else:
            raise NotImplementedError("Only global spatial mode is currently supported")
        self.param.location.objects = location_choices
        self.param.location.default = location_choices[0]
        # Year range choices
        self.param.year_range.bounds = [
            self.ds_base.time.values[0].year,
            self.ds_base.time.values[-1].year,
        ]
        self.param.year_range.default = [
            self.ds_base.time.values[0].year,
            self.ds_base.time.values[-1].year,
        ]
        # Ensemble member choices
        if self._base_modes["ensemble"] == "ensemble":
            self.param.realization.bounds = self.ds_base.realization.values[[0, -1]]
            self.param.realization.default = self.ds_base.realization.values[0]

    @param.depends("plot_type_choice")
    def _update_variable_param_choices(self):
        # Temporal mode choices
        if (
            self.plot_type == "time_series"
            and self._base_modes["temporal"] == "monthly"
        ):
            temporal_mode_choices = [
                "annual",
                "monthly",
            ]
        elif (
            self.plot_type == "time_series" and self._base_modes["temporal"] == "annual"
        ):
            temporal_mode_choices = [
                "annual",
            ]
        elif self.plot_type == "map" and self._base_modes["temporal"] in [
            "monthly",
            "annual",
        ]:
            temporal_mode_choices = [
                "annual",
                "difference between years",
            ]
        else:
            raise NotImplementedError(
                "Only monthly and annual temporal modes are currently supported"
            )
        self.param.temporal_mode.objects = temporal_mode_choices
        self.param.temporal_mode.default = temporal_mode_choices[0]
        # Ensemble mode choices
        if (
            self.plot_type == "time_series"
            and self._base_modes["ensemble"] == "ensemble"
        ):
            ensemble_mode_choices = [
                "mean",
                "mean_ci",
                "min",
                "max",
                "single_run",
            ]
        elif self.plot_type == "map" and self._base_modes["ensemble"] == "ensemble":
            ensemble_mode_choices = [
                "mean",
                "min",
                "max",
                "single_run",
            ]
        else:
            raise NotImplementedError("Only ensemble mode is currently supported")
        self.param.ensemble_mode.objects = ensemble_mode_choices
        self.param.ensemble_mode.default = ensemble_mode_choices[0]

    @param.depends("plot_type", "ensemble_mode")
    def _update_controls(self):
        plot_widgets = {
            "data_var": {"name": "Data variable"},
            "plot_type": {"name": "Plot type"},
            "temporal_mode": {"name": "Temporal mode"},
            "ensemble_mode": {"name": "Ensemble mode"},
        }
        if self.plot_type == "time_series":
            plot_widgets["location"] = {"name": "Location"}
        if self.ensemble_mode == "single_run":
            plot_widgets["realization"] = {"name": "Realization"}
        self.controls = pn.Param(self, widgets=plot_widgets, show_name=False)


class EmptyVisualizer:
    """Empty visualizer for the dashboard."""

    def __init__(self, ds_type):
        if ds_type == "climate":
            self.controls = pn.Column("Climate data not loaded")
        if ds_type == "epidemic":
            self.controls = pn.Column("Epidemiogical model not run")
