"""Module defining the classes and methods underlying the climepi app."""

import pathlib
import tempfile

import dask.diagnostics
import numpy as np
import panel as pn
import param
import xarray as xr
import xcdat.temporal

from climepi import climdata, epimod
from climepi._core import ClimEpiDatasetAccessor  # noqa
from climepi.utils import get_data_var_and_bnds, list_non_bnd_data_vars

# Pure functions


def _load_clim_data_func(clim_dataset_name, base_dir):
    # Load climate data from the data source.
    ds_clim = climdata.get_example_dataset(
        clim_dataset_name, base_dir=base_dir, parallel=True, engine="h5netcdf"
    )
    return ds_clim


def _get_epi_model_func(example_name=None, temperature_range=None):
    if example_name is not None and temperature_range is None:
        # Get the example model.
        epi_model = epimod.get_example_model(example_name)
    elif example_name is None and temperature_range is not None:
        # Create a new suitability model.
        epi_model = epimod.SuitabilityModel(temperature_range=temperature_range)
    else:
        raise ValueError(
            "Exactly one of example_name and temperature_range must be provided"
        )
    return epi_model


def _run_epi_model_func(
    ds_clim,
    epi_model,
    return_months_suitable=False,
    suitability_threshold=0,
    save_path=None,
):
    # Get and run the epidemiological model.
    ds_suitability = ds_clim.climepi.run_epi_model(epi_model)
    if return_months_suitable:
        save_path_suitability = save_path.parent / "ds_suitability.nc"
    else:
        save_path_suitability = save_path
    ds_suitability = _compute_to_file_reopen(ds_suitability, save_path_suitability)
    if return_months_suitable:
        ds_months_suitable = ds_suitability.climepi.months_suitable(
            suitability_threshold=suitability_threshold
        )
        ds_months_suitable = _compute_to_file_reopen(ds_months_suitable, save_path)
        ds_suitability.close()
        save_path_suitability.unlink()
        return ds_months_suitable
    return ds_suitability


def _get_scope_dict(ds_in):
    # pylint: disable-next=protected-access
    temporal_scope_xcdat = xcdat.temporal._infer_freq(ds_in.time)  # noqa
    xcdat_freq_map = {"year": "yearly", "month": "monthly", "day": "daily"}
    temporal_scope = xcdat_freq_map[temporal_scope_xcdat]
    spatial_scope = (
        "single"
        if ds_in.lon.size == 1 and ds_in.lat.size == 1
        else "list"
        if "location" in ds_in.dims
        else "grid"
    )
    ensemble_scope = (
        "multiple"
        if "realization" in ds_in.dims and len(ds_in.realization) > 1
        else "single"
    )
    scenario_scope = (
        "multiple" if "scenario" in ds_in.dims and len(ds_in.scenario) > 1 else "single"
    )
    model_scope = (
        "multiple" if "model" in ds_in.dims and len(ds_in.model) > 1 else "single"
    )
    scope_dict = {
        "temporal": temporal_scope,
        "spatial": spatial_scope,
        "ensemble": ensemble_scope,
        "scenario": scenario_scope,
        "model": model_scope,
    }
    return scope_dict


def _get_view_func(ds_in, plot_settings):
    plotter = _Plotter(ds_in, plot_settings)
    plotter.generate_plot()
    view = plotter.view
    return view


def _compute_to_file_reopen(ds_in, save_path):
    chunks = ds_in.chunks.mapping
    delayed_obj = ds_in.to_netcdf(save_path, compute=False)
    with dask.diagnostics.ProgressBar():
        delayed_obj.compute()
    ds_out = xr.open_dataset(save_path, chunks=chunks)
    return ds_out


# Classes


class _Plotter:
    """Class for generating plots."""

    def __init__(self, ds_in, plot_settings):
        self.view = None
        self._ds_base = ds_in
        self._scope_dict_base = _get_scope_dict(ds_in)
        self._plot_settings = plot_settings
        self._ds_plot = None

    def generate_plot(self):
        """Generate the plot."""
        self._get_ds_plot()
        ds_plot = self._ds_plot
        plot_settings = self._plot_settings
        plot_type = plot_settings["plot_type"]
        if plot_type == "map":
            plot = ds_plot.climepi.plot_map()
        elif plot_type == "time series":
            p1 = ds_plot.climepi.plot_ci_plume()
            p2 = ds_plot.rename(
                scenario="example scenario",
                model="example model",
                realization="example realization",
            ).climepi.plot_time_series(label="Example trajectory")
            plot = (p1 * p2).opts(legend_position="top_left")
        elif plot_type == "variance decomposition":
            p1 = ds_plot.climepi.plot_var_decomp(fraction=False)
            p2 = ds_plot.climepi.plot_var_decomp(fraction=True)
            plot = (p1 + p2).cols(1).opts(shared_axes=False)
        else:
            raise ValueError("Unsupported plot options")
        view = pn.panel(
            plot,
            center=True,
            widget_location="bottom",
            linked_axes=False,
        )
        self.view = view

    def _get_ds_plot(self):
        self._ds_plot = self._ds_base
        self._sel_data_var_ds_plot()
        self._spatial_index_ds_plot()
        self._temporal_index_ds_plot()
        self._ensemble_index_ds_plot()
        self._model_index_ds_plot()
        self._scenario_index_ds_plot()
        self._temporal_ops_ds_plot()
        self._ensemble_ops_ds_plot()

    def _sel_data_var_ds_plot(self):
        data_var = self._plot_settings["data_var"]
        ds_plot = self._ds_plot
        ds_plot = get_data_var_and_bnds(ds_plot, data_var)
        self._ds_plot = ds_plot

    def _spatial_index_ds_plot(self):
        plot_type = self._plot_settings["plot_type"]
        spatial_scope_base = self._scope_dict_base["spatial"]
        ds_plot = self._ds_plot
        if spatial_scope_base == "single" or plot_type == "map":
            pass
        elif spatial_scope_base == "list":
            location = self._plot_settings["location_selection"]
            if location != "all":
                ds_plot = ds_plot.sel(location=location)
        elif spatial_scope_base == "grid" and plot_type in [
            "time series",
            "variance decomposition",
        ]:
            location = self._plot_settings["location_string"]
            ds_plot = ds_plot.climepi.sel_geo(location)
        else:
            raise ValueError("Unsupported spatial base scope and plot type combination")
        self._ds_plot = ds_plot

    def _temporal_index_ds_plot(self):
        temporal_scope = self._plot_settings["temporal_scope"]
        year_range = self._plot_settings["year_range"]
        ds_plot = self._ds_plot
        if temporal_scope == "difference between years":
            if any(year not in ds_plot.time.dt.year.values for year in year_range):
                raise ValueError(
                    """Only years in the dataset can be used as a range with the
                    'difference between years' temporal scope."""
                )
            ds_plot = ds_plot.isel(time=ds_plot.time.dt.year.isin(year_range))
        else:
            ds_plot = ds_plot.sel(time=slice(str(year_range[0]), str(year_range[1])))
        self._ds_plot = ds_plot

    def _ensemble_index_ds_plot(self):
        realization = self._plot_settings["realization"]
        ensemble_scope_base = self._scope_dict_base["ensemble"]
        ds_plot = self._ds_plot
        if ensemble_scope_base == "multiple" and realization != "all":
            ds_plot = ds_plot.sel(realization=realization)
        self._ds_plot = ds_plot

    def _model_index_ds_plot(self):
        model = self._plot_settings["model"]
        model_scope_base = self._scope_dict_base["model"]
        ds_plot = self._ds_plot
        if model_scope_base == "multiple" and model != "all":
            ds_plot = ds_plot.sel(model=model)
        self._ds_plot = ds_plot

    def _scenario_index_ds_plot(self):
        scenario = self._plot_settings["scenario"]
        scenario_scope_base = self._scope_dict_base["scenario"]
        ds_plot = self._ds_plot
        if scenario_scope_base == "multiple" and scenario != "all":
            ds_plot = ds_plot.sel(scenario=scenario)
        self._ds_plot = ds_plot

    def _temporal_ops_ds_plot(self):
        temporal_scope = self._plot_settings["temporal_scope"]
        temporal_scope_base = self._scope_dict_base["temporal"]
        ds_plot = self._ds_plot
        if temporal_scope not in ["difference between years", temporal_scope_base]:
            ds_plot = ds_plot.climepi.temporal_group_average(frequency=temporal_scope)
        if temporal_scope == "difference between years":
            if temporal_scope_base != "yearly":
                ds_plot = ds_plot.climepi.yearly_average()
            if "time_bnds" in ds_plot:
                ds_plot = ds_plot.drop("time_bnds")
            year_range = self._plot_settings["year_range"]
            ds_plot = ds_plot.sel(time=str(year_range[1])).squeeze(
                "time", drop=True
            ) - ds_plot.sel(time=str(year_range[0])).squeeze("time", drop=True)
        self._ds_plot = ds_plot

    def _ensemble_ops_ds_plot(self):
        plot_type = self._plot_settings["plot_type"]
        ensemble_stat = self._plot_settings["ensemble_stat"]
        ds_plot = self._ds_plot
        if plot_type == "map" and ensemble_stat != "individual realization(s)":
            ds_plot = ds_plot.climepi.ensemble_stats().sel(ensemble_stat=ensemble_stat)
        self._ds_plot = ds_plot


class _PlotController(param.Parameterized):
    """Plot controller class."""

    plot_type = param.ObjectSelector(precedence=1)
    data_var = param.ObjectSelector(precedence=1)
    location_string = param.String(default="[Type location]", precedence=-1)
    location_selection = param.ObjectSelector(precedence=-1)
    temporal_scope = param.ObjectSelector(precedence=1)
    year_range = param.Range(precedence=1)
    scenario = param.ObjectSelector(precedence=1)
    model = param.ObjectSelector(precedence=1)
    realization = param.ObjectSelector(precedence=1)
    ensemble_stat = param.ObjectSelector(precedence=1)
    plot_initiator = param.Event(precedence=1)
    plot_generated = param.Boolean(default=False, precedence=-1)
    plot_status = param.String(default="Plot not yet generated", precedence=1)
    view_refresher = param.Event(precedence=-1)

    def __init__(self, ds_in=None, **params):
        super().__init__(**params)
        self.view = pn.Row()
        self.controls = pn.Row()
        self._ds_base = None
        self._scope_dict_base = None
        self.initialize(ds_in)

    @param.depends()
    def initialize(self, ds_in=None):
        """Initialize the plot controller."""
        self.view.clear()
        self.param.trigger("view_refresher")
        self.controls.clear()
        self._ds_base = ds_in
        if ds_in is None:
            self._scope_dict_base = None
            return
        self._scope_dict_base = _get_scope_dict(ds_in)
        self._initialize_params()
        widgets = {
            "plot_type": {"name": "Plot type"},
            "data_var": {"name": "Data variable"},
            "location_string": {"name": "Location"},
            "location_selection": {"name": "Location"},
            "temporal_scope": {"name": "Temporal"},
            "year_range": {"name": "Year range"},
            "scenario": {"name": "Scenario"},
            "model": {"name": "Model"},
            "realization": {"name": "Realization"},
            "ensemble_stat": {
                "name": "Ensemble statistic (estimated if only one realization)"
            },
            "plot_initiator": pn.widgets.Button(name="Generate plot"),
            "plot_status": {
                "widget_type": pn.widgets.StaticText,
                "name": "",
            },
        }
        self.controls.append(pn.Param(self, widgets=widgets, show_name=False))

    @param.depends()
    def _initialize_params(self):
        ds_base = self._ds_base
        scope_dict_base = self._scope_dict_base
        # Plot type choices
        if scope_dict_base["spatial"] == "grid":
            plot_type_choices = ["time series", "map", "variance decomposition"]
        elif scope_dict_base["spatial"] in ["single", "list"]:
            plot_type_choices = ["time series", "variance decomposition"]
        self.param.plot_type.objects = plot_type_choices
        self.param.plot_type.default = plot_type_choices[0]
        # Data variable choices
        data_var_choices = list_non_bnd_data_vars(ds_base)
        self.param.data_var.objects = data_var_choices
        self.param.data_var.default = data_var_choices[0]
        # Location choices
        if scope_dict_base["spatial"] == "list":
            location_values = ["all", *ds_base.location.values.tolist()]
            self.param.location_selection.objects = location_values
            self.param.location_selection.default = location_values[0]
        # Temporal scope choices
        if scope_dict_base["temporal"] == "yearly":
            temporal_scope_choices = ["yearly"]
        elif scope_dict_base["temporal"] == "monthly":
            temporal_scope_choices = ["yearly", "monthly"]
        elif scope_dict_base["temporal"] == "daily":
            temporal_scope_choices = ["yearly", "monthly", "daily"]
        self.param.temporal_scope.objects = temporal_scope_choices
        self.param.temporal_scope.default = temporal_scope_choices[0]
        # Year range choices
        data_years = np.unique(ds_base.time.dt.year.values)
        self.param.year_range.bounds = (
            data_years[0],
            data_years[-1],
        )
        self.param.year_range.default = (
            data_years[0],
            data_years[-1],
        )
        data_year_diffs = np.diff(data_years)
        if np.all(data_year_diffs == data_year_diffs[0]):
            self.param.year_range.step = data_year_diffs[0]
        else:
            self.param.year_range.step = 1
        # Scenario choices
        if scope_dict_base["scenario"] == "multiple":
            scenario_choices = ["all", *ds_base.scenario.values.tolist()]
            self.param.scenario.objects = scenario_choices
            self.param.scenario.default = scenario_choices[0]
            self.param.scenario.precedence = 1
        elif scope_dict_base["scenario"] == "single":
            self.param.scenario.precedence = -1
        # Model choices
        if scope_dict_base["model"] == "multiple":
            model_choices = ["all", *ds_base.model.values.tolist()]
            self.param.model.objects = model_choices
            self.param.model.default = model_choices[0]
            self.param.model.precedence = 1
        elif scope_dict_base["model"] == "single":
            self.param.model.precedence = -1
        # Realization choices
        if scope_dict_base["ensemble"] == "multiple":
            realization_choices = ["all", *ds_base.realization.values.tolist()]
            self.param.realization.objects = realization_choices
            self.param.realization.default = realization_choices[0]
            self.param.realization.precedence = 1
        elif scope_dict_base["ensemble"] == "single":
            self.param.realization.precedence = -1
        # Ensemble stat choices
        ensemble_stat_choices = [
            "mean",
            "std",
            "var",
            "min",
            "max",
            "lower",
            "upper",
            "individual realization(s)",
        ]
        self.param.ensemble_stat.objects = ensemble_stat_choices
        self.param.ensemble_stat.default = ensemble_stat_choices[0]
        # Set parameters to defaults
        for par in [
            "plot_type",
            "data_var",
            "location_string",
            "location_selection",
            "temporal_scope",
            "year_range",
            "scenario",
            "model",
            "realization",
            "ensemble_stat",
        ]:
            setattr(self, par, self.param[par].default)
        # Update variable parameter choices and precedence
        self._update_variable_param_choices()
        self._update_precedence()

    @param.depends("plot_initiator", watch=True)
    def _update_view(self):
        # Update the plot view.
        if self.plot_generated:
            return
        self.view.clear()  # figure sizing issue if not done before generating new plot
        self.param.trigger("view_refresher")
        self.plot_status = "Generating plot..."
        try:
            ds_base = self._ds_base
            plot_settings = self.param.values()
            view = _get_view_func(ds_base, plot_settings)
            self.view.append(view)
            self.param.trigger("view_refresher")
            self.plot_status = "Plot generated"
            self.plot_generated = True
        except Exception as exc:
            self.plot_status = f"Plot generation failed: {exc}"
            raise

    @param.depends("plot_type", watch=True)
    def _update_variable_param_choices(self):
        # Add difference between years to temporal scope if plot type is map, and
        # remove it if it is not.
        if (
            self.plot_type == "map"
            and "difference between years" not in self.param.temporal_scope.objects
        ):
            self.param.temporal_scope.objects.append("difference between years")
        elif (
            self.plot_type != "map"
            and "difference between years" in self.param.temporal_scope.objects
        ):
            self.param.temporal_scope.objects.remove("difference between years")
            self.temporal_scope = self.param.temporal_scope.default

    @param.depends("plot_type", watch=True)
    def _update_precedence(self):
        if self._scope_dict_base["spatial"] == "grid" and self.plot_type != "map":
            self.param.location_string.precedence = 1
        else:
            self.param.location_string.precedence = -1
        if self._scope_dict_base["spatial"] == "list":
            self.param.location_selection.precedence = 1
        else:
            self.param.location_selection.precedence = -1
        if self.plot_type == "map":
            self.param.ensemble_stat.precedence = 1
        else:
            self.param.ensemble_stat.precedence = -1
        self._revert_plot_status()

    @param.depends(
        "data_var",
        "location_string",
        "location_selection",
        "temporal_scope",
        "year_range",
        "scenario",
        "model",
        "realization",
        "ensemble_stat",
        watch=True,
    )
    def _revert_plot_status(self):
        # Revert the plot status (but retain plot view). Some degeneracy here as this
        # can be called multiple times when changing a single parameter.
        self.plot_status = "Plot not yet generated"
        self.plot_generated = False


class Controller(param.Parameterized):
    """Main controller class for the dashboard."""

    clim_dataset_name = param.ObjectSelector(precedence=1)
    clim_data_load_initiator = param.Event(default=False, precedence=1)
    clim_data_loaded = param.Boolean(default=False, precedence=-1)
    clim_data_status = param.String(default="Data not loaded", precedence=1)
    epi_model_option = param.ObjectSelector(
        default="Example model",
        objects=["Example model", "Custom temperature-dependent suitability model"],
        precedence=-1,
    )
    epi_example_name = param.ObjectSelector(precedence=1)
    epi_temperature_range = param.Range(
        default=(15, 30), bounds=(0, 50), step=0.25, precedence=-1
    )
    epi_output_choice = param.ObjectSelector(
        objects=[
            "Months of suitability each year",
            "Suitability values",
        ],
        default="Months of suitability each year",
        precedence=-1,
    )
    suitabilty_threshold = param.Number(
        default=0, bounds=(0, 1), step=0.01, precedence=-1
    )
    epi_model_run_initiator = param.Event(default=False, precedence=1)
    epi_model_ran = param.Boolean(default=False, precedence=-1)
    epi_model_status = param.String(default="Model has not been run", precedence=1)
    clim_plot_controller = param.ClassSelector(
        default=_PlotController(), class_=_PlotController, precedence=-1
    )
    epi_plot_controller = param.ClassSelector(
        default=_PlotController(), class_=_PlotController, precedence=-1
    )

    def __init__(
        self,
        clim_dataset_example_base_dir=None,
        clim_dataset_example_names=None,
        epi_model_example_names=None,
        enable_custom_epi_model=True,
        **params,
    ):
        super().__init__(**params)
        self.param.clim_dataset_name.objects = (
            clim_dataset_example_names or climdata.EXAMPLE_NAMES
        )
        self.param.clim_dataset_name.default = self.param.clim_dataset_name.objects[0]
        self.clim_dataset_name = self.param.clim_dataset_name.default
        self.param.epi_example_name.objects = (
            epi_model_example_names or epimod.EXAMPLE_NAMES
        )
        self.param.epi_example_name.default = self.param.epi_example_name.objects[0]
        self.epi_example_name = self.param.epi_example_name.default
        if enable_custom_epi_model:
            self.param.epi_model_option.precedence = 1
        self._clim_dataset_example_base_dir = clim_dataset_example_base_dir
        self._ds_clim = None
        self._epi_model = None
        self._ds_epi = None
        self._ds_epi_path = (
            pathlib.Path(tempfile.mkdtemp(suffix="_climepi_app")) / "ds_epi.nc"
        )
        data_widgets = {
            "clim_dataset_name": {"name": "Climate dataset"},
            "clim_data_load_initiator": pn.widgets.Button(name="Load data"),
            "clim_data_status": {
                "widget_type": pn.widgets.StaticText,
                "name": "",
            },
            "epi_model_option": {"name": "Epidemiological model option"},
            "epi_example_name": {"name": "Example epidemiological model"},
            "epi_temperature_range": {"name": "Temperature range of suitability (°C)"},
            "epi_output_choice": {"name": "Return"},
            "suitabilty_threshold": {"name": "Suitability threshold"},
            "epi_model_run_initiator": pn.widgets.Button(name="Run model"),
            "epi_model_status": {
                "widget_type": pn.widgets.StaticText,
                "name": "",
            },
        }
        self.data_controls = pn.Param(self, widgets=data_widgets, show_name=False)
        self._get_epi_model()

    def clim_plot_controls(self):
        """Return the climate data plot controls."""
        return self.clim_plot_controller.controls

    @param.depends("clim_plot_controller.view_refresher")
    def clim_plot_view(self):
        """Return the climate data plot."""
        return self.clim_plot_controller.view

    @param.depends("_get_epi_model")
    def epi_model_plot_view(self):
        """Return the epidemiological model plot."""
        epi_model = self._epi_model
        try:
            plot = epi_model.plot_suitability_region()
            view = pn.Row(plot)
        except Exception as exc:
            view = pn.Row(f"Error generating plot: {exc}")
        return view

    def epi_plot_controls(self):
        """Return the epidemiological projection plot controls."""
        return self.epi_plot_controller.controls

    @param.depends("epi_plot_controller.view_refresher")
    def epi_plot_view(self):
        """Return the epidemiological projection plot."""
        return self.epi_plot_controller.view

    @param.depends("clim_data_load_initiator", watch=True)
    def _load_clim_data(self):
        # Load data from the data source.
        if self.clim_data_loaded:
            return
        try:
            self.clim_data_status = "Loading data..."
            ds_clim = _load_clim_data_func(
                self.clim_dataset_name, base_dir=self._clim_dataset_example_base_dir
            )
            self._ds_clim = ds_clim
            self.clim_plot_controller.initialize(ds_clim)
            self.clim_data_status = "Data loaded"
            self.clim_data_loaded = True
            self.epi_plot_controller.initialize()
            self.epi_model_status = "Model has not been run"
            self.epi_model_ran = False
        except Exception as exc:
            self.clim_data_status = f"Data load failed: {exc}"
            raise

    @param.depends(
        "epi_model_option", "epi_example_name", "epi_temperature_range", watch=True
    )
    def _get_epi_model(self):
        self._epi_model = None
        epi_model = None
        if self.epi_model_option == "Example model":
            example_name = self.epi_example_name
            temperature_range = None
        elif self.epi_model_option == "Custom temperature-dependent suitability model":
            example_name = None
            temperature_range = self.epi_temperature_range
        else:
            self.epi_model_status = "Model option does not seem to be valid"
            raise ValueError(self.epi_model_status)
        try:
            epi_model = _get_epi_model_func(
                example_name=example_name, temperature_range=temperature_range
            )
        except Exception as exc:
            self.epi_model_status = f"Error getting epidemiological model: {exc}"
            raise
        finally:
            self._epi_model = epi_model
            # Options specific to suitability models
            if isinstance(epi_model, epimod.SuitabilityModel):
                self.param.epi_output_choice.precedence = 1
                self._update_suitability_threshold()
            else:
                self.epi_output_choice = "Suitability values"
                self.param.epi_output_choice.precedence = -1

    @param.depends("epi_model_run_initiator", watch=True)
    def _run_epi_model(self):
        # Setup and run the epidemiological model.
        if self.epi_model_ran:
            return
        if not self.clim_data_loaded:
            self.epi_model_status = "Need to load climate data"
            return
        if self._epi_model is None:
            self.epi_model_status = "Need to select a valid epidemiological model"
            return
        try:
            self.epi_model_status = "Running model..."
            return_months_suitable = bool(
                self.epi_output_choice == "Months of suitability each year"
            )
            if self._ds_epi is not None:
                self._ds_epi.close()
            ds_epi = _run_epi_model_func(
                self._ds_clim,
                self._epi_model,
                return_months_suitable=return_months_suitable,
                suitability_threshold=self.suitabilty_threshold,
                save_path=self._ds_epi_path,
            )
            self._ds_epi = ds_epi
            self.epi_plot_controller.initialize(ds_epi.copy())
            self.epi_model_status = "Model run complete"
            self.epi_model_ran = True
        except Exception as exc:
            self.epi_model_status = f"Model run failed: {exc}"
            raise

    @param.depends("clim_dataset_name", watch=True)
    def _revert_clim_data_load_status(self):
        # Revert the climate data load status (but retain data for plotting).
        self.clim_data_status = "Data not loaded"
        self.clim_data_loaded = False

    @param.depends(
        "clim_dataset_name",
        "epi_model_option",
        "epi_example_name",
        "epi_temperature_range",
        "epi_output_choice",
        "suitabilty_threshold",
        watch=True,
    )
    def _revert_epi_model_run_status(self):
        # Revert the epi model run status (but retain data for plotting).
        self.epi_model_status = "Model has not been run"
        self.epi_model_ran = False

    @param.depends("epi_model_option", watch=True)
    def _update_epi_example_model_temperature_range_precedence(self):
        # Update the example model and temperature range parameter precedence.
        if self.epi_model_option == "Example model":
            self.param.epi_example_name.precedence = 1
            self.param.epi_temperature_range.precedence = -1
        elif self.epi_model_option == "Custom temperature-dependent suitability model":
            self.param.epi_example_name.precedence = -1
            self.param.epi_temperature_range.precedence = 1
        else:
            raise ValueError(
                f"Unrecognised epidemiological model option: {self.epi_model_option}"
            )

    @param.depends("epi_output_choice", watch=True)
    def _update_suitability_threshold(self):
        # Update the suitability threshold parameter.
        if self.epi_output_choice == "Suitability values":
            self.param.suitabilty_threshold.precedence = -1
        elif self.epi_output_choice == "Months of suitability each year":
            if self._epi_model.temperature_range is not None:
                self.suitabilty_threshold = 0
                self.param.suitabilty_threshold.precedence = -1
            else:
                self.param.suitabilty_threshold.bounds = (
                    0,
                    self._epi_model.get_max_suitability(),
                )
                self.param.suitabilty_threshold.precedence = 1

        else:
            raise ValueError("Unrecognised epidemiological model output choice.")

    def cleanup_temp_file(self):
        """Cleanup the temporary file created for the epidemiological model output."""
        if self._ds_epi is not None:
            self._ds_epi.close()
        if self._ds_epi_path.exists():
            self._ds_epi_path.unlink()
            self._ds_epi_path.parent.rmdir()
        print("\nDeleted temporary file.")
