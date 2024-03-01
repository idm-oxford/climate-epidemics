"""Module defining the classes and methods underlying the climepi app."""

import atexit
import functools
import pathlib

import dask.diagnostics
import numpy as np
import panel as pn
import param
import xarray as xr
from xcdat.temporal import _infer_freq

import climepi  # noqa # pylint: disable=unused-import
from climepi import climdata, epimod

# Constants

_EXAMPLE_CLIM_DATASET_NAMES = climdata.EXAMPLE_NAMES
_EXAMPLE_CLIM_DATASET_NAMES.append("The googly")
_EXAMPLE_CLIM_DATASET_GETTER_DICT = {
    name: functools.partial(climdata.get_example_dataset, name=name)
    for name in _EXAMPLE_CLIM_DATASET_NAMES
}

_EXAMPLE_EPI_MODEL_NAMES = [
    "Kaye ecological niche",
    "Also Kaye ecological niche",
    "The flipper",
]
_EXAMPLE_EPI_MODEL_GETTER_DICT = {
    name: epimod.ecolniche.get_kaye_model for name in _EXAMPLE_EPI_MODEL_NAMES
}
_EXAMPLE_EPI_MODEL_GETTER_DICT["The flipper"] = functools.partial(ValueError, "Ouch!")

_TEMP_FILE_DIR = pathlib.Path(__file__).parent / "temp"
_TEMP_FILE_DIR.mkdir(exist_ok=True, parents=True)

# Global variables

_file_ds_dict = {}

# Pure functions


def _load_clim_data_func(clim_dataset_name):
    # Load climate data from the data source.
    ds_clim = _EXAMPLE_CLIM_DATASET_GETTER_DICT[clim_dataset_name]()
    return ds_clim


def _run_epi_model_func(ds_clim, epi_model_name):
    # Get and run the epidemiological model.
    epi_model = _EXAMPLE_EPI_MODEL_GETTER_DICT[epi_model_name]()
    ds_clim.epimod.model = epi_model
    ds_suitability = ds_clim.epimod.run_model()
    ds_suitability = _compute_to_file_reopen(ds_suitability, "suitability")
    ds_months_suitable = ds_suitability.epimod.months_suitable()
    ds_months_suitable = _compute_to_file_reopen(ds_months_suitable, "months_suitable")
    return ds_months_suitable


def _get_scope_dict(ds_in):
    temporal_scope_xcdat = _infer_freq(ds_in.time)
    xcdat_freq_map = {"year": "yearly", "month": "monthly", "day": "daily"}
    temporal_scope = xcdat_freq_map[temporal_scope_xcdat]
    spatial_scope = "single" if len(ds_in.lon) == 1 and len(ds_in.lat) == 1 else "grid"
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


def _compute_to_file_reopen(ds_in, name, dask_scheduler=None):
    temp_file_path = _TEMP_FILE_DIR / f"{name}.nc"
    try:
        _file_ds_dict[name].close()
    except KeyError:
        pass
    try:
        temp_file_path.unlink()
    except FileNotFoundError:
        pass
    chunks = ds_in.chunks.mapping
    delayed_obj = ds_in.to_netcdf(temp_file_path, compute=False)
    with dask.diagnostics.ProgressBar():
        delayed_obj.compute(scheduler=dask_scheduler)
    _file_ds_dict[name] = xr.open_dataset(temp_file_path, chunks=chunks)
    ds_out = _file_ds_dict[name].copy()
    return ds_out


@atexit.register
def _cleanup_temp_files():
    for name, ds_file in _file_ds_dict.items():
        ds_file.close()
        temp_file_path = _TEMP_FILE_DIR / f"{name}.nc"
        temp_file_path.unlink()
    print("Deleted temporary files.")


# Classes


class _Plotter:
    """Class for generating plots"""

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
            p2 = ds_plot.climepi.plot_time_series()
            plot = p1 * p2
        elif plot_type == "variance decomposition":
            plot = ds_plot.climepi.plot_var_decomp()
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
        ds_plot = ds_plot.climepi.sel_data_var(data_var)
        self._ds_plot = ds_plot

    def _spatial_index_ds_plot(self):
        plot_type = self._plot_settings["plot_type"]
        location = self._plot_settings["location"]
        spatial_scope_base = self._scope_dict_base["spatial"]
        ds_plot = self._ds_plot
        if spatial_scope_base == "single" or plot_type == "map":
            pass
        elif spatial_scope_base == "grid" and plot_type in [
            "time series",
            "variance decomposition",
        ]:
            ds_plot = ds_plot.climepi.sel_geopy(location)
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
        if temporal_scope not in ["difference between years", temporal_scope_base] or (
            temporal_scope == "difference between years"
            and temporal_scope_base != "yearly"
        ):
            ds_plot = ds_plot.climepi.temporal_group_average(frequency=temporal_scope)
        if temporal_scope == "difference between years":
            year_range = self._plot_settings["year_range"]
            ds_plot = ds_plot.sel(time=str(year_range[0])) - ds_plot.sel(
                time=str(year_range[1])
            )
            if "time_bnds" in ds_plot:
                ds_plot = ds_plot.drop_vars("time_bnds")
        self._ds_plot = ds_plot

    def _ensemble_ops_ds_plot(self):
        plot_type = self._plot_settings["plot_type"]
        ensemble_stat = self._plot_settings["ensemble_stat"]
        ds_plot = self._ds_plot
        if plot_type == "map" and ensemble_stat in [
            "mean",
            "std",
            "var",
            "min",
            "max",
            "lower",
            "upper",
        ]:
            ds_plot = ds_plot.climepi.ensemble_stats().sel(ensemble_stat=ensemble_stat)
        self._ds_plot = ds_plot


class _PlotController(param.Parameterized):
    """Plot controller class."""

    plot_type = param.ObjectSelector(precedence=1)
    data_var = param.ObjectSelector(precedence=1)
    location = param.String(default="London", precedence=-1)
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
            "location": {"name": "Location"},
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
        # Data variable choices
        data_var_choices = ds_base.climepi.get_non_bnd_data_vars()
        self.param.data_var.objects = data_var_choices
        self.param.data_var.default = data_var_choices[0]
        # Plot type choices
        if scope_dict_base["spatial"] == "grid":
            plot_type_choices = ["time series", "map", "variance decomposition"]
        elif scope_dict_base["spatial"] == "single":
            plot_type_choices = ["time series", "variance decomposition"]
        self.param.plot_type.objects = plot_type_choices
        self.param.plot_type.default = plot_type_choices[0]
        # Temporal scope choices
        if scope_dict_base["temporal"] == "yearly":
            temporal_scope_choices = ["yearly", "difference between years"]
        elif scope_dict_base["temporal"] == "monthly":
            temporal_scope_choices = ["monthly", "yearly", "difference between years"]
        elif scope_dict_base["temporal"] == "daily":
            temporal_scope_choices = [
                "daily",
                "monthly",
                "yearly",
                "difference between years",
            ]
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
        elif scope_dict_base["scenario"] == "single":
            self.param.scenario.precedence = -1
        # Model choices
        if scope_dict_base["model"] == "multiple":
            model_choices = ["all", *ds_base.model.values.tolist()]
            self.param.model.objects = model_choices
            self.param.model.default = model_choices[0]
        elif scope_dict_base["model"] == "single":
            self.param.model.precedence = -1
        # Realization choices
        if scope_dict_base["ensemble"] == "multiple":
            realization_choices = ["all", *ds_base.realization.values.tolist()]
            self.param.realization.objects = realization_choices
            self.param.realization.default = realization_choices[0]
        elif scope_dict_base["ensemble"] == "single":
            self.param.realization.precedence = -1
        # Ensemble stat choices
        ensemble_stat_choices = [
            "individual realization(s)",
            "mean",
            "std",
            "var",
            "min",
            "max",
            "lower",
            "upper",
        ]
        self.param.ensemble_stat.objects = ensemble_stat_choices
        self.param.ensemble_stat.default = ensemble_stat_choices[0]
        # Set parameters to defaults
        for par in [
            "plot_type",
            "data_var",
            "location",
            "temporal_scope",
            "year_range",
            "scenario",
            "model",
            "realization",
            "ensemble_stat",
        ]:
            setattr(self, par, self.param[par].default)
        # Update precedence (in turn triggering update to plot status)
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
    def _update_precedence(self):
        if (
            self.plot_type == "time series"
            and self._scope_dict_base["spatial"] == "grid"
        ):
            self.param.location.precedence = 1
        else:
            self.param.location.precedence = -1
        if self.plot_type == "map":
            self.param.ensemble_stat.precedence = 1
        else:
            self.param.ensemble_stat.precedence = -1
        self._revert_plot_status()

    @param.depends(
        "data_var",
        "location",
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

    clim_dataset_name = param.ObjectSelector(
        default=_EXAMPLE_CLIM_DATASET_NAMES[0],
        objects=_EXAMPLE_CLIM_DATASET_NAMES,
        precedence=1,
    )
    clim_data_load_initiator = param.Event(default=False, precedence=1)
    clim_data_loaded = param.Boolean(default=False, precedence=-1)
    clim_data_status = param.String(default="Data not loaded", precedence=1)
    epi_model_name = param.ObjectSelector(
        default=_EXAMPLE_EPI_MODEL_NAMES[0],
        objects=_EXAMPLE_EPI_MODEL_NAMES,
        precedence=1,
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

    def __init__(self, **params):
        super().__init__(**params)
        self._ds_clim = None
        data_widgets = {
            "clim_dataset_name": {"name": "Climate dataset"},
            "clim_data_load_initiator": pn.widgets.Button(name="Load data"),
            "clim_data_status": {
                "widget_type": pn.widgets.StaticText,
                "name": "",
            },
            "epi_model_name": {"name": "Epidemiological model"},
            "epi_model_run_initiator": pn.widgets.Button(name="Run model"),
            "epi_model_status": {
                "widget_type": pn.widgets.StaticText,
                "name": "",
            },
        }
        self.data_controls = pn.Param(self, widgets=data_widgets, show_name=False)

    # @param.depends()
    def clim_plot_controls(self):
        """The climate data plot controls."""
        return self.clim_plot_controller.controls

    @param.depends("clim_plot_controller.view_refresher")
    def clim_plot_view(self):
        """The climate data plot."""
        return self.clim_plot_controller.view

    # @param.depends()
    def epi_plot_controls(self):
        """The epidemiological model plot controls."""
        return self.epi_plot_controller.controls

    @param.depends("epi_plot_controller.view_refresher")
    def epi_plot_view(self):
        """The epidemiological model plot."""
        return self.epi_plot_controller.view

    @param.depends("clim_data_load_initiator", watch=True)
    def _load_clim_data(self):
        # Load data from the data source.
        if self.clim_data_loaded:
            return
        try:
            self.clim_data_status = "Loading data..."
            ds_clim = _load_clim_data_func(self.clim_dataset_name)
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

    @param.depends("epi_model_run_initiator", watch=True)
    def _run_epi_model(self):
        # Setup and run the epidemiological model.
        if self.epi_model_ran:
            return
        if not self.clim_data_loaded:
            self.epi_model_status = "Need to load climate data"
            return
        try:
            self.epi_model_status = "Running model..."
            ds_epi = _run_epi_model_func(self._ds_clim, self.epi_model_name)
            self.epi_plot_controller.initialize(ds_epi)
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

    @param.depends("clim_dataset_name", "epi_model_name", watch=True)
    def _revert_epi_model_run_status(self):
        # Revert the epi model run status (but retain data for plotting).
        self.epi_model_status = "Model has not been run"
        self.epi_model_ran = False
