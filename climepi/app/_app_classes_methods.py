import atexit
import functools
import pathlib

import dask.diagnostics
import numpy as np
import panel as pn
import param
import xarray as xr

import climepi  # noqa
import climepi.epimod  # noqa
from climepi.climdata import cesm
from climepi.epimod import ecolniche

# Constants

_EXAMPLE_CLIM_DATASET_NAMES = cesm.EXAMPLE_NAMES
_EXAMPLE_CLIM_DATASET_NAMES.append("The googly")
_EXAMPLE_CLIM_DATASET_GETTER_DICT = {
    name: functools.partial(cesm.get_example_dataset, name=name)
    for name in _EXAMPLE_CLIM_DATASET_NAMES
}

_EXAMPLE_EPI_MODEL_NAMES = [
    "Kaye ecological niche",
    "Also Kaye ecological niche",
    "The flipper",
]
_EXAMPLE_EPI_MODEL_GETTER_DICT = {
    name: ecolniche.import_kaye_model for name in _EXAMPLE_EPI_MODEL_NAMES
}
_EXAMPLE_EPI_MODEL_GETTER_DICT["The flipper"] = functools.partial(ValueError, "Ouch!")

_TEMP_FILE_DIR = pathlib.Path(__file__).parent / "temp"
_TEMP_FILE_DIR.mkdir(exist_ok=True, parents=True)

# Global variables

_file_ds_dict = {}

# Pure functions


def _load_clim_data_func(clim_dataset_name):
    """Load climate data from the data source."""
    ds_clim = _EXAMPLE_CLIM_DATASET_GETTER_DICT[clim_dataset_name]()
    return ds_clim


def _run_epi_model_func(ds_clim, epi_model_name):
    """Get and run the epidemiological model."""
    epi_model = _EXAMPLE_EPI_MODEL_GETTER_DICT[epi_model_name]()
    ds_clim.epimod.model = epi_model
    ds_suitability = ds_clim.epimod.run_model()
    ds_suitability = _compute_to_file_reopen(ds_suitability, "suitability")
    ds_months_suitable = ds_suitability.epimod.months_suitable()
    ds_months_suitable = _compute_to_file_reopen(ds_months_suitable, "months_suitable")
    return ds_months_suitable


def _compute_to_file_reopen(ds, name, dask_scheduler=None):
    temp_file_path = _TEMP_FILE_DIR / f"{name}.nc"
    try:
        _file_ds_dict[name].close()
    except KeyError:
        pass
    try:
        temp_file_path.unlink()
    except FileNotFoundError:
        pass
    chunks = ds.chunks.mapping
    climepi_modes = ds.climepi.modes
    delayed_obj = ds.to_netcdf(temp_file_path, compute=False)
    with dask.diagnostics.ProgressBar():
        delayed_obj.compute(scheduler=dask_scheduler)
    _file_ds_dict[name] = xr.open_dataset(temp_file_path, chunks=chunks)
    ds = _file_ds_dict[name].copy()
    ds.climepi.modes = climepi_modes
    return ds


@atexit.register
def _cleanup_temp_files():
    for name, ds in _file_ds_dict.items():
        ds.close()
        temp_file_path = _TEMP_FILE_DIR / f"{name}.nc"
        temp_file_path.unlink()
    print("Delted temporary files.")


# Classes


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

    def __init__(self, **params):
        super().__init__(**params)
        self._ds_clim = None
        self._clim_plot_controller = _PlotController()
        self._epi_plot_controller = _PlotController()
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

    def clim_plot_controls(self):
        """The climate data plot controls."""
        return self._clim_plot_controller.controls

    def clim_plot_view(self):
        """The climate data plot."""
        return self._clim_plot_controller.view

    def epi_plot_controls(self):
        """The epidemiological model plot controls."""
        return self._epi_plot_controller.controls

    def epi_plot_view(self):
        """The epidemiological model plot."""
        return self._epi_plot_controller.view

    @param.depends("clim_data_load_initiator", watch=True)
    def _load_clim_data(self):
        # Load data from the data source.
        if self.clim_data_loaded:
            return
        try:
            self.clim_data_status = "Loading data..."
            ds_clim = _load_clim_data_func(self.clim_dataset_name)
            self._ds_clim = ds_clim
            self._clim_plot_controller.initialize(ds_clim)
            self.clim_data_status = "Data loaded"
            self.clim_data_loaded = True
            self._epi_plot_controller.initialize()
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
            self._epi_plot_controller.initialize(ds_epi)
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


class _PlotController(param.Parameterized):
    """Plot controller class."""

    data_var = param.ObjectSelector(precedence=1)
    plot_type = param.ObjectSelector(precedence=1)
    location = param.String(default="Cape Town", precedence=-1)
    temporal_mode = param.ObjectSelector(precedence=1)
    year_range = param.Range(precedence=1)
    ensemble_mode = param.ObjectSelector(precedence=1)
    realization = param.Integer(precedence=-1)
    plot_initiator = param.Event(precedence=1)
    plot_generated = param.Boolean(default=False, precedence=-1)
    plot_status = param.String(default="Plot not yet generated", precedence=1)

    def __init__(self, ds_in=None, **params):
        super().__init__(**params)
        self._view = pn.Row()
        self._controls = pn.Row()
        self._plotter = None
        self._ds_base = None
        self._base_modes = None
        self.initialize(ds_in)

    def view(self):
        """Return the plot."""
        return self._view

    def controls(self):
        """Return the controls."""
        return self._controls

    def initialize(self, ds_in=None):
        """Initialize the plot controller."""
        self._view.clear()
        self._controls.clear()
        self._plotter = _Plotter(ds_in)
        self._ds_base = ds_in
        if ds_in is None:
            self._base_modes = None
            return
        self._ds_base = ds_in
        self._base_modes = ds_in.climepi.modes
        self._initialize_params()
        self._update_variable_param_choices()
        self._update_precedence()
        widgets = {
            "data_var": {"name": "Data variable"},
            "plot_type": {"name": "Plot type"},
            "location": {"name": "Location"},
            "temporal_mode": {"name": "Temporal mode"},
            "year_range": {"name": "Year range"},
            "ensemble_mode": {"name": "Ensemble mode"},
            "realization": {"name": "Realization"},
            "plot_initiator": pn.widgets.Button(name="Generate plot"),
            "plot_status": {
                "widget_type": pn.widgets.StaticText,
                "name": "",
            },
        }
        self._controls.append(pn.Param(self, widgets=widgets, show_name=False))

    def _initialize_params(self):
        ds_base = self._ds_base
        base_modes = self._base_modes
        # Data variable choices
        data_var_choices = ds_base.climepi.get_non_bnd_data_vars()
        self.param.data_var.objects = data_var_choices
        self.param.data_var.default = data_var_choices[0]
        # Plot type choices
        if base_modes["spatial"] == "global":
            plot_type_choices = ["time series", "map"]
        elif base_modes["spatial"] == "single":
            plot_type_choices = ["time series"]
        else:
            raise NotImplementedError(
                "Only global and single spatial modes are currently supported"
            )
        self.param.plot_type.objects = plot_type_choices
        self.param.plot_type.default = plot_type_choices[0]
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
        # Ensemble member choices
        if base_modes["ensemble"] == "ensemble":
            self.param.realization.bounds = [
                ds_base.realization.values[0].item(),
                ds_base.realization.values[-1].item(),
            ]
            self.param.realization.default = ds_base.realization.values[0].item()
        # Set parameters to defaults (automatically triggers updates to variable
        # parameter choices and precedence)
        self.data_var = self.param.data_var.default
        self.plot_type = self.param.plot_type.default
        self.location = self.param.location.default
        self.year_range = self.param.year_range.default
        self.realization = self.param.realization.default

    @param.depends("plot_initiator", watch=True)
    def _update_view(self):
        # Update the plot view.
        if self.plot_generated:
            return
        self._view.clear()  # figure sizing issue if not done before generating new plot
        self.plot_status = "Generating plot..."
        try:
            self._plotter.generate_plot(self.param.values())
            # self._view.clear()
            self._view.append(self._plotter.plot)
            self.plot_status = "Plot generated"
            self.plot_generated = True
        except Exception as exc:
            self.plot_status = f"Plot generation failed: {exc}"
            raise

    @param.depends(
        "data_var",
        "plot_type",
        "location",
        "temporal_mode",
        "year_range",
        "ensemble_mode",
        "realization",
        watch=True,
    )
    def _revert_plot_status(self):
        # Revert the plot status (but retain plot view).
        self.plot_status = "Plot not yet generated"
        self.plot_generated = False

    @param.depends("plot_type", watch=True)
    def _update_variable_param_choices(self):
        base_modes = self._base_modes
        # Temporal mode choices
        if self.plot_type == "time series" and base_modes["temporal"] == "monthly":
            temporal_mode_choices = [
                "annual",
                "monthly",
            ]
        elif self.plot_type == "time series" and base_modes["temporal"] == "annual":
            temporal_mode_choices = [
                "annual",
            ]
        elif self.plot_type == "map" and base_modes["temporal"] in [
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
        self.temporal_mode = self.param.temporal_mode.default
        # Ensemble mode choices
        if self.plot_type == "time series" and base_modes["ensemble"] == "ensemble":
            ensemble_mode_choices = [
                "mean",
                "mean and 90% confidence interval",
                "std",
                "min",
                "max",
                "single run",
            ]
        elif self.plot_type == "map" and base_modes["ensemble"] == "ensemble":
            ensemble_mode_choices = [
                "mean",
                "std",
                "min",
                "max",
                "single run",
            ]
        else:
            raise NotImplementedError(
                "Only 'ensemble' base mode is currently supported"
            )
        self.param.ensemble_mode.objects = ensemble_mode_choices
        self.param.ensemble_mode.default = ensemble_mode_choices[0]
        self.ensemble_mode = self.param.ensemble_mode.default

    @param.depends("plot_type", "ensemble_mode", watch=True)
    def _update_precedence(self):
        if self.plot_type == "time series" and self._base_modes["spatial"] == "global":
            self.param.location.precedence = 1
        else:
            self.param.location.precedence = -1
            self.location = self.param.location.default  # could remove this
        if self.ensemble_mode == "single run":
            self.param.realization.precedence = 1
        else:
            self.param.realization.precedence = -1
            self.realization = self.param.realization.default  # could remove this


class _Plotter:
    """Class for generating plots"""

    def __init__(self, ds_in=None):
        self.plot = None
        self._ds_base = ds_in
        self._base_modes = None
        if ds_in is not None:
            self._base_modes = ds_in.climepi.modes
        self._ds_plot = None
        self._plot_modes = None

    def generate_plot(self, plot_modes):
        """Generate the plot."""
        plot_type = plot_modes["plot_type"]
        ensemble_mode = plot_modes["ensemble_mode"]
        self._plot_modes = plot_modes
        self._update_ds_plot()
        ds_plot = self._ds_plot
        if plot_type == "map":
            self.plot = pn.panel(
                ds_plot.climepi.plot_map(),
                center=True,
                widget_location="bottom",
            )
        elif (
            plot_type == "time series"
            and ensemble_mode == "mean and 90% confidence interval"
        ):
            self.plot = pn.panel(
                ds_plot.climepi.plot_ensemble_ci_time_series(), center=True
            )
        elif plot_type == "time series":
            self.plot = pn.panel(ds_plot.climepi.plot_time_series(), center=True)
        else:
            raise ValueError("Unsupported plot options")

    def _update_ds_plot(self):
        self._ds_plot = self._ds_base
        self._sel_data_var_ds_plot()
        self._spatial_index_ds_plot()
        self._temporal_index_ds_plot()
        self._ensemble_index_ds_plot()
        self._temporal_ops_ds_plot()
        self._ensemble_ops_ds_plot()

    def _sel_data_var_ds_plot(self):
        data_var = self._plot_modes["data_var"]
        ds_plot = self._ds_plot
        ds_plot = ds_plot.climepi.sel_data_var(data_var)
        self._ds_plot = ds_plot

    def _spatial_index_ds_plot(self):
        plot_type = self._plot_modes["plot_type"]
        location = self._plot_modes["location"]
        spatial_base_mode = self._base_modes["spatial"]
        ds_plot = self._ds_plot
        if spatial_base_mode == "single" or plot_type == "map":
            pass
        elif spatial_base_mode == "global" and plot_type == "time series":
            ds_plot = ds_plot.climepi.sel_geopy(location)
        else:
            raise ValueError("Unsupported spatial base mode and plot type combination")
        self._ds_plot = ds_plot

    def _temporal_index_ds_plot(self):
        temporal_mode = self._plot_modes["temporal_mode"]
        year_range = self._plot_modes["year_range"]
        temporal_base_mode = self._base_modes["temporal"]
        ds_plot = self._ds_plot
        if temporal_base_mode not in ["annual", "monthly"]:
            raise ValueError("Unsupported temporal base mode")
        if temporal_mode in ["annual", "monthly"]:
            ds_plot = ds_plot.sel(time=slice(str(year_range[0]), str(year_range[1])))
        elif temporal_mode == "difference between years":
            if any(year not in ds_plot.time.dt.year.values for year in year_range):
                raise ValueError(
                    """Only years in the dataset can be used as a range with the
                    'difference between years' temporal mode"""
                )
            ds_plot = ds_plot.isel(time=ds_plot.time.dt.year.isin(year_range))
        else:
            raise ValueError(f"Unknown temporal mode: {temporal_mode}")
        self._ds_plot = ds_plot

    def _ensemble_index_ds_plot(self):
        ensemble_mode = self._plot_modes["ensemble_mode"]
        realization = self._plot_modes["realization"]
        ensemble_base_mode = self._base_modes["ensemble"]
        ds_plot = self._ds_plot
        if ensemble_base_mode != "ensemble":
            raise ValueError("Unsupported ensemble base mode")
        if ensemble_mode == "single run":
            ds_plot = ds_plot.sel(realization=realization)
        elif ensemble_mode in [
            "mean",
            "mean and 90% confidence interval",
            "std",
            "min",
            "max",
        ]:
            pass
        else:
            raise ValueError(f"Unknown ensemble mode: {ensemble_mode}")
        self._ds_plot = ds_plot

    def _temporal_ops_ds_plot(self):
        temporal_mode = self._plot_modes["temporal_mode"]
        temporal_base_mode = self._base_modes["temporal"]
        ds_plot = self._ds_plot
        if temporal_base_mode == "monthly" and temporal_mode == "monthly":
            pass
        elif temporal_base_mode == "monthly" and temporal_mode in [
            "annual",
            "difference between years",
        ]:
            ds_plot = ds_plot.climepi.annual_mean()
        elif temporal_base_mode == "annual" and temporal_mode in [
            "annual",
            "difference between years",
        ]:
            pass
        else:
            raise ValueError("Unsupported base and plot temporal mode combination")
        if temporal_mode == "difference between years":
            data_var = self._plot_modes["data_var"]
            year_range = self._plot_modes["year_range"]
            da_start = ds_plot[data_var].sel(time=str(year_range[0])).squeeze()
            da_end = ds_plot[data_var].sel(time=str(year_range[1])).squeeze()
            ds_plot[data_var] = da_end - da_start
        self._ds_plot = ds_plot

    def _ensemble_ops_ds_plot(self):
        ensemble_mode = self._plot_modes["ensemble_mode"]
        ensemble_base_mode = self._base_modes["ensemble"]
        ds_plot = self._ds_plot
        if ensemble_base_mode != "ensemble":
            raise ValueError("Unsupported ensemble base mode")
        if ensemble_mode == "single run":
            pass
        elif ensemble_mode in ["mean", "std", "min", "max"]:
            ds_plot = ds_plot.climepi.ensemble_stats().sel(
                ensemble_statistic=ensemble_mode
            )
        elif ensemble_mode == "mean and 90% confidence interval":
            ds_plot = ds_plot.climepi.ensemble_stats()
        self._ds_plot = ds_plot
