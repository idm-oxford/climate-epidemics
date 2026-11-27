"""Module defining the classes and methods underlying the climepi app."""

import html
import os
import pathlib
import tempfile
from datetime import timedelta
from functools import partial
from threading import Lock
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np
import panel as pn
import param
import xarray as xr

from climepi import climdata, epimod
from climepi._core import ClimEpiDatasetAccessor  # noqa
from climepi._xcdat import _infer_freq
from climepi.app._climate_download import ClimateDownload
from climepi.app._app_ui import (
    CONTROL_HELP,
    LINK_CSS,
    StatusMessage,
    control_section,
    linkify,
    parameter_controls,
    source_divider,
)
from climepi.app._background import SessionTask
from climepi.climdata._utils import _drop_string_width_encodings
from climepi.utils import get_data_var_and_bnds, list_non_bnd_data_vars

_EXAMPLE_LOAD_LOCK = Lock()
_DOWNLOADED_DATA = "Downloaded dataset"
_YEARLY_PORTION = "Suitable portion of each year"
_SUITABILITY = "Suitability values"
# A cubic fit needs more than four observations to estimate residual variance.
_MIN_FIT_TIMESTEPS = 5


def _load_clim_data_func(
    clim_data_option: Literal["Example dataset", "Custom dataset"],
    clim_example_name: str | None = None,
    clim_example_base_dir: str | os.PathLike | None = None,
    custom_clim_data_dir: str | os.PathLike | None = None,
) -> xr.Dataset:
    if clim_data_option == "Example dataset":
        if clim_example_name is None:
            raise ValueError("Select an example climate dataset.")
        # Pooch's on-disk example cache is shared by all sessions.
        with _EXAMPLE_LOAD_LOCK:
            ds_clim = climdata.get_example_dataset(
                clim_example_name, base_dir=clim_example_base_dir
            )
    elif clim_data_option == "Custom dataset":
        # Load custom climate data from the specified directory.
        if custom_clim_data_dir is None:
            raise ValueError("Configure a directory containing custom NetCDF files.")
        ds_clim = xr.open_mfdataset(
            f"{custom_clim_data_dir}/*.nc",
            data_vars="minimal",
            chunks={},
            coords="minimal",  # will become xarray default
            compat="override",  # will become xarray default
        )
        _drop_string_width_encodings(ds_clim)
        if "time_bnds" in ds_clim:
            # Load time_bnds if present (otherwise issues in _compute_to_file_reopen as
            # encoding is not set).
            try:
                ds_clim.time_bnds.load()
            except Exception:
                ds_clim.close()
                raise
    else:
        raise ValueError(f"Unrecognised climate data option: {clim_data_option}")
    return ds_clim


def _get_epi_model_func(
    example_name: str | None = None,
    temperature_range: tuple[float, float] | None = None,
) -> epimod.SuitabilityModel:
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
    ds_clim: xr.Dataset,
    epi_model: epimod.SuitabilityModel,
    *,
    save_path: pathlib.Path,
    return_yearly_portion_suitable: bool = False,
    suitability_threshold: float = 0,
) -> xr.Dataset:
    # Get and run the epidemiological model.
    ds_suitability = ds_clim.climepi.run_epi_model(epi_model)
    if not return_yearly_portion_suitable:
        return _compute_to_file_reopen(ds_suitability, save_path)
    save_path_suitability = save_path.with_name("ds_suitability.nc")
    try:
        with _compute_to_file_reopen(ds_suitability, save_path_suitability) as ds:
            ds_portion_suitable = ds.climepi.yearly_portion_suitable(
                suitability_threshold=suitability_threshold
            )
            return _compute_to_file_reopen(ds_portion_suitable, save_path)
    finally:
        save_path_suitability.unlink(missing_ok=True)


def _get_scope_dict(ds_in: xr.Dataset) -> dict[str, str]:
    if not ds_in.sizes.get("time") or not all(
        coord in ds_in and ds_in[coord].size for coord in ("lat", "lon")
    ):
        raise ValueError(
            "Climate data must have nonempty time, latitude and longitude coordinates."
        )
    if not list_non_bnd_data_vars(ds_in):
        raise ValueError("No climate or suitability variables found in the dataset.")
    if ds_in.sizes["time"] == 1 and "time_bnds" in ds_in:
        # A subset can have just one timestep. Its bounds still give the resolution.
        temporal_scope_xcdat = _infer_freq(ds_in.time_bnds.isel(time=0))
    elif ds_in.sizes["time"] == 1:
        temporal_scope_xcdat = "year"
    else:
        temporal_scope_xcdat = _infer_freq(ds_in.time)
    xcdat_freq_map = {"year": "yearly", "month": "monthly", "day": "daily"}
    if temporal_scope_xcdat not in xcdat_freq_map:
        raise ValueError("Plotting requires daily, monthly or yearly data.")
    temporal_scope = xcdat_freq_map[temporal_scope_xcdat]
    spatial_scope = (
        "single"
        if ds_in.lon.size == 1 and ds_in.lat.size == 1
        else "list"
        if "location" in ds_in.dims
        else "grid"
    )
    return {
        "temporal": temporal_scope,
        "spatial": spatial_scope,
        **{
            label: "multiple" if ds_in.sizes.get(dim, 0) > 1 else "single"
            for label, dim in (
                ("ensemble", "realization"),
                ("scenario", "scenario"),
                ("model", "model"),
            )
        },
    }


def _get_view_func(
    ds_in: xr.Dataset, plot_settings: dict[str, Any]
) -> pn.viewable.Viewable | pn.viewable.ServableMixin:
    plotter = _Plotter(ds_in, plot_settings)
    plotter.generate_plot()
    view = plotter.view
    assert view is not None
    return view


def _compute_to_file_reopen(
    ds_in: xr.Dataset, save_path: str | os.PathLike
) -> xr.Dataset:
    chunks = dict(ds_in.chunksizes)
    ds_in.to_netcdf(save_path, engine="h5netcdf", compute=False).compute()
    return xr.open_dataset(save_path, engine="h5netcdf", chunks=chunks)


_CAPITALIZED_LABEL_CSS = """
    label { display: inline-block; }
    label::first-letter { text-transform: uppercase; }
"""


def _style_plot_widgets(widget_box: pn.Row) -> None:
    # Show the chart's dimension selectors as a wrapping row of compact controls.
    widget_box.param.update(
        margin=(12, 0, 0, 0),
        stylesheets=[":host { flex-wrap: wrap; gap: 4px 16px; }"],
    )
    for widget in widget_box:
        # The pane matches widgets to dimensions by label, so capitalize the
        # dimension name (e.g. "example model") for display only.
        widget.stylesheets = [*widget.stylesheets, _CAPITALIZED_LABEL_CSS]
        if isinstance(widget, pn.widgets.Select):
            widget.width = 200


# Classes


class _Plotter:
    """Class for generating plots."""

    def __init__(self, ds_in: xr.Dataset, plot_settings: dict[str, Any]) -> None:
        self.view: pn.viewable.Viewable | pn.viewable.ServableMixin | None = None
        self._ds_base = ds_in
        self._scope_dict_base = _get_scope_dict(ds_in)
        self._plot_settings = plot_settings
        self._ds_plot: xr.Dataset | None = None
        # Full address found when a typed location is geocoded.
        self._address = ""

    def generate_plot(self) -> None:
        """Generate the plot."""
        self._get_ds_plot()
        assert self._ds_plot is not None
        ds_plot = self._ds_plot.compute()
        plot_settings = self._plot_settings
        plot_type = plot_settings["plot_type"]
        if plot_type not in ("map", "time series", "variance decomposition"):
            raise ValueError(f"Unsupported plot type: {plot_type}")
        if not np.isfinite(ds_plot[plot_settings["data_var"]]).any().item():
            raise ValueError(
                "This selection has no data to plot. Try a different location or year range."
            )
        time_count = ds_plot.sizes.get("time", 0)
        can_estimate_uncertainty = (
            ds_plot.sizes.get("realization", 1) > 1 or time_count >= _MIN_FIT_TIMESTEPS
        )
        chart_size = {"responsive": True, "height": 340}
        if plot_type == "map":
            plot = ds_plot.climepi.plot_map(**chart_size)
        elif plot_type == "time series":
            time_axis = {**chart_size, "xticks": 4}
            if time_count == 1:
                # Bokeh can leave a datetime scatter's range collapsed to the point.
                timestamp = ds_plot.indexes["time"][0]
                days = {"daily": 1, "monthly": 31, "yearly": 365}
                span = timedelta(days=days[plot_settings["temporal_scope"]] / 2)
                plot = ds_plot.climepi.plot_time_series(
                    xlim=(timestamp - span, timestamp + span),
                    size=81,
                    color="#1759a6",
                    **time_axis,
                )
            elif not can_estimate_uncertainty:
                plot = ds_plot.climepi.plot_time_series(**time_axis)
            else:
                uncertainty = ds_plot.climepi.plot_uncertainty_interval_decomposition(
                    kwargs_baseline=time_axis, **time_axis
                )
                trajectory = ds_plot.rename(
                    {
                        key: f"example {key}"
                        for key in ["realization", "model", "scenario"]
                        if key in ds_plot
                    }
                ).climepi.plot_time_series(
                    label="Example trajectory", color="#1759a6", **time_axis
                )
                plot = (uncertainty * trajectory).opts(legend_position="top_left")
        else:
            if time_count == 1:
                raise ValueError(
                    "Variance decomposition needs at least two timestamps. "
                    "Choose a finer resolution or plot a time series."
                )
            if not can_estimate_uncertainty:
                raise ValueError(
                    "With one realization, variance decomposition needs at least "
                    "five timestamps. Choose a finer resolution or plot a time series."
                )
            p1 = ds_plot.climepi.plot_variance_decomposition(
                fraction=False, xticks=4, **chart_size
            )
            p2 = ds_plot.climepi.plot_variance_decomposition(
                fraction=True, xticks=4, **chart_size
            )
            plot = (p1 + p2).cols(1).opts(shared_axes=False)
        pane = pn.pane.HoloViews(
            plot,
            center=True,
            widget_location="bottom",
            widget_layout=pn.Row,
            linked_axes=False,
            sizing_mode="stretch_width",
        )
        _style_plot_widgets(pane.widget_box)
        view = cast(pn.viewable.Viewable, pane.layout)
        if self._address:
            # Show the match so a user can check it is the intended place.
            caption = pn.pane.HTML(
                f"Location found: {html.escape(self._address)}",
                margin=(8, 0, 0, 0),
                styles=CONTROL_HELP,
                sizing_mode="stretch_width",
            )
            view = pn.Column(view, caption, margin=0, sizing_mode="stretch_width")
        self.view = view

    def _get_ds_plot(self) -> None:
        self._ds_plot = self._ds_base
        self._sel_data_var_ds_plot()
        self._spatial_index_ds_plot()
        self._temporal_index_ds_plot()
        self._ensemble_index_ds_plot()
        self._temporal_ops_ds_plot()
        self._ensemble_ops_ds_plot()

    def _sel_data_var_ds_plot(self) -> None:
        data_var = self._plot_settings["data_var"]
        ds_plot = self._ds_plot
        assert ds_plot is not None
        ds_plot = get_data_var_and_bnds(ds_plot, data_var)
        self._ds_plot = ds_plot

    def _spatial_index_ds_plot(self) -> None:
        plot_type = self._plot_settings["plot_type"]
        spatial_scope_base = self._scope_dict_base["spatial"]
        ds_plot = self._ds_plot
        assert ds_plot is not None
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
            location = self._plot_settings["location_string"].strip()
            if not location:
                raise ValueError("Enter a location for this plot.")
            ds_plot = ds_plot.climepi.sel_geo(location)
            self._address = ds_plot["location"].attrs.get("geocoded_address", "")
        else:
            raise ValueError("Unsupported spatial base scope and plot type combination")
        self._ds_plot = ds_plot

    def _temporal_index_ds_plot(self) -> None:
        temporal_scope = self._plot_settings["temporal_scope"]
        year_range = self._plot_settings["year_range"]
        ds_plot = self._ds_plot
        assert ds_plot is not None
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

    def _ensemble_index_ds_plot(self) -> None:
        ds_plot = self._ds_plot
        assert ds_plot is not None
        for dim in ("realization", "model", "scenario"):
            if ds_plot.sizes.get(dim, 0) > 1 and self._plot_settings[dim] != "all":
                ds_plot = ds_plot.sel({dim: self._plot_settings[dim]})
        self._ds_plot = ds_plot

    def _temporal_ops_ds_plot(self) -> None:
        temporal_scope = self._plot_settings["temporal_scope"]
        temporal_scope_base = self._scope_dict_base["temporal"]
        ds_plot = self._ds_plot
        assert ds_plot is not None
        if temporal_scope not in ["difference between years", temporal_scope_base]:
            ds_plot = ds_plot.climepi.temporal_group_average(frequency=temporal_scope)
        if temporal_scope == "difference between years":
            if temporal_scope_base != "yearly":
                ds_plot = ds_plot.climepi.yearly_average()
            if "time_bnds" in ds_plot:
                ds_plot = ds_plot.drop_vars("time_bnds")
            year_range = self._plot_settings["year_range"]
            ds_plot = ds_plot.sel(time=str(year_range[1])).squeeze(
                "time", drop=True
            ) - ds_plot.sel(time=str(year_range[0])).squeeze("time", drop=True)
        self._ds_plot = ds_plot

    def _ensemble_ops_ds_plot(self) -> None:
        plot_type = self._plot_settings["plot_type"]
        ensemble_stat = self._plot_settings["ensemble_stat"]
        ds_plot = self._ds_plot
        assert ds_plot is not None
        if plot_type == "map" and ensemble_stat != "individual realization(s)":
            if (
                ds_plot.sizes.get("time", 0) < _MIN_FIT_TIMESTEPS
                and ds_plot.sizes.get("realization", 1) <= 1
            ):
                if ensemble_stat != "mean":
                    raise ValueError(
                        "This map has one realization and too few timestamps to "
                        "estimate uncertainty. Choose Mean or individual realizations."
                    )
                if "realization" in ds_plot.dims:
                    ds_plot = ds_plot.squeeze("realization", drop=True)
            else:
                ds_plot = ds_plot.climepi.ensemble_stats().sel(stat=ensemble_stat)
        self._ds_plot = ds_plot


class _PlotController(param.Parameterized):
    """Plot controller class."""

    plot_type = param.ObjectSelector(precedence=1)
    data_var = param.ObjectSelector(precedence=1)
    location_string = param.String(default="", precedence=-1)
    location_selection = param.ObjectSelector(precedence=-1)
    temporal_scope = param.ObjectSelector(precedence=1)
    year_range = param.Range(precedence=1)
    scenario = param.ObjectSelector(precedence=1)
    model = param.ObjectSelector(precedence=1)
    realization = param.ObjectSelector(precedence=1)
    ensemble_stat = param.ObjectSelector(precedence=1)
    plot_initiator = param.Event(precedence=1)
    plot_generated = param.Boolean(default=False, precedence=-1)
    plot_generating = param.Boolean(default=False, precedence=-1)
    plot_status = param.String(default="Plot not yet generated", precedence=1)

    def __init__(
        self,
        ds_in: xr.Dataset | None = None,
        *,
        task: SessionTask | None = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._task = task if task is not None else SessionTask()
        self._plot_revision = 0
        self.view = pn.Row(margin=0, sizing_mode="stretch_width")
        self.controls = pn.Row(margin=0, sizing_mode="stretch_width")
        self._generate_button = pn.widgets.Button(
            label="Generate plot",
            color="primary",
            margin=(6, 0),
            sizing_mode="stretch_width",
            disabled=self._task.param.busy,
        )
        self._ds_base: xr.Dataset | None = None
        self._scope_dict_base: dict[str, Any] | None = None
        self.initialize(ds_in)

    @param.depends()
    def initialize(self, ds_in: xr.Dataset | None = None) -> None:
        """Initialize the plot controller."""
        scope = _get_scope_dict(ds_in) if ds_in is not None else None
        self.view.clear()
        self.controls.clear()
        self._revert_plot_status()
        self._ds_base = ds_in
        self._scope_dict_base = scope
        if ds_in is None:
            return
        self._initialize_params()
        widgets = {
            "plot_type": {"label": "Plot type"},
            "data_var": {"label": "Data variable"},
            "location_string": {
                "label": "Location",
                "placeholder": "e.g. Oxford, United Kingdom",
                "description": "Geocoding uses OpenStreetMap "
                "(https://openstreetmap.org/copyright).",
            },
            "location_selection": {"label": "Location"},
            "temporal_scope": {"label": "Temporal resolution"},
            "year_range": {"label": "Year range"},
            "scenario": {"label": "Scenario"},
            "model": {"label": "Model"},
            "realization": {"label": "Realization"},
            "ensemble_stat": {
                "label": "Ensemble statistic",
                "description": "Estimated if there is only one realization.",
            },
            "plot_initiator": self._generate_button,
        }
        self.controls.append(
            pn.Column(
                parameter_controls(self, widgets),
                StatusMessage(
                    value=self.param.plot_status,
                    margin=(6, 0, 0, 0),
                    sizing_mode="stretch_width",
                ),
                margin=0,
                sizing_mode="stretch_width",
            )
        )

    @param.depends()
    def _initialize_params(self) -> None:
        ds_base = self._ds_base
        scope_dict_base = self._scope_dict_base
        assert ds_base is not None
        assert scope_dict_base is not None
        # Plot type choices
        if scope_dict_base["spatial"] == "grid":
            # A map needs no location, so it works as soon as a grid is loaded.
            plot_type_choices = ["map", "time series", "variance decomposition"]
        elif scope_dict_base["spatial"] in ["single", "list"]:
            plot_type_choices = ["time series", "variance decomposition"]
        else:
            raise ValueError(
                f"Unrecognised spatial scope: {scope_dict_base['spatial']}"
            )
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
        # Temporal resolution choices
        if scope_dict_base["temporal"] == "yearly":
            temporal_scope_choices = ["yearly"]
        elif scope_dict_base["temporal"] == "monthly":
            temporal_scope_choices = ["yearly", "monthly"]
        elif scope_dict_base["temporal"] == "daily":
            temporal_scope_choices = ["yearly", "monthly", "daily"]
        else:
            raise ValueError(
                f"Unrecognised temporal scope: {scope_dict_base['temporal']}"
            )
        data_years = np.unique(ds_base.time.dt.year.values)
        self.param.temporal_scope.objects = temporal_scope_choices
        self.param.temporal_scope.default = (
            scope_dict_base["temporal"] if data_years.size == 1 else "yearly"
        )
        # Year range choices
        self.param.year_range.bounds = (
            data_years[0],
            data_years[-1],
        )
        self.param.year_range.default = (
            data_years[0],
            data_years[-1],
        )
        self.param.year_range.precedence = 1 if data_years.size > 1 else -1
        data_year_diffs = np.diff(data_years)
        if data_year_diffs.size > 0 and np.all(data_year_diffs == data_year_diffs[0]):
            self.param.year_range.step = data_year_diffs[0]
        else:
            self.param.year_range.step = 1
        for dimension, scope_key in (
            ("scenario", "scenario"),
            ("model", "model"),
            ("realization", "ensemble"),
        ):
            selector = self.param[dimension]
            scope = scope_dict_base[scope_key]
            if scope == "multiple":
                selector.objects = ["all", *ds_base[dimension].values.tolist()]
                selector.default = "all"
                selector.precedence = 1
            elif scope == "single":
                selector.objects = []
                selector.default = None
                selector.precedence = -1
            else:
                raise ValueError(f"Unrecognised {scope_key} scope: {scope}")
        # Ensemble stat choices
        ensemble_stat_choices = [
            "mean",
            "std",
            "var",
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
    def _update_view(self) -> None:
        ds_base = self._ds_base
        if self.plot_generated or self._task.busy or ds_base is None:
            return
        self.view.clear()  # figure sizing issue if not done before generating new plot
        revision = self._plot_revision
        self.plot_status = "Generating plot..."
        self.plot_generating = True

        def apply(view) -> None:
            self.plot_generating = False
            if self._plot_revision != revision:
                return
            self.view.append(view)
            self.plot_status = "Plot generated"
            self.plot_generated = True

        def fail(exc: Exception) -> None:
            self.plot_generating = False
            if self._plot_revision == revision:
                self.plot_status = f"Plot generation failed: {exc}"

        self._task.run(
            partial(_get_view_func, ds_base, self.param.values()), apply, fail
        )

    @param.depends("plot_type", watch=True)
    def _update_variable_param_choices(self) -> None:
        difference = "difference between years"
        selector = self.param.temporal_scope
        years = self.param.year_range.bounds
        if self.plot_type == "map" and years is not None and years[0] != years[1]:
            if difference not in selector.objects:
                selector.objects = [*selector.objects, difference]
        elif difference in selector.objects:
            if self.temporal_scope == difference:
                self.temporal_scope = selector.default
            selector.objects = [
                choice for choice in selector.objects if choice != difference
            ]

    @param.depends("plot_type", watch=True)
    def _update_precedence(self) -> None:
        scope_dict_base = self._scope_dict_base
        assert scope_dict_base is not None
        if scope_dict_base["spatial"] == "grid" and self.plot_type != "map":
            self.param.location_string.precedence = 1
        else:
            self.param.location_string.precedence = -1
        if scope_dict_base["spatial"] == "list":
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
    def _revert_plot_status(self) -> None:
        # Revert the plot status (but retain plot view). Some degeneracy here as this
        # can be called multiple times when changing a single parameter.
        self._plot_revision += 1
        self.plot_status = "Plot not yet generated"
        self.plot_generated = False


class Controller(param.Parameterized):
    """Main controller class for the dashboard."""

    clim_data_option = param.ObjectSelector(
        default="Example dataset",
        objects=["Example dataset", "Custom dataset"],
        precedence=-1,
    )
    clim_example_name = param.ObjectSelector(precedence=1)
    clim_example_doc = param.String(precedence=1)
    clim_data_load_initiator = param.Event(default=False, precedence=1)
    clim_data_loaded = param.Boolean(default=False, precedence=-1)
    clim_data_status = param.String(default="Data not loaded", precedence=1)
    epi_model_option = param.ObjectSelector(
        default="Example model",
        objects=["Example model", "Custom temperature-dependent suitability model"],
        precedence=-1,
    )
    epi_example_name = param.ObjectSelector(precedence=1)
    epi_example_doc = param.String(precedence=1)
    epi_temperature_range = param.Range(
        default=(15, 30), bounds=(0, 50), step=0.25, precedence=-1
    )
    epi_output_choice = param.ObjectSelector(
        objects=[_YEARLY_PORTION, _SUITABILITY],
        default=_YEARLY_PORTION,
        precedence=1,
    )
    suitability_threshold = param.Number(
        default=0, bounds=(0, 1), step=0.01, precedence=-1
    )
    epi_model_run_initiator = param.Event(default=False, precedence=1)
    epi_model_ran = param.Boolean(default=False, precedence=-1)
    epi_model_status = param.String(default="Model has not been run", precedence=1)

    def __init__(
        self,
        clim_dataset_example_base_dir: str | os.PathLike | None = None,
        clim_dataset_example_names: list[str] | None = None,
        enable_custom_clim_dataset: bool = False,
        custom_clim_data_dir: str | os.PathLike | None = None,
        epi_model_example_names: list[str] | None = None,
        enable_custom_epi_model: bool = True,
        enable_climate_download: bool = False,
        **params: Any,
    ) -> None:
        self._task = SessionTask()
        self._clim_revision = 0
        self._epi_revision = 0
        super().__init__(**params)
        self.clim_plot_controller = _PlotController(task=self._task)
        self.epi_plot_controller = _PlotController(task=self._task)
        self.param.clim_example_name.objects = (
            clim_dataset_example_names or climdata.EXAMPLE_NAMES
        )
        self.param.clim_example_name.default = self.param.clim_example_name.objects[0]
        self.clim_example_name = self.param.clim_example_name.default
        if enable_custom_clim_dataset:
            self.param.clim_data_option.precedence = 1
        self._update_clim_example_name_doc()
        self.param.epi_example_name.objects = (
            epi_model_example_names or epimod.EXAMPLE_NAMES
        )
        self.param.epi_example_name.default = self.param.epi_example_name.objects[0]
        self.epi_example_name = self.param.epi_example_name.default
        if enable_custom_epi_model:
            self.param.epi_model_option.precedence = 1
        self._update_epi_example_doc()
        self._custom_clim_data_dir = custom_clim_data_dir
        self._clim_dataset_example_base_dir = clim_dataset_example_base_dir
        self._ds_clim: xr.Dataset | None = None
        self._downloaded_climate: xr.Dataset | None = None
        self._downloaded_climate_name = ""
        self._epi_model: epimod.SuitabilityModel | None = None
        self._ds_epi: xr.Dataset | None = None
        self._temp_dir = tempfile.TemporaryDirectory(suffix="_climepi_app")
        self._ds_epi_path = pathlib.Path(self._temp_dir.name) / "ds_epi.nc"
        self.climate_download: ClimateDownload | None = None
        if enable_climate_download:
            self.climate_download = ClimateDownload(self._use_downloaded_climate)
        self._load_button = pn.widgets.Button(
            label="Load data",
            color="primary",
            margin=(6, 0),
            sizing_mode="stretch_width",
            disabled=self._task.param.busy,
        )
        self._run_button = pn.widgets.Button(
            label="Run model",
            color="primary",
            margin=(6, 0),
            sizing_mode="stretch_width",
            disabled=self._task.param.busy,
        )
        help_widget = {
            "widget_type": pn.widgets.StaticText,
            "label": "",
            "styles": CONTROL_HELP,
            "stylesheets": [LINK_CSS],
        }
        clim_widgets = {
            "clim_data_option": {"label": "Climate data option"},
            "clim_example_name": {"label": "Example dataset"},
            "clim_example_doc": help_widget,
            "clim_data_load_initiator": self._load_button,
        }
        model_widgets = {
            "epi_model_option": {"label": "Model option"},
            "epi_example_name": {"label": "Example model"},
            "epi_example_doc": help_widget,
            "epi_temperature_range": {"label": "Suitable temperature range (°C)"},
            "epi_output_choice": {"label": "Model output"},
            "suitability_threshold": {"label": "Suitability threshold"},
            "epi_model_run_initiator": self._run_button,
        }
        climate_section = control_section(
            "Climate data", parameter_controls(self, clim_widgets)
        )
        if self.climate_download is not None:
            climate_section.extend([source_divider(), self.climate_download.button])
        climate_section.append(
            StatusMessage(
                value=self.param.clim_data_status,
                margin=(6, 0, 0, 0),
                sizing_mode="stretch_width",
            )
        )
        self.data_controls = pn.Column(
            climate_section,
            control_section(
                "Suitability model",
                parameter_controls(self, model_widgets),
                StatusMessage(
                    value=self.param.epi_model_status,
                    margin=(6, 0, 0, 0),
                    sizing_mode="stretch_width",
                ),
            ),
            margin=0,
            sizing_mode="stretch_width",
        )
        self._get_epi_model()

    @param.depends("_get_epi_model")
    def epi_model_plot_view(self) -> pn.Row:
        """Return the epidemiological model plot."""
        epi_model = self._epi_model
        if epi_model is None:
            return pn.Row(
                "Select a valid epidemiological model to see its suitability curve."
            )
        if self.epi_model_option == "Example model":
            title = self.epi_example_name
        else:
            low, high = self.epi_temperature_range
            title = f"Custom model: suitable from {low:g} to {high:g} °C"
        try:
            plot = epi_model.plot_suitability(responsive=True, height=340, title=title)
            view = pn.Row(plot, sizing_mode="stretch_width")
        except Exception as exc:
            view = pn.Row(f"Error generating plot: {exc}")
        return view

    @param.depends("clim_data_load_initiator", watch=True)
    def _load_clim_data(self) -> None:
        if self.clim_data_loaded or self._task.busy:
            return
        if self.clim_data_option == _DOWNLOADED_DATA:
            if self._downloaded_climate is not None:
                self._set_climate_dataset(
                    self._downloaded_climate, name=self._downloaded_climate_name
                )
            return

        revision = self._clim_revision
        name = (
            self.clim_example_name if self.clim_data_option == "Example dataset" else ""
        )
        self.clim_data_status = "Loading data..."

        def apply(ds_clim: xr.Dataset) -> None:
            if self._clim_revision != revision:
                ds_clim.close()
                return
            try:
                self._set_climate_dataset(ds_clim, name=name)
            except Exception:
                ds_clim.close()
                raise

        def fail(exc: Exception) -> None:
            if self._clim_revision == revision:
                self.clim_data_status = f"Data load failed: {exc}"

        self._task.run(
            partial(
                _load_clim_data_func,
                self.clim_data_option,
                clim_example_name=self.clim_example_name,
                clim_example_base_dir=self._clim_dataset_example_base_dir,
                custom_clim_data_dir=self._custom_clim_data_dir,
            ),
            apply,
            fail,
            discard=xr.Dataset.close,
        )

    def _set_climate_dataset(self, ds_clim: xr.Dataset, *, name: str = "") -> None:
        # Initializing the plot controls validates the dataset before it is adopted.
        self.clim_plot_controller.initialize(ds_clim)
        self.epi_plot_controller.initialize()
        self._clear_epi_output()
        previous, self._ds_clim = self._ds_clim, ds_clim
        self._close_if_unused(previous)
        plot_scope = self.clim_plot_controller._scope_dict_base
        if (
            plot_scope is not None
            and plot_scope["temporal"] != "yearly"
            and ds_clim.sizes["time"] > 1
        ):
            self.param.epi_output_choice.objects = [_YEARLY_PORTION, _SUITABILITY]
        else:
            self.epi_output_choice = _SUITABILITY
            self.param.epi_output_choice.objects = [_SUITABILITY]
        self.clim_data_status = f"Data loaded: {name}" if name else "Data loaded"
        self.clim_data_loaded = True
        self._revert_epi_model_run_status()

    def _use_downloaded_climate(self, dataset: xr.Dataset, name: str) -> None:
        if self._task.busy:
            raise ValueError("Wait for the current operation to finish and try again.")
        self._set_climate_dataset(dataset, name=name)
        previous, self._downloaded_climate = self._downloaded_climate, dataset
        self._downloaded_climate_name = name
        self._close_if_unused(previous)
        selector = self.param.clim_data_option
        if _DOWNLOADED_DATA not in selector.objects:
            available = (
                selector.objects if selector.precedence > 0 else ["Example dataset"]
            )
            selector.objects = [*available, _DOWNLOADED_DATA]
            selector.precedence = 1
        # Switching the option reverts the load status, so restore it afterwards.
        self.clim_data_option = _DOWNLOADED_DATA
        self.clim_data_status = f"Data loaded: {name}"
        self.clim_data_loaded = True

    def _close_if_unused(self, dataset: xr.Dataset | None) -> None:
        # The downloaded subset stays open so it can be selected again.
        if dataset is not None and all(
            dataset is not kept for kept in (self._ds_clim, self._downloaded_climate)
        ):
            dataset.close()

    @param.depends(
        "epi_model_option", "epi_example_name", "epi_temperature_range", watch=True
    )
    def _get_epi_model(self) -> None:
        self._epi_model = None
        if self.epi_model_option == "Example model":
            example_name = self.epi_example_name
            temperature_range = None
        elif self.epi_model_option == "Custom temperature-dependent suitability model":
            example_name = None
            temperature_range = self.epi_temperature_range
        else:
            self.epi_model_status = (
                f"Unrecognised epidemiological model option: {self.epi_model_option}"
            )
            raise ValueError(self.epi_model_status)
        try:
            epi_model = _get_epi_model_func(
                example_name=example_name, temperature_range=temperature_range
            )
        except Exception as exc:
            self.epi_model_status = f"Error getting epidemiological model: {exc}"
            raise
        self._epi_model = epi_model
        self._update_suitability_threshold()

    @param.depends("epi_model_run_initiator", watch=True)
    def _run_epi_model(self) -> None:
        if self.epi_model_ran or self._task.busy:
            return
        ds_clim = self._ds_clim
        if not self.clim_data_loaded or ds_clim is None:
            self.epi_model_status = "Need to load climate data"
            return
        epi_model = self._epi_model
        if epi_model is None:
            self.epi_model_status = "Need to select a valid epidemiological model"
            return
        revision = self._epi_revision
        save_path = self._ds_epi_path
        if self._ds_epi is not None:
            save_path = save_path.with_name(f"ds_epi_{uuid4().hex}.nc")
        self.epi_model_status = "Running model..."

        def discard(ds_epi: xr.Dataset) -> None:
            ds_epi.close()
            save_path.unlink(missing_ok=True)

        def apply(ds_epi: xr.Dataset) -> None:
            if self._epi_revision != revision:
                discard(ds_epi)
                return
            try:
                self.epi_plot_controller.initialize(ds_epi)
            except Exception:
                discard(ds_epi)
                raise
            self._clear_epi_output()
            self._ds_epi = ds_epi
            self._ds_epi_path = save_path
            self.epi_model_status = "Model run complete"
            self.epi_model_ran = True

        def fail(exc: Exception) -> None:
            save_path.unlink(missing_ok=True)
            if self._epi_revision == revision:
                self.epi_model_status = f"Model run failed: {exc}"

        self._task.run(
            partial(
                _run_epi_model_func,
                ds_clim,
                epi_model,
                return_yearly_portion_suitable=(
                    self.epi_output_choice == _YEARLY_PORTION
                ),
                suitability_threshold=self.suitability_threshold,
                save_path=save_path,
            ),
            apply,
            fail,
            discard=discard,
        )

    def _clear_epi_output(self) -> None:
        if self._ds_epi is not None:
            self._ds_epi.close()
            self._ds_epi = None
            self._ds_epi_path.unlink(missing_ok=True)

    @param.depends("clim_data_option", "clim_example_name", watch=True)
    def _revert_clim_data_load_status(self) -> None:
        # Revert the climate data load status (but retain data for plotting).
        self._clim_revision += 1
        self.clim_data_status = "Data not loaded"
        self.clim_data_loaded = False

    @param.depends(
        "clim_data_option",
        "clim_example_name",
        "epi_model_option",
        "epi_example_name",
        "epi_temperature_range",
        "epi_output_choice",
        "suitability_threshold",
        watch=True,
    )
    def _revert_epi_model_run_status(self) -> None:
        # Revert the epi model run status (but retain data for plotting).
        self._epi_revision += 1
        self.epi_model_status = "Model has not been run"
        self.epi_model_ran = False

    @param.depends("clim_data_option", "clim_example_name", watch=True)
    def _update_clim_example_name_doc(self) -> None:
        # Name and details of the climate dataset.
        if self.clim_data_option == "Example dataset":
            self.param.clim_example_name.precedence = 1
            self.clim_example_doc = linkify(
                str(climdata.EXAMPLES[self.clim_example_name].get("doc", ""))
            )
            self.param.clim_example_doc.precedence = 1
        elif self.clim_data_option in ("Custom dataset", _DOWNLOADED_DATA):
            self.param.clim_example_name.precedence = -1
            self.param.clim_example_doc.precedence = -1
        else:
            raise ValueError(
                f"Unrecognised climate data option: {self.clim_data_option}"
            )

    @param.depends("epi_model_option", "epi_example_name", watch=True)
    def _update_epi_example_doc(self) -> None:
        # Details of the epidemiological model.
        if self.epi_model_option == "Example model":
            self.epi_example_doc = linkify(
                epimod.EXAMPLES[self.epi_example_name].get("doc", "")
            )
            self.param.epi_example_doc.precedence = 1
        else:
            self.param.epi_example_doc.precedence = -1

    @param.depends("epi_model_option", watch=True)
    def _update_epi_example_model_temperature_range_precedence(self) -> None:
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
    def _update_suitability_threshold(self) -> None:
        # Update the suitability threshold parameter.
        if self.epi_output_choice == _SUITABILITY:
            self.param.suitability_threshold.precedence = -1
        elif self.epi_output_choice == _YEARLY_PORTION:
            self.suitability_threshold = 0
            epi_model = self._epi_model
            if (
                epi_model is None
                or epi_model.temperature_range is not None
                or (
                    epi_model.suitability_table is not None
                    and np.issubdtype(
                        epi_model.suitability_table[
                            epi_model._suitability_var_name
                        ].dtype,
                        bool,
                    )
                )
            ):
                self.param.suitability_threshold.precedence = -1
            else:
                self.param.suitability_threshold.bounds = (
                    0,
                    epi_model.get_max_suitability(),
                )
                self.param.suitability_threshold.precedence = 1
        else:
            raise ValueError("Unrecognised epidemiological model output choice.")

    def cleanup_temp_file(self) -> None:
        """Release the session's datasets and remove temporary files."""
        if self.climate_download is not None:
            self.climate_download.close()
        self._task.close(self._cleanup_datasets)

    def _cleanup_datasets(self) -> None:
        try:
            self._clear_epi_output()
            # Closing an xarray dataset twice is harmless if both refer to one.
            for dataset in (self._ds_clim, self._downloaded_climate):
                if dataset is not None:
                    dataset.close()
        finally:
            self._temp_dir.cleanup()
