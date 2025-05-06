"""
Core module for the climepi package.

Contains the ClimEpiDatasetAccessor class for xarray datasets.
"""

import geoviews.feature as gf
import holoviews as hv
import hvplot.xarray  # noqa # pylint: disable=unused-import
import numpy as np
import scipy.interpolate
import scipy.stats
import xarray as xr
from xarray.plot.utils import label_from_attrs

from climepi._ensemble_stats import (
    _ensemble_stats_direct,
    _ensemble_stats_fit,
)
from climepi._geocoding import geocode
from climepi._xcdat import (  # noqa
    BoundsAccessor,
    TemporalAccessor,
    _infer_freq,
    center_times,
)
from climepi.utils import (
    add_bnds_from_other,
    add_var_attrs_from_other,
    list_non_bnd_data_vars,
)


@xr.register_dataset_accessor("climepi")
class ClimEpiDatasetAccessor:
    """
    Accessor class for xarray datasets accessed through the ``.climepi`` attribute.

    Provides core methods, including for computing temporal and ensemble statistics, and
    for plotting.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def run_epi_model(self, epi_model, **kwargs):
        """
        Run the epidemiological model on a climate dataset.

        Parameters
        ----------
        epi_model : climepi.epimod.EpiModel
            The epidemiological model to run.
        **kwargs : dict, optional
            Keyword arguments to pass to the model's run method. For suitability models,
            passing "return_yearly_portion_suitable=True" will return the number of
            days/months suitable each year (depending on the resolution of the data),
            rather than the full suitability dataset, and additionally passing a value
            for "suitability_threshold" will set the minimum suitability threshold for a
            day/month to be considered suitable (default is 0).

        Returns
        -------
        xarray.Dataset:
            The output of the model's run method.
        """
        ds_epi = epi_model.run(self._obj, **kwargs)
        return ds_epi

    def sel_geo(self, location, lon=None, lat=None, **kwargs):
        """
        Get data for the nearest grid point(s) to a specified location(s).

        Finds the nearest grid point(s) using either provided longitude and latitude
        values, or if these are not provided, using geopy's Nominatim geocoder (uses
        OpenStreetMap data https://openstreetmap.org/copyright). Returns a dataset with
        a new "location" coordinate, which is used as a dimension coordinate in place of
        the lon and lat coordinates if multiple locations are provided.

        Parameters
        ----------
        location : str or list of str
            Name(s) of the location(s) to select. If 'lon' and 'lat' are not provided,
            the location(s) will be geocoded using geopy's Nominatim geocoder, with
            the location(s) provided used as search strings.
        lon : float or list of float, optional
            Longitude(s) of the location(s) to select. If provided, 'lat' must also be
            provided. If 'location' is a list, 'lon' and 'lat' must also be lists of the
            same length (if provided). If not provided, the location(s) will be geocoded
            using geopy's Nominatim geocoder.
        lat : float or list of float, optional
            Latitude(s) of the location(s) to select. If provided, 'lon' must also be
            provided. If 'location' is a list, 'lon' and 'lat' must also be lists of the
            same length (if provided). If not provided, the location(s) will be geocoded
            using geopy's Nominatim geocoder.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the geocode method of the Nominatim
            geocoder.

        Returns
        -------
        xarray.Dataset
            A new dataset containing the data for the specified location.
        """
        if isinstance(location, list):
            if lon is None and lat is None:
                lon = [None] * len(location)
                lat = [None] * len(location)
            ds_list = [
                self.sel_geo(location_curr, lon=lon_curr, lat=lat_curr, **kwargs)
                for location_curr, lon_curr, lat_curr in zip(
                    location, lon, lat, strict=True
                )
            ]
            concat_vars = [var for var in self._obj.data_vars if var != "time_bnds"]
            ds_new = xr.concat(
                ds_list,
                dim="location_dim",  # if "location", time_bnds seems to be concatenated
                data_vars=concat_vars,
                coords=["lat", "lon", "location"],
            ).swap_dims(location_dim="location")
            return ds_new
        if lon is None and lat is None:
            location_geopy = geocode(location, **kwargs)
            lat = location_geopy.latitude
            lon = location_geopy.longitude  # in the range [-180, 180]
        elif lon is None or lat is None:
            raise ValueError(
                "If 'lon' or 'lat' is provided, both 'lon' and 'lat' must be provided.",
            )
        lon_min = min(self._obj.lon)
        lon_max = max(self._obj.lon)
        if lon_max > 180.0001:
            # Deals with the case where the longitude co-ordinates of the datasetare in
            # the range [0, 360] (slightly crude)
            lon = lon % 360
        if lon > lon_max and (lon_min - (lon - 360)) < (lon - lon_max):
            lon = lon - 360
        elif lon < lon_min and ((lon + 360) - lon_max) < (lon_min - lon):
            lon = lon + 360
        ds_new = self._obj.sel(lat=lat, lon=lon, method="nearest")
        ds_new = ds_new.assign_coords(location=location)
        return ds_new

    def temporal_group_average(self, data_var=None, frequency="yearly", **kwargs):
        """
        Compute the group average of a data variable.

        Wraps xcdat temporal.group_average.

        Parameters
        ----------
        data_var : str or list, optional
            Name(s) of the data variable(s) to compute the group average for. If not
            provided, all non-bound data variables will be used.
        frequency : str, optional
            Frequency to compute the group average for (options are "yearly", "monthly"
            or "daily"). Default is "yearly".
        **kwargs : dict, optional
            Additional keyword arguments to pass to xcdat temporal.group_average.

        Returns
        -------
        xarray.Dataset
            A new dataset containing the group average of the selected data
            variable(s) at the specified frequency.
        """
        try:
            data_var = self._process_data_var_argument(data_var)
        except ValueError:
            data_var_list = self._process_data_var_argument(data_var, as_list=True)
            return xr.merge(
                [
                    self.temporal_group_average(data_var_curr, frequency, **kwargs)
                    for data_var_curr in data_var_list
                ]
            )
        if np.issubdtype(self._obj[data_var].dtype, np.integer) or np.issubdtype(
            self._obj[data_var].dtype, bool
        ):
            # Workaround for bug in xcdat temporal.group_average using integer or
            # boolean data types
            ds_copy = self._obj.copy()
            ds_copy[data_var] = ds_copy[data_var].astype("float64")
            return ds_copy.climepi.temporal_group_average(data_var, frequency, **kwargs)
        xcdat_freq_map = {"yearly": "year", "monthly": "month", "daily": "day"}
        xcdat_freq = xcdat_freq_map[frequency]
        # Add time bounds if necessary and ensure these are loaded into memory (can
        # play weirdly with Dask otherwise)
        ds_in = self._obj.bounds.add_missing_bounds(axes="T")
        ds_in["time_bnds"] = ds_in["time_bnds"].compute()
        # Compute the group average
        ds_m = ds_in.temporal.group_average(data_var, freq=xcdat_freq, **kwargs)
        if ds_m.time.size > 1:
            # Add time bounds and center times (only if there is more than one time
            # point, as xcdat add_time_bounds does not work for a single time point)
            ds_m = ds_m.bounds.add_time_bounds(method="freq", freq=xcdat_freq)
            # Workaround for bug in xcdat.center_times when longitude and/or latitude
            # are non-dimension singleton coordinates (otherwise, longitude and/or
            # latitude are incorrectly treated as time coordinates, leading to an error
            # being raised)
            centered_times = center_times(ds_m[["time", "time_bnds"]])
            ds_m["time"] = centered_times.time
            ds_m["time_bnds"] = centered_times.time_bnds
        return ds_m

    def yearly_average(self, data_var=None, **kwargs):
        """
        Compute the yearly mean of a data variable.

        Thin wrapper around group_average.

        Parameters
        ----------
        data_var : str or list, optional
            Name(s) of the data variable(s) to compute the yearly mean for. If not
            provided, all non-bound data variables will be used.
        **kwargs : dict, optional
            Additional keyword arguments to pass to xcdat temporal.group_average.

        Returns
        -------
        xarray.Dataset
            A new dataset containing the yearly mean of the selected data variable(s).
        """
        return self.temporal_group_average(
            data_var=data_var, frequency="yearly", **kwargs
        )

    def monthly_average(self, data_var=None, **kwargs):
        """
        Compute the monthly mean of a data variable.

        Thin wrapper around group_average.

        Parameters
        ----------
        data_var : str or list, optional
            Name(s) of the data variable(s) to compute the monthly mean for. If not
            provided, all non-bound data variables will be used.
        **kwargs : dict, optional
            Additional keyword arguments to pass to xcdat temporal.group_average.

        Returns
        -------
        xarray.Dataset
            A new dataset containing the monthly mean of the selected data
            variable(s).
        """
        return self.temporal_group_average(
            data_var=data_var, frequency="monthly", **kwargs
        )

    def yearly_portion_suitable(
        self, suitability_var_name=None, suitability_threshold=0
    ):
        """
        Calculate the portion of each year that is suitable given suitability data.

        Suitability data must be provided on either a monthly or daily basis; the
        suitable portion of each year is given as the number of months or days that are
        suitable in the respective cases.

        Parameters
        ----------
        suitability_var_name : str, optional
            Name of the suitability variable to use. If not provided, the method will
            attempt to automatically select a suitable variable.
        suitability_threshold : float or int, optional
            Minimum suitability threshold for a month to be considered suitable. Default
            is 0.

        Returns
        -------
        xarray.Dataset:
            Dataset with a single non-bounds data variable "portion_suitable", with
            units of months (for monthly suitability data) or days (for daily
            suitability data) each year.
        """
        if suitability_var_name is None:
            non_bnd_data_vars = list_non_bnd_data_vars(self._obj)
            if len(non_bnd_data_vars) == 1:
                suitability_var_name = non_bnd_data_vars[0]
            elif "suitability" in non_bnd_data_vars:
                suitability_var_name = "suitability"
            else:
                raise ValueError(
                    """No suitability data found. To calculate the number of months
                    suitable from a climate dataset, first run the suitability model and
                    then apply this method to the output dataset. If the suitability
                    variable is not named "suitability", specify the name using the
                    suitability_var_name argument.""",
                )
        freq_xcdat = _infer_freq(self._obj.time)  # noqa
        if freq_xcdat not in ["month", "day"]:
            raise ValueError(
                "Suitability data must be provided on either a monthly or daily "
                f"basis. Inferred frequency of the time coordinate is '{freq_xcdat}'.",
            )
        da_suitability = self._obj[suitability_var_name]
        ds_suitable_bool = xr.Dataset(
            {"suitable": da_suitability > suitability_threshold}
        )
        ds_portion_suitable = (
            ds_suitable_bool.groupby("time.year")
            .sum("time")
            .rename(suitable="portion_suitable", year="time")
        )
        # Fairly hacky way to convert years to datetime objects and add bounds
        # (likely inefficient if not using Dask)
        ds_yearly_avg = self._obj.climepi.yearly_average(suitability_var_name)
        ds_portion_suitable = ds_portion_suitable.assign_coords(time=ds_yearly_avg.time)
        ds_portion_suitable = add_bnds_from_other(ds_portion_suitable, ds_yearly_avg)
        # Add long_name attribute
        ds_portion_suitable["portion_suitable"].attrs = {
            "long_name": f"{freq_xcdat.capitalize()}s where {suitability_var_name} > "
            + str(suitability_threshold)
        }
        return ds_portion_suitable

    def ensemble_stats(
        self,
        data_var=None,
        uncertainty_level=90,
        internal_variability_method=None,
        deg=3,
        lam=None,
    ):
        """
        Compute a range of ensemble statistics for a data variable.

        Parameters
        ----------
        data_var : str or list, optional
            Name(s) of the data variable(s) to compute the ensemble statistics for.
            If not provided, all non-bound data variables will be used.
        uncertainty_level : float, optional
            Uncertainty level (percentage) for computing ensemble percentiles. Default
            is 90.
        internal_variability_method : str, optional
            Whether to compute statistics directly at each time point ("direct") or
            to estimate them using a polynomial ("polyfit") or spline ("splinefit")
            fit to the time series, assuming the variance is constant in time. By
            default, the "direct" method is used if multiple realizations are available
            (i.e., the dataset has a non-singleton "realization" dimension), and the
            "polyfit" method is used if only a single realization is available.
        deg : int, optional
            Degree of the polynomial to fit to the time series if using the "polyfit"
            method (ignored if using other methods). Default is 3.
        lam: float, optional
            Smoothing parameter passed to scipy.interpolate.make_smoothing_spline if
            using the "splinefit" method (ignored if using other methods). Default is
            None.

        Returns
        -------
        xarray.Dataset
            A new dataset containing the computed ensemble statistics for the
            selected data variable(s).
        """
        # Process the data variable argument
        data_var_list = self._process_data_var_argument(data_var, as_list=True)
        # Drop bounds for now (re-add at end)
        ds_raw = self._obj[data_var_list]
        if internal_variability_method is None:
            internal_variability_method = (
                "direct"
                if ("realization" in self._obj.dims and self._obj.realization.size > 1)
                else "polyfit"
            )
        if internal_variability_method == "direct":
            ds_stat = _ensemble_stats_direct(
                ds_raw, uncertainty_level=uncertainty_level
            )
        elif internal_variability_method in ["polyfit", "splinefit"]:
            ds_stat = _ensemble_stats_fit(
                ds_raw,
                uncertainty_level=uncertainty_level,
                internal_variability_method=internal_variability_method,
                deg=deg,
                lam=lam,
            )
        else:
            raise ValueError(
                f"Invalid internal_variability_method '{internal_variability_method}'. "
                "Valid methods are 'direct', 'polyfit' and 'splinefit'."
            )
        ds_stat.attrs = self._obj.attrs
        ds_stat = add_var_attrs_from_other(ds_stat, self._obj, var=data_var_list)
        ds_stat = add_bnds_from_other(ds_stat, self._obj)
        return ds_stat

    def variance_decomposition(
        self,
        data_var=None,
        fraction=False,
        internal_variability_method=None,
        deg=3,
        lam=None,
    ):
        """
        Decompose variance contributions from different climate uncertainty sources.

        Partitions the variance of a data variable at each time point into contributions
        from internal variability, model uncertainty and scenario uncertainty.

        Parameters
        ----------
        data_var : str or list of str, optional
            Name of the data variable(s) to decompose.
        fraction : bool, optional
            Whether to calculate the variance contributions as fractions of the total
            variance at each time, rather than the raw variances. Default is False.
        internal_variability_method : str, optional
            Whether to characterize internal variability by computing ensemble
            statistics directly at each time point ("direct") or by estimating them
            using a polynomial ("polyfit") or spline ("splinefit") fit to the time
            series, assuming the variance is constant in time. By default, the "direct"
            method is used if multiple realizations are available (i.e., the dataset has
            a non-singleton "realization" dimension), and the "polyfit" method is used
            if only a single realization is available.
        deg : int, optional
            Degree of the polynomial to fit to the time series if using the "polyfit"
            method (ignored if using other methods). Default is 3.
        lam: float, optional
            Smoothing parameter passed to scipy.interpolate.make_smoothing_spline if
            using the "splinefit" method (ignored if using other methods). Default is
            None.

        Returns
        -------
        xarray.Dataset
            A new dataset containing the variance decomposition of the selected data
            variable(s) along a new "source" dimension.
        """
        data_var_list = self._process_data_var_argument(data_var, as_list=True)
        for dim in ["scenario", "model"]:
            # Deal with cases with a single scenario and/or model that is not a
            # dimension
            if dim not in self._obj[data_var_list].dims:
                ds_expanded = self._obj.copy()
                for data_var_curr in data_var_list:
                    ds_expanded[data_var_curr] = ds_expanded[data_var_curr].expand_dims(
                        dim=dim
                    )
                return ds_expanded.climepi.variance_decomposition(
                    data_var_list,
                    fraction=fraction,
                    internal_variability_method=internal_variability_method,
                    deg=deg,
                    lam=lam,
                )
        # Calculate or estimate ensemble statistics characterizing internal variability
        ds_stat = self.ensemble_stats(
            data_var_list,
            internal_variability_method=internal_variability_method,
            deg=deg,
            lam=lam,
        )[data_var_list]
        # Calculate the internal, model and scenario contributions to the variance
        ds_var_internal = ds_stat.sel(stat="var", drop=True).mean(
            dim=["scenario", "model"]
        )
        ds_var_model = (
            ds_stat.sel(stat="mean", drop=True).var(dim="model").mean(dim="scenario")
        )
        ds_var_scenario = (
            ds_stat.sel(stat="mean", drop=True).mean(dim="model").var(dim="scenario")
        )
        ds_var_decomp = xr.concat(
            [ds_var_internal, ds_var_model, ds_var_scenario],
            dim=xr.Variable("source", ["internal", "model", "scenario"]),
            coords="minimal",
        )
        # Express contributions as a fraction of the total variance if required
        if fraction:
            ds_var_decomp = ds_var_decomp / ds_var_decomp.sum(dim="source")
        # Copy and update attributes and bounds
        ds_var_decomp = add_bnds_from_other(ds_var_decomp, self._obj)
        ds_var_decomp.attrs = self._obj.attrs
        if fraction:
            for data_var_curr in data_var_list:
                ds_var_decomp[data_var_curr].attrs["long_name"] = "Fraction of variance"
        else:
            ds_var_decomp = add_var_attrs_from_other(
                ds_var_decomp, self._obj, var=data_var_list
            )
            for data_var_curr in data_var_list:
                if "units" in ds_var_decomp[data_var_curr].attrs:
                    units_in = ds_var_decomp[data_var_curr].attrs["units"]
                    if any(x in units_in for x in ["/", " ", "^"]):
                        units_in = "(" + units_in + ")"
                    ds_var_decomp[data_var_curr].attrs["units"] = units_in + "²"
                if "long_name" in ds_var_decomp[data_var_curr].attrs:
                    long_name_in = ds_var_decomp[data_var_curr].attrs["long_name"]
                    long_name_in = long_name_in[0].lower() + long_name_in[1:]
                    ds_var_decomp[data_var_curr].attrs["long_name"] = (
                        "Variance of " + long_name_in
                    )
                else:
                    ds_var_decomp[data_var_curr].attrs["long_name"] = (
                        "Variance of " + data_var_curr
                    )
        return ds_var_decomp

    def uncertainty_interval_decomposition(
        self,
        data_var=None,
        uncertainty_level=90,
        internal_variability_method=None,
        deg=3,
        lam=None,
    ):
        """
        Decompose uncertainty interval contributions.

        Partitions the uncertainty interval of a data variable at each time point into
        contributions from internal variability, model uncertainty and scenario
        uncertainty.

        Parameters
        ----------
        data_var : str or list of str, optional
            Name(s) of the data variable to decompose.
        uncertainty_level : float, optional
            Uncertainty level for the uncertainty intervals (percentage). Default is 90.
        internal_variability_method : str, optional
            Whether to characterize internal variability by computing ensemble
            statistics directly at each time point ("direct") or by estimating them
            using a polynomial ("polyfit") or spline ("splinefit") fit to the time
            series, assuming the variance is constant in time. By default, the "direct"
            method is used if multiple realizations are available (i.e., the dataset has
            a non-singleton "realization" dimension), and the "polyfit" method is used
            if only a single realization is available.
        deg : int, optional
            Degree of the polynomial to fit to the time series if using the "polyfit"
            method (ignored if using other methods). Default is 3.
        lam: float, optional
            Smoothing parameter passed to scipy.interpolate.make_smoothing_spline if
            using the "splinefit" method (ignored if using other methods). Default is
            None.

        Returns
        -------
        xarray.Dataset
            A new dataset containing the uncertainty interval decomposition of the
            selected data variable(s) along a new "level" dimension.
        """
        data_var_list = self._process_data_var_argument(data_var, as_list=True)
        ds_raw = self._obj[data_var_list].squeeze()  # drops bounds for now
        # Make "scenario", "model" and "realization" dimensions of the data variable if
        # they are not present or are (singleton) non-dimension coordinates (reduces
        # number of cases to handle; note this partially reverses the effect of the
        # squeeze operation above, which still removes other singleton dimensions).
        for dim in ["scenario", "model", "realization"]:
            if dim not in ds_raw.dims:
                ds_raw = ds_raw.expand_dims(dim)
        # Get ensemble statistics, baseline estimate, and if necessary a decomposition
        # of the variance and z value for approximate uncertainty intervals
        ds_stat = ds_raw.climepi.ensemble_stats(
            uncertainty_level=uncertainty_level,
            internal_variability_method=internal_variability_method,
            deg=deg,
            lam=lam,
        )
        ds_baseline = ds_stat.sel(stat="mean", drop=True).mean(
            dim=["scenario", "model"], keep_attrs=True
        )
        ds_var_decomp = ds_raw.climepi.variance_decomposition(
            fraction=False,
            internal_variability_method=internal_variability_method,
            deg=deg,
            lam=lam,
        )
        z = scipy.stats.norm.ppf(0.5 + uncertainty_level / 200)
        # Create a dataset for the uncertainty interval decomposition
        multiple_realizations = ds_raw.realization.size > 1
        if multiple_realizations or (internal_variability_method != "direct"):
            # Obtain uncertainty interval contribution from internal variability if there
            # are multiple realizations or if internal variability is to be estimated
            if ds_raw.scenario.size == 1 and ds_raw.model.size == 1:
                ds_internal_lower = ds_stat.squeeze(
                    ["model", "scenario"], drop=True
                ).sel(stat="lower", drop=True)
                ds_internal_upper = ds_stat.squeeze(
                    ["model", "scenario"], drop=True
                ).sel(stat="upper", drop=True)
            else:
                ds_std_internal = np.sqrt(
                    ds_var_decomp.sel(source="internal", drop=True)
                )
                ds_internal_lower = ds_baseline - z * ds_std_internal
                ds_internal_upper = ds_baseline + z * ds_std_internal
        else:
            ds_internal_lower = ds_baseline
            ds_internal_upper = ds_baseline
        if ds_raw.model.size > 1:
            # Model variability if there are multiple models
            if ds_raw.scenario.size == 1 and not (
                multiple_realizations or (internal_variability_method != "direct")
            ):
                ds_model_lower = (
                    ds_raw.squeeze(["scenario", "realization"], drop=True)
                    .quantile(0.5 - uncertainty_level / 200, dim="model")
                    .drop_vars("quantile")
                )
                ds_model_upper = (
                    ds_raw.squeeze(["scenario", "realization"], drop=True)
                    .quantile(0.5 + uncertainty_level / 200, dim="model")
                    .drop_vars("quantile")
                )
            else:
                ds_std_internal_model = np.sqrt(
                    ds_var_decomp.sel(source=["internal", "model"]).sum(dim="source")
                )
                ds_model_lower = ds_baseline - z * ds_std_internal_model
                ds_model_upper = ds_baseline + z * ds_std_internal_model
        else:
            ds_model_lower = ds_internal_lower
            ds_model_upper = ds_internal_upper
        if ds_raw.scenario.size > 1:
            # Scenario variability if there are multiple scenarios
            if ds_raw.model.size == 1 and not (
                multiple_realizations or (internal_variability_method != "direct")
            ):
                ds_scenario_lower = (
                    ds_raw.squeeze(["model", "realization"], drop=True)
                    .quantile(0.5 - uncertainty_level / 200, dim="scenario")
                    .drop_vars("quantile")
                )
                ds_scenario_upper = (
                    ds_raw.squeeze(["model", "realization"], drop=True)
                    .quantile(0.5 + uncertainty_level / 200, dim="scenario")
                    .drop_vars("quantile")
                )
            else:
                ds_std_internal_model_scenario = np.sqrt(
                    ds_var_decomp.sum(dim="source")
                )
                ds_scenario_lower = ds_baseline - z * ds_std_internal_model_scenario
                ds_scenario_upper = ds_baseline + z * ds_std_internal_model_scenario
        else:
            ds_scenario_lower = ds_model_lower
            ds_scenario_upper = ds_model_upper
        # Combine into a single dataset
        ds_decomp = xr.concat(
            [
                ds_scenario_lower,
                ds_model_lower,
                ds_internal_lower,
                ds_baseline,
                ds_internal_upper,
                ds_model_upper,
                ds_scenario_upper,
            ],
            dim=xr.Variable(
                "level",
                [
                    "scenario_lower",
                    "model_lower",
                    "internal_lower",
                    "baseline",
                    "internal_upper",
                    "model_upper",
                    "scenario_upper",
                ],
            ),
        )
        # Copy and update attributes and bounds
        ds_decomp.attrs = self._obj.attrs
        ds_decomp = add_bnds_from_other(ds_decomp, self._obj)
        ds_decomp = add_var_attrs_from_other(ds_decomp, self._obj, var=data_var_list)
        return ds_decomp

    def plot_time_series(self, data_var=None, **kwargs):
        """
        Generate a time series plot of a data variable.

        Wraps hvplot.line.

        Parameters
        ----------
        data_var : str, optional
            Name of the data variable to plot. If not provided, the function
            will attempt to automatically select a suitable variable.
        **kwargs : dict
            Additional keyword arguments to pass to hvplot.line.

        Returns
        -------
        hvplot object
            The resulting time series plot.
        """
        data_var = self._process_data_var_argument(data_var)
        da_plot = self._obj[data_var].squeeze()
        kwargs_hvplot = {"x": "time", **kwargs}
        plot_obj = da_plot.hvplot.line(**kwargs_hvplot)
        return plot_obj

    def plot_map(self, data_var=None, include_ocean=False, **kwargs):
        """
        Generate a map plot of a data variable.

        Wraps hvplot.quadmesh.

        Parameters
        ----------
        data_var : str, optional
            Name of the data variable to plot. If not provided, the function
            will attempt to automatically select a suitable variable.
        include_ocean : bool, optional
            Whether or not to include ocean data in the plot. Default is False.
        **kwargs : dict, optional
            Additional keyword arguments to pass to hvplot.quadmesh.

        Returns
        -------
        hvplot object
            The resulting map plot.
        """
        data_var = self._process_data_var_argument(data_var)
        da_plot = self._obj[data_var].squeeze()
        kwargs_hvplot = {
            "x": "lon",
            "y": "lat",
            "cmap": "viridis",
            "project": True,
            "geo": True,
            "rasterize": True,
            "coastline": True,
            "dynamic": False,
            **kwargs,
        }
        plot_obj = da_plot.hvplot.quadmesh(**kwargs_hvplot)
        if not include_ocean:
            plot_obj *= gf.ocean.options(fill_color="white")
        return plot_obj

    def plot_variance_decomposition(
        self,
        data_var=None,
        fraction=False,
        internal_variability_method=None,
        deg=3,
        lam=None,
        **kwargs,
    ):
        """
        Plot decomposition of variance from different climate uncertainty sources.

        Partitions the variance of a data variable at each time point into contributions
        from internal variability, model uncertainty and scenario uncertainty, and
        creates an area plot showing these contributions over time.

        Wraps hvplot.area.

        Parameters
        ----------
        data_var : str
            Name of the data variable to plot.
        fraction : bool, optional
            Whether to plot the variance contributions as fractions of the total
            variance at each time, rather than the raw variances. Default is False.
        internal_variability_method : str, optional
            Whether to characterize internal variability by computing ensemble
            statistics directly at each time point ("direct") or by estimating them
            using a polynomial ("polyfit") or spline ("splinefit") fit to the time
            series, assuming the variance is constant in time. By default, the "direct"
            method is used if multiple realizations are available (i.e., the dataset has
            a non-singleton "realization" dimension), and the "polyfit" method is used
            if only a single realization is available.
        deg : int, optional
            Degree of the polynomial to fit to the time series if using the "polyfit"
            method (ignored if using other methods). Default is 3.
        lam: float, optional
            Smoothing parameter passed to scipy.interpolate.make_smoothing_spline if
            using the "splinefit" method (ignored if using other methods). Default is
            None.
        **kwargs : dict, optional
            Additional keyword arguments to pass to hvplot.area.

        Returns
        -------
        hvplot object
            The resulting plot object.
        """
        data_var = self._process_data_var_argument(data_var)
        da_var_decomp = self.variance_decomposition(
            data_var,
            fraction=fraction,
            internal_variability_method=internal_variability_method,
            deg=deg,
            lam=lam,
        )[data_var]
        xlabel = (
            label_from_attrs(da_var_decomp.time).replace("[", "(").replace("]", ")")
        )
        ylabel = (  # Need to drop attrs to avoid issues with some versions of hvplot
            label_from_attrs(da_var_decomp)  # so first get ylabel from attrs
            .replace("[", "(")
            .replace("]", ")")
        )
        da_var_decomp = da_var_decomp.drop_attrs()
        ds_plot = xr.Dataset(
            {
                "Internal variability": da_var_decomp.sel(source="internal", drop=True),
                "Model uncertainty": da_var_decomp.sel(source="model", drop=True),
                "Scenario uncertainty": da_var_decomp.sel(source="scenario", drop=True),
            }
        ).squeeze()
        kwargs_hvplot = {
            "x": "time",
            "y": ["Internal variability", "Model uncertainty", "Scenario uncertainty"],
            "xlabel": xlabel,
            "ylabel": ylabel,
            "group_label": "Uncertainty type",
            **kwargs,
        }
        plot_obj = ds_plot.hvplot.area(**kwargs_hvplot)
        return plot_obj

    def plot_uncertainty_interval_decomposition(
        self,
        data_var=None,
        uncertainty_level=90,
        internal_variability_method=None,
        deg=3,
        lam=None,
        kwargs_baseline=None,
        **kwargs_area,
    ):
        """
        Plot contributions of climate uncertainty sources to uncertainty intervals.

        Generates a plume plot showing contributions of internal variability, model
        uncertainty and scenario uncertainty (as applicable) to uncertainty intervals for
        a data variable over time.

        Wraps hvplot.area.

        Parameters
        ----------
        data_var : str
            Name of the data variable to plot.
        uncertainty_level : float, optional
            Uncertainty level for the uncertainty intervals (percentage). Default is 90.
        internal_variability_method : str, optional
            Whether to characterize internal variability by computing ensemble
            statistics directly at each time point ("direct") or by estimating them
            using a polynomial ("polyfit") or spline ("splinefit") fit to the time
            series, assuming the variance is constant in time. By default, the "direct"
            method is used if multiple realizations are available (i.e., the dataset has
            a non-singleton "realization" dimension), and the "polyfit" method is used
            if only a single realization is available.
        deg : int, optional
            Degree of the polynomial to fit to the time series if using the "polyfit"
            method (ignored if using other methods). Default is 3.
        lam: float, optional
            Smoothing parameter passed to scipy.interpolate.make_smoothing_spline if
            using the "splinefit" method (ignored if using other methods). Default is
            None.
        kwargs_baseline : dict, optional
            Additional keyword arguments to pass to hvplot.line for the baseline
            estimate.
        **kwargs_area : dict, optional
            Additional keyword arguments to pass to hvplot.area for the all uncertainty
            interval plots.

        Returns
        -------
        hvplot object
            The resulting plot object.
        """
        data_var = self._process_data_var_argument(data_var)
        da_decomp = self.uncertainty_interval_decomposition(
            data_var,
            uncertainty_level=uncertainty_level,
            internal_variability_method=internal_variability_method,
            deg=deg,
            lam=lam,
        )[data_var]
        xlabel = label_from_attrs(da_decomp.time).replace("[", "(").replace("]", ")")
        ylabel = (  # Need to drop attrs to avoid issues with some versions of hvplot
            label_from_attrs(da_decomp)  # so first get ylabel from attrs
            .replace("[", "(")
            .replace("]", ")")
        )
        da_decomp = da_decomp.drop_attrs()
        kwargs_baseline_in = {} if kwargs_baseline is None else kwargs_baseline
        kwargs_area_in = {} if kwargs_area is None else kwargs_area
        kwargs_baseline = {
            **{
                "x": "time",
                "label": "Mean",
                "color": "black",
                "xlabel": xlabel,
                "ylabel": ylabel,
            },
            **kwargs_baseline_in,
        }
        colors = hv.Cycle().values
        kwargs_area = {
            **{
                "x": "time",
                "alpha": 0.6,
            },
            **kwargs_area_in,
        }
        kwargs_internal = {
            "label": "Internal variability",
            "color": colors[0],
            **kwargs_area,
        }
        kwargs_model = {"label": "Model uncertainty", "color": colors[1], **kwargs_area}
        kwargs_scenario = {
            "label": "Scenario uncertainty",
            "color": colors[2],
            **kwargs_area,
        }

        # Plot uncertainty intervals
        plot_obj_list = []
        if "scenario" in self._obj.dims and self._obj.scenario.size > 1:
            plot_obj_scenario_lower = xr.Dataset(
                {
                    "lower": da_decomp.sel(level="scenario_lower", drop=True),
                    "upper": da_decomp.sel(level="model_lower", drop=True),
                }
            ).hvplot.area(y="lower", y2="upper", **kwargs_scenario)
            plot_obj_scenario_upper = xr.Dataset(
                {
                    "lower": da_decomp.sel(level="model_upper", drop=True),
                    "upper": da_decomp.sel(level="scenario_upper", drop=True),
                }
            ).hvplot.area(
                y="lower", y2="upper", **{**kwargs_scenario, **{"label": None}}
            )
            plot_obj_list.extend([plot_obj_scenario_lower, plot_obj_scenario_upper])
        if "model" in self._obj.dims and self._obj.model.size > 1:
            plot_obj_model_lower = xr.Dataset(
                {
                    "lower": da_decomp.sel(level="model_lower", drop=True),
                    "upper": da_decomp.sel(level="internal_lower", drop=True),
                }
            ).hvplot.area(y="lower", y2="upper", **kwargs_model)
            plot_obj_model_upper = xr.Dataset(
                {
                    "lower": da_decomp.sel(level="internal_upper", drop=True),
                    "upper": da_decomp.sel(level="model_upper", drop=True),
                }
            ).hvplot.area(y="lower", y2="upper", **{**kwargs_model, **{"label": None}})
            plot_obj_list.extend([plot_obj_model_lower, plot_obj_model_upper])
        if ("realization" in self._obj.dims and self._obj.realization.size > 1) or (
            internal_variability_method != "direct"
        ):
            plot_obj_internal = xr.Dataset(
                {
                    "lower": da_decomp.sel(level="internal_lower", drop=True),
                    "upper": da_decomp.sel(level="internal_upper", drop=True),
                }
            ).hvplot.area(y="lower", y2="upper", **kwargs_internal)
            plot_obj_list.append(plot_obj_internal)
        # Plot the baseline estimate
        plot_obj_baseline = da_decomp.sel(level="baseline", drop=True).hvplot.line(
            **kwargs_baseline
        )
        plot_obj_list.append(plot_obj_baseline)
        # Combine the plots
        plot_obj = hv.Overlay(plot_obj_list).collate()
        return plot_obj

    def _process_data_var_argument(self, data_var_in=None, as_list=False):
        # Method for processing the data_var argument in the various methods of the
        # ClimEpiDatasetAccessor class, in order to allow for automatic specification of
        # the data variable(s) if not provided, when this is possible.
        if data_var_in is not None:
            if as_list:
                if isinstance(data_var_in, str):
                    return [data_var_in]
                if isinstance(data_var_in, list):
                    return data_var_in
                raise ValueError(
                    """The method only accepts a scalar string or list argument for the
                    data variable."""
                )
            if isinstance(data_var_in, str):
                return data_var_in
            raise ValueError(
                """The method only accepts a scalar string argument for the data
                variable."""
            )
        non_bnd_data_vars = list_non_bnd_data_vars(self._obj)
        if as_list:
            return non_bnd_data_vars
        if len(non_bnd_data_vars) == 1:
            return non_bnd_data_vars[0]
        raise ValueError(
            """Multiple data variables present. The data variable to use must be
            specified."""
        )
