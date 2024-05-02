"""Core module for the climepi package. This module contains the
ClimEpiDatasetAccessor class for xarray datasets.
"""

import geoviews.feature as gf
import holoviews as hv
import hvplot.xarray  # noqa # pylint: disable=unused-import
import numpy as np
import scipy.stats
import xarray as xr
import xcdat  # noqa # pylint: disable=unused-import
from geopy.geocoders import Nominatim
from xarray.plot.utils import label_from_attrs

from climepi.utils import (
    add_bnds_from_other,
    add_var_attrs_from_other,
    list_non_bnd_data_vars,
)

geolocator = Nominatim(user_agent="climepi")


@xr.register_dataset_accessor("climepi")
class ClimEpiDatasetAccessor:
    """
    Accessor class providing core methods, including for computing temporal and
    ensemble statistics, and for plotting, to xarray datasets through the ``.climepi``
    attribute.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def run_epi_model(self, epi_model, **kwargs):
        """
        Runs the epidemiological model on a climate dataset.

        Parameters:
        -----------
        epi_model : climepi.epimod.EpiModel
            The epidemiological model to run.
        **kwargs : dict, optional
            Keyword arguments to pass to the model's run method. For suitability models,
            passing "return_months_suitable=True" will return the number of months
            suitable each year, rather than the full suitability dataset, and
            additionally passing a value for "suitability_threshold" will set the
            minimum suitability threshold for a month to be considered suitable (default
            is 0).

        Returns:
        --------
        xarray.Dataset:
            The output of the model's run method.
        """
        ds_epi = epi_model.run(self._obj, **kwargs)
        return ds_epi

    def sel_geo(self, location, **kwargs):
        """
        Obtains the latitude and longitude co-ordinates of a specified location using
        geopy's Nominatim geocoder, and returns a new dataset containing the data for
        the nearest grid point.


        Parameters
        ----------
        location : str
            Name of the location to select.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the geocode method of the Nominatim
            geocoder.

        Returns
        -------
        xarray.Dataset
            A new dataset containing the data for the specified location.
        """
        if len(self._obj.lon) == 1 or len(self._obj.lat) == 1:
            print(
                "Warning: Trying to select a location from a dataset with only one",
                "longitude and/or latitude co-ordinate.",
            )
        location_geopy = geolocator.geocode(location, **kwargs)
        lat = location_geopy.latitude
        lon = location_geopy.longitude
        if max(self._obj.lon) > 180.001:
            # Deals with the case where the longitude co-ordinates are in the range
            # [0, 360] (slightly crude)
            lon = lon % 360
        if (
            lat < min(self._obj.lat)
            or lat > max(self._obj.lat)
            or lon < min(self._obj.lon)
            or lon > max(self._obj.lon)
        ):
            print(
                "Warning: The requested location is outside the range of the",
                "dataset. Returning the nearest grid point.",
            )
        ds_new = self._obj.sel(lat=lat, lon=lon, method="nearest")
        return ds_new

    def temporal_group_average(self, data_var=None, frequency="yearly", **kwargs):
        """
        Computes the group average of a data variable. Wraps xcdat
        temporal.group_average.

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
        ds_m = self._obj.temporal.group_average(data_var, freq=xcdat_freq, **kwargs)
        if ds_m.time.size > 1:
            # Add time bounds and center times (only if there is more than one time
            # point, as xcdat add_time_bounds does not work for a single time point)
            ds_m = ds_m.bounds.add_time_bounds(method="freq", freq=xcdat_freq)
            # Workaround for bug in xcdat.center_times when longitude and/or latitude
            # are non-dimension singleton coordinates (otherwise, longitude and/or
            # latitude are incorrectly treated as time coordinates, leading to an error
            # being raised)
            centered_times = xcdat.center_times(ds_m[["time", "time_bnds"]])
            ds_m["time"] = centered_times.time
            ds_m["time_bnds"] = centered_times.time_bnds
        return ds_m

    def yearly_average(self, data_var=None, **kwargs):
        """
        Computes the yearly mean of a data variable. Thin wrapper around group_average.

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
        Computes the monthly mean of a data variable. Thin wrapper around group_average.

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

    def months_suitable(self, suitability_var_name=None, suitability_threshold=0):
        """
        Calculates the number of months suitable each year from monthly suitability
        data.

        Parameters:
        -----------
        suitability_var_name : str, optional
            Name of the suitability variable to use. If not provided, the method will
            attempt to automatically select a suitable variable.
        suitability_threshold : float or int, optional
            Minimum suitability threshold for a month to be considered suitable. Default
            is 0.

        Returns:
        --------
        xarray.Dataset:
            Dataset with a single non-bound data variable "months_suitable".
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
        da_suitability = self._obj[suitability_var_name]
        ds_suitable_bool = xr.Dataset(
            {"suitable": da_suitability > suitability_threshold}
        )
        ds_suitable_bool = add_bnds_from_other(ds_suitable_bool, self._obj)
        ds_suitable_mean = ds_suitable_bool.climepi.yearly_average(weighted=False)
        ds_months_suitable = ds_suitable_mean.assign(
            months_suitable=12 * ds_suitable_mean["suitable"]
        ).drop_vars("suitable")
        ds_months_suitable.months_suitable.attrs.update(
            long_name="Months where "
            + suitability_var_name
            + " > "
            + str(suitability_threshold)
        )
        return ds_months_suitable

    def ensemble_stats(
        self,
        data_var=None,
        conf_level=90,
        estimate_internal_variability=True,
        polyfit_degree=4,
    ):
        """
        Computes a range of ensemble statistics for a data variable.

        Parameters
        ----------
        data_var : str or list, optional
            Name(s) of the data variable(s) to compute the ensemble statistics for.
            If not provided, all non-bound data variables will be used.
        conf_level : float, optional
            Confidence level for computing ensemble percentiles.
        estimate_internal_variability : bool, optional
            Whether to estimate internal variability using the estimate_ensemble_stats
            method if only a single realization is available for each model and scenario
            (ignored if multiple realizations are available). Default is True.
        polyfit_degree : int, optional
            Degree of the polynomial to fit to the time series if estimating internal
            variability. Default is 4.

        Returns
        -------
        xarray.Dataset
            A new dataset containing the computed ensemble statistics for the
            selected data variable(s).
        """
        # Process the data variable argument
        data_var_list = self._process_data_var_argument(data_var, as_list=True)
        # Deal with cases where only a single realization is available for each model
        # and scenario
        if estimate_internal_variability and not (
            "realization" in self._obj.dims and len(self._obj.realization) > 1
        ):
            return self.estimate_ensemble_stats(
                data_var_list, conf_level=conf_level, polyfit_degree=polyfit_degree
            )
        if "realization" not in self._obj[data_var_list].dims:
            ds_expanded = self._obj.copy()
            for data_var_curr in data_var_list:
                ds_expanded[data_var_curr] = ds_expanded[data_var_curr].expand_dims(
                    dim="realization"
                )
            return ds_expanded.climepi.ensemble_stats(
                data_var_list,
                conf_level=conf_level,
                estimate_internal_variability=False,
            )
        # Compute ensemble statistics
        ds_raw = self._obj[data_var_list]  # drops bounds for now (re-add at end)
        ds_mean = ds_raw.mean(dim="realization").expand_dims(
            dim={"ensemble_stat": ["mean"]}, axis=-1
        )
        ds_std = ds_raw.std(dim="realization").expand_dims(
            dim={"ensemble_stat": ["std"]}, axis=-1
        )
        ds_var = ds_raw.var(dim="realization").expand_dims(
            dim={"ensemble_stat": ["var"]}, axis=-1
        )
        ds_quantile = (
            ds_raw.chunk({"realization": -1})
            .quantile(
                [0, 0.5 - conf_level / 200, 0.5, 0.5 + conf_level / 200, 1],
                dim="realization",
            )
            .rename({"quantile": "ensemble_stat"})
        )
        ds_quantile["ensemble_stat"] = ["min", "lower", "median", "upper", "max"]
        ds_stat = xr.concat(
            [ds_mean, ds_std, ds_var, ds_quantile],
            dim="ensemble_stat",
            coords="minimal",
        )
        ds_stat.attrs = self._obj.attrs
        ds_stat = add_var_attrs_from_other(ds_stat, self._obj, var=data_var_list)
        ds_stat = add_bnds_from_other(ds_stat, self._obj)
        return ds_stat

    def estimate_ensemble_stats(self, data_var=None, conf_level=90, polyfit_degree=4):
        """
        Estimates ensemble statistics for a data variable by fitting a polynomial to
        time series for a single ensemble member.

        Parameters
        ----------
        data_var : str or list, optional
            Name(s) of the data variable(s) to estimate the ensemble statistics for.
            If not provided, all non-bound data variables will be used.
        conf_level : float, optional
            Confidence level for computing ensemble percentiles.
        polyfit_degree : int, optional
            Degree of the polynomial to fit to the time series. Default is 4.

        Returns
        -------
        xarray.Dataset
            A new dataset containing the estimated ensemble statistics for the
            selected data variable(s).
        """
        # Process the data variable argument
        data_var_list = self._process_data_var_argument(data_var, as_list=True)
        # Deal with cases where the dataset includes a realization coordinate
        if "realization" in self._obj.dims:
            if len(self._obj.realization) > 1:
                raise ValueError(
                    """The estimate_ensemble_stats method is only implemented for
                    datasets with a single ensemble member. Use the ensemble_stats
                    method instead.""",
                )
            return self._obj.squeeze(
                "realization", drop=True
            ).climepi.estimate_ensemble_stats(
                data_var_list, conf_level=conf_level, polyfit_degree=polyfit_degree
            )
        if "realization" in self._obj.coords:
            return self._obj.drop_vars("realization").climepi.estimate_ensemble_stats(
                data_var_list, conf_level=conf_level, polyfit_degree=polyfit_degree
            )
        # Estimate ensemble mean by fitting a polynomial to each time series.
        ds_raw = self._obj[data_var_list]
        fitted_polys = ds_raw.polyfit(dim="time", deg=polyfit_degree, full=True)
        poly_coeff_data_var_list = [x + "_polyfit_coefficients" for x in data_var_list]
        ds_mean = xr.polyval(
            coord=ds_raw.time,
            coeffs=fitted_polys[poly_coeff_data_var_list],
        ).rename(dict(zip(poly_coeff_data_var_list, data_var_list)))
        # Estimate ensemble variance/standard deviation using residuals from polynomial
        # fits (with an implicit assumption that the variance is constant in time).
        # Note that the calls to broadcast_like ensure broadcasting along the time
        # dimension (this should be equivalent to adding coords="minimal" when
        # concatenating the datasets, but is done explicitly here for clarity).
        poly_residual_data_var_list = [x + "_polyfit_residuals" for x in data_var_list]
        ds_var = (fitted_polys[poly_residual_data_var_list] / len(ds_raw.time)).rename(
            dict(zip(poly_residual_data_var_list, data_var_list))
        )
        ds_std = np.sqrt(ds_var)
        ds_var = ds_var.broadcast_like(ds_mean)
        ds_std = ds_std.broadcast_like(ds_mean)
        # Estimate confidence intervals
        z = scipy.stats.norm.ppf(0.5 + conf_level / 200)
        ds_lower = ds_mean - z * ds_std
        ds_upper = ds_mean + z * ds_std
        # Combine into a single dataset
        ds_stat = xr.concat(
            [ds_mean, ds_var, ds_std, ds_lower, ds_upper],
            dim=xr.Variable("ensemble_stat", ["mean", "var", "std", "lower", "upper"]),
            coords="minimal",
        )
        for coord_var in ds_raw.coords:
            # Add coordinate variables from the raw dataset that do not appear in the
            # dimensions of the data variable (e.g. if lon and lat are not direct dims)
            if (
                coord_var not in ds_stat.coords
                and coord_var != "realization"
                and "realization" not in ds_raw[coord_var].dims
            ):
                ds_stat = ds_stat.assign_coords({coord_var: ds_raw[coord_var]})
        ds_stat.attrs = self._obj.attrs
        ds_stat = add_var_attrs_from_other(ds_stat, self._obj, var=data_var_list)
        ds_stat = add_bnds_from_other(ds_stat, self._obj)
        return ds_stat

    def var_decomp(
        self,
        data_var=None,
        fraction=False,
        estimate_internal_variability=True,
        polyfit_degree=4,
    ):
        """
        Decomposes the variance of a data variable into internal, model and scenario
        uncertainty at each time point.

        Parameters
        ----------
        data_var : str
            Name of the data variable(s) to decompose.
        fraction : bool, optional
            Whether to calculate the variance contributions as fractions of the total
            variance at each time, rather than the raw variances. Default is False.
        estimate_internal_variability : bool, optional
            Whether to estimate internal variability if only a single realization is
            available for each model and scenario (ignored if multiple realizations
            are available). Default is True.
        polyfit_degree : int, optional
            Degree of the polynomial to fit to the time series if estimating internal
            variability. Default is 4.

        Returns
        -------
        xarray.Dataset
            A new dataset containing the variance decomposition of the selected data
            variable(s).
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
                return ds_expanded.climepi.var_decomp(
                    data_var_list,
                    fraction=fraction,
                    estimate_internal_variability=estimate_internal_variability,
                    polyfit_degree=polyfit_degree,
                )
        # Calculate or estimate ensemble statistics characterizing internal variability
        ds_stat = self.ensemble_stats(
            data_var_list,
            estimate_internal_variability=estimate_internal_variability,
            polyfit_degree=polyfit_degree,
        )[data_var_list]
        # Make "scenario" and "model" dimensions of ds_stat if they are not present
        # or are (singleton) non-dimension coordinates (reduces number of cases to
        # handle)
        for dim in ["scenario", "model"]:
            if dim not in ds_stat.dims:
                ds_stat = ds_stat.expand_dims(dim)
        # Calculate the internal, model and scenario contributions to the variance
        ds_var_internal = ds_stat.sel(ensemble_stat="var", drop=True).mean(
            dim=["scenario", "model"]
        )
        ds_var_model = (
            ds_stat.sel(ensemble_stat="mean", drop=True)
            .var(dim="model")
            .mean(dim="scenario")
        )
        ds_var_scenario = (
            ds_stat.sel(ensemble_stat="mean", drop=True)
            .mean(dim="model")
            .var(dim="scenario")
        )
        ds_var_decomp = xr.concat(
            [ds_var_internal, ds_var_model, ds_var_scenario],
            dim=xr.Variable("var_type", ["internal", "model", "scenario"]),
            coords="minimal",
        )
        # Express contributions as a fraction of the total variance if required
        if fraction:
            ds_var_decomp = ds_var_decomp / ds_var_decomp.sum(dim="var_type")
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

    def plot_time_series(self, data_var=None, **kwargs):
        """
        Generates a time series plot of a data variable. Wraps hvplot.line.

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
        kwargs_hvplot = {"x": "time"}
        kwargs_hvplot.update(kwargs)
        plot_obj = da_plot.hvplot.line(**kwargs_hvplot)
        return plot_obj

    def plot_map(self, data_var=None, include_ocean=False, **kwargs):
        """
        Generates a map plot of a data variable. Wraps hvplot.quadmesh.

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
        }
        kwargs_hvplot.update(kwargs)
        plot_obj = da_plot.hvplot.quadmesh(**kwargs_hvplot)
        if not include_ocean:
            plot_obj *= gf.ocean.options(fill_color="white")
        return plot_obj

    def plot_var_decomp(
        self,
        data_var=None,
        fraction=False,
        estimate_internal_variability=True,
        polyfit_degree=4,
        **kwargs,
    ):
        """
        Plots the contributions of internal, model and scenario uncertainty to the total
        variance of a data variable over time. Wraps hvplot.area.

        Parameters
        ----------
        data_var : str
            Name of the data variable to plot.
        fraction : bool, optional
            Whether to plot the variance contributions as fractions of the total
            variance at each time, rather than the raw variances. Default is False.
        estimate_internal_variability : bool, optional
            Whether to estimate internal variability if only a single realization is
            available for each model and scenario (ignored if multiple realizations
            are available). Default is True.
        polyfit_degree : int, optional
            Degree of the polynomial to fit to the time series if estimating internal
            variability. Default is 4.
        **kwargs : dict, optional
            Additional keyword arguments to pass to hvplot.area.

        Returns
        -------
        hvplot object
            The resulting plot object.
        """
        data_var = self._process_data_var_argument(data_var)
        ds_var_decomp = self.var_decomp(
            data_var,
            fraction=fraction,
            estimate_internal_variability=estimate_internal_variability,
            polyfit_degree=polyfit_degree,
        )
        ds_plot = xr.Dataset(
            {
                "Internal": ds_var_decomp[data_var].sel(var_type="internal", drop=True),
                "Model": ds_var_decomp[data_var].sel(var_type="model", drop=True),
                "Scenario": ds_var_decomp[data_var].sel(var_type="scenario", drop=True),
            }
        ).squeeze()
        kwargs_hvplot = {
            "x": "time",
            "y": ["Internal", "Model", "Scenario"],
            "ylabel": label_from_attrs(ds_var_decomp[data_var])
            .replace("[", "(")
            .replace("]", ")"),
            "group_label": "Uncertainty type",
        }
        kwargs_hvplot.update(kwargs)
        plot_obj = ds_plot.hvplot.area(**kwargs_hvplot)
        return plot_obj

    def plot_ci_plume(
        self,
        data_var=None,
        conf_level=90,
        estimate_internal_variability=True,
        polyfit_degree=4,
        kwargs_baseline=None,
        **kwargs_area,
    ):
        """
        Generates a plume plot showing contributions of internal, model and scenario
        uncertainty (as applicable) to confidence intervals for a data variable over
        time. Wraps hvplot.area.

        Parameters
        ----------
        data_var : str
            Name of the data variable to plot.
        conf_level : float, optional
            Confidence level for the confidence intervals. Default is 90.
        estimate_internal_variability : bool, optional
            Whether to estimate internal variability if only a single ensemble member is
            available for each model and realization. Default is True.
        polyfit_degree : int, optional
            Degree of the polynomial to fit to the time series if estimating internal
            variability. Default is 4.
        kwargs_baseline : dict, optional
            Additional keyword arguments to pass to hvplot.line for the baseline
            estimate.
        **kwargs_area : dict, optional
            Additional keyword arguments to pass to hvplot.area for the all confidence
            interval plots.

        Returns
        -------
        hvplot object
            The resulting plot object.
        """

        kwargs_baseline_in = {} if kwargs_baseline is None else kwargs_baseline
        kwargs_area_in = {} if kwargs_area is None else kwargs_area
        kwargs_baseline = {
            **{"x": "time", "label": "Mean", "color": "black"},
            **kwargs_baseline_in,
        }
        colors = hv.Cycle().values
        kwargs_area = {**{"x": "time", "alpha": 0.6}, **kwargs_area_in}
        kwargs_internal = {
            "label": "Internal variability",
            "color": colors[0],
            **kwargs_area,
        }
        kwargs_model = {"label": "Model spread", "color": colors[1], **kwargs_area}
        kwargs_scenario = {
            "label": "Scenario spread",
            "color": colors[2],
            **kwargs_area,
        }
        data_var = self._process_data_var_argument(data_var)
        if isinstance(data_var, list):
            # Avoid bug with np.sqrt for an xarray Dataset with a single data variable
            # (this ensures DataArrays are used instead when necessary)
            data_var = data_var[0]
        da_raw = self._obj[data_var].squeeze()
        # Make "scenario", "model" and "realization" dimensions of the data variable if
        # they are not present or are (singleton) non-dimension coordinates (reduces
        # number of cases to handle; note this partially reverses the effect of the
        # squeeze operation above, which still removes other singleton dimensions).
        for dim in ["scenario", "model", "realization"]:
            if dim not in da_raw.dims:
                da_raw = da_raw.expand_dims(dim)
        # Get ensemble statistics, baseline estimate, and if necessary a decomposition
        # of the variance and z value for approximate confidence intervals
        da_stat = da_raw.to_dataset().climepi.ensemble_stats(
            data_var,
            conf_level=conf_level,
            estimate_internal_variability=estimate_internal_variability,
            polyfit_degree=polyfit_degree,
        )[data_var]
        da_baseline = da_stat.sel(ensemble_stat="mean", drop=True).mean(
            dim=["scenario", "model"], keep_attrs=True
        )
        da_var_decomp = self.var_decomp(
            data_var,
            fraction=False,
            estimate_internal_variability=estimate_internal_variability,
            polyfit_degree=polyfit_degree,
        )[data_var].squeeze()
        z = scipy.stats.norm.ppf(0.5 + conf_level / 200)
        # Create a dataset for the confidence interval plots
        ds_plume = xr.Dataset()
        multiple_realizations = len(da_raw.realization) > 1
        if estimate_internal_variability or multiple_realizations:
            # Obtain confidence interval contribution from internal variability if there are
            # multiple realizations or if internal variability is to be estimated
            if len(da_raw.scenario) == 1 and len(da_raw.model) == 1:
                ds_plume["internal_lower"] = da_stat.squeeze(
                    ["model", "scenario"], drop=True
                ).sel(ensemble_stat="lower", drop=True)
                ds_plume["internal_upper"] = da_stat.squeeze(
                    ["model", "scenario"], drop=True
                ).sel(ensemble_stat="upper", drop=True)
            else:
                da_std_internal = np.sqrt(
                    da_var_decomp.sel(var_type="internal", drop=True)
                )
                ds_plume["internal_lower"] = da_baseline - z * da_std_internal
                ds_plume["internal_upper"] = da_baseline + z * da_std_internal
            ds_plume["internal_lower"].attrs = da_baseline.attrs
            ds_plume["internal_lower"].attrs = da_baseline.attrs
        else:
            ds_plume["internal_lower"] = da_baseline
            ds_plume["internal_upper"] = da_baseline
        if len(da_raw.model) > 1:
            # Plot model variability if there are multiple models
            if len(da_raw.scenario) == 1 and not (
                multiple_realizations or estimate_internal_variability
            ):
                da_raw_rechunked = da_raw.squeeze(
                    ["scenario", "realization"], drop=True
                ).chunk({"model": -1})
                ds_plume["model_lower"] = da_raw_rechunked.quantile(
                    0.5 - conf_level / 200, dim="model"
                ).drop("quantile")
                ds_plume["model_upper"] = da_raw_rechunked.quantile(
                    0.5 + conf_level / 200, dim="model"
                ).drop("quantile")
            else:
                da_std_internal_model = np.sqrt(
                    da_var_decomp.sel(var_type=["internal", "model"]).sum(
                        dim="var_type"
                    )
                )
                ds_plume["model_lower"] = da_baseline - z * da_std_internal_model
                ds_plume["model_upper"] = da_baseline + z * da_std_internal_model
            ds_plume["model_lower"].attrs = da_baseline.attrs
            ds_plume["model_lower"].attrs = da_baseline.attrs
        else:
            ds_plume["model_lower"] = ds_plume["internal_lower"]
            ds_plume["model_upper"] = ds_plume["internal_upper"]
        if len(da_raw.scenario) > 1:
            # Plot scenario variability if there are multiple scenarios
            if len(da_raw.model) == 1 and not (
                multiple_realizations or estimate_internal_variability
            ):
                da_raw_rechunked = da_raw.squeeze(
                    ["model", "realization"], drop=True
                ).chunk({"scenario": -1})
                ds_plume["scenario_lower"] = da_raw_rechunked.quantile(
                    0.5 - conf_level / 200, dim="scenario"
                ).drop("quantile")
                ds_plume["scenario_upper"] = da_raw_rechunked.quantile(
                    0.5 + conf_level / 200, dim="scenario"
                ).drop("quantile")
            else:
                da_std_internal_model_scenario = np.sqrt(
                    da_var_decomp.sum(dim="var_type")
                )
                ds_plume["scenario_lower"] = (
                    da_baseline - z * da_std_internal_model_scenario
                )
                ds_plume["scenario_upper"] = (
                    da_baseline + z * da_std_internal_model_scenario
                )
            ds_plume["scenario_lower"].attrs = da_baseline.attrs
            ds_plume["scenario_lower"].attrs = da_baseline.attrs
        # Plot confidence intervals
        plot_obj_list = []
        if len(da_raw.scenario) > 1:
            plot_obj_scenario_lower = ds_plume[
                ["scenario_lower", "model_lower"]
            ].hvplot.area(y="scenario_lower", y2="model_lower", **kwargs_scenario)
            plot_obj_scenario_upper = ds_plume[
                ["model_upper", "scenario_upper"]
            ].hvplot.area(
                y="model_upper",
                y2="scenario_upper",
                **{**kwargs_scenario, **{"label": None}},
            )
            plot_obj_list.extend([plot_obj_scenario_lower, plot_obj_scenario_upper])
        if len(da_raw.model) > 1:
            plot_obj_model_lower = ds_plume[
                ["model_lower", "internal_lower"]
            ].hvplot.area(y="model_lower", y2="internal_lower", **kwargs_model)
            plot_obj_model_upper = ds_plume[
                ["internal_upper", "model_upper"]
            ].hvplot.area(
                y="internal_upper",
                y2="model_upper",
                **{**kwargs_model, **{"label": None}},
            )
            plot_obj_list.extend([plot_obj_model_lower, plot_obj_model_upper])
        if estimate_internal_variability or multiple_realizations:
            plot_obj_internal = ds_plume[
                ["internal_lower", "internal_upper"]
            ].hvplot.area(y="internal_lower", y2="internal_upper", **kwargs_internal)
            plot_obj_list.append(plot_obj_internal)
        # Plot the baseline estimate
        plot_obj_baseline = da_baseline.hvplot.line(**kwargs_baseline)
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
