"""Module defining functions used to characterize internal climate variability."""

from typing import Literal

import scipy.interpolate
import scipy.stats
import xarray as xr
from xarray.computation.computation import _ensure_numeric


def _ensemble_stats_direct(
    ds_in: xr.Dataset, *, uncertainty_level: float
) -> xr.Dataset:
    # Add trivial "realization" dimension if necessary
    if "realization" not in ds_in.dims:
        ds_in = ds_in.expand_dims(dim="realization")
    # Compute ensemble statistics
    ds_mean = ds_in.mean(dim="realization").expand_dims(dim={"stat": ["mean"]}, axis=-1)
    ds_std = ds_in.std(dim="realization").expand_dims(dim={"stat": ["std"]}, axis=-1)
    ds_var = ds_in.var(dim="realization").expand_dims(dim={"stat": ["var"]}, axis=-1)
    ds_quantile = ds_in.quantile(
        [0, 0.5 - uncertainty_level / 200, 0.5, 0.5 + uncertainty_level / 200, 1],
        dim="realization",
    ).rename({"quantile": "stat"})
    ds_quantile["stat"] = ["min", "lower", "median", "upper", "max"]
    ds_stat = xr.concat(
        [ds_mean, ds_std, ds_var, ds_quantile],
        dim="stat",
        coords="minimal",
    )
    return ds_stat


def _ensemble_stats_fit(
    ds_in: xr.Dataset,
    *,
    uncertainty_level: float,
    internal_variability_method: Literal["polyfit", "splinefit"],
    deg: int | None = None,
    lam: float | None = None,
) -> xr.Dataset:
    # Estimate ensemble statistics by fitting a polynomial to each time series
    # Drop trivial "realization" co-ordinate if present
    if "realization" in ds_in.dims and ds_in.realization.size == 1:
        ds_in = ds_in.squeeze("realization", drop=True)
    elif "realization" in ds_in.coords:
        ds_in = ds_in.drop_vars("realization")
    # Call appropriate method to estimate ensemble mean and variance
    if internal_variability_method == "polyfit":
        assert deg is not None
        ds_mean, ds_var = _ensemble_mean_var_polyfit(ds_in, deg=deg)
    elif internal_variability_method == "splinefit":
        ds_mean, ds_var = _ensemble_mean_var_splinefit(ds_in, lam=lam)
    else:
        raise ValueError(
            f"Unknown internal_variability_method: {internal_variability_method}"
        )
    # Calculate standard deviation and broadcast along time dimension
    ds_std = xr.ufuncs.sqrt(ds_var)
    ds_var = ds_var.broadcast_like(ds_mean)
    ds_std = ds_std.broadcast_like(ds_mean)
    # Estimate uncertainty intervals
    z = scipy.stats.norm.ppf(0.5 + uncertainty_level / 200)
    ds_lower = ds_mean - z * ds_std
    ds_upper = ds_mean + z * ds_std
    # Combine into a single dataset
    ds_stat = xr.concat(
        [ds_mean, ds_var, ds_std, ds_lower, ds_upper],
        dim=xr.Variable("stat", ["mean", "var", "std", "lower", "upper"]),
        coords="minimal",
    )
    return ds_stat


def _ensemble_mean_var_polyfit(
    ds_in: xr.Dataset, *, deg: int
) -> tuple[xr.Dataset, xr.Dataset]:
    # Estimate ensemble mean by fitting a polynomial to each time series.
    # Deal with cases where the dataset includes a realization coordinate
    if "realization" in ds_in.dims:
        return _ensemble_mean_var_polyfit_multiple_realizations(ds_in, deg=deg)
    fitted_polys = ds_in.polyfit(dim="time", deg=deg, full=True)
    data_var_list = [str(k) for k in ds_in.data_vars]
    poly_coeff_data_var_list = [x + "_polyfit_coefficients" for x in data_var_list]
    ds_mean = xr.polyval(
        coord=ds_in.time,
        coeffs=fitted_polys[poly_coeff_data_var_list],
    ).rename(dict(zip(poly_coeff_data_var_list, data_var_list, strict=True)))
    # Estimate ensemble variance/standard deviation using residuals from polynomial
    # fits (with an implicit assumption that the variance is constant in time).
    # Note that the calls to broadcast_like ensure broadcasting along the time
    # dimension (this should be equivalent to adding coords="minimal" when
    # concatenating the datasets, but is done explicitly here for clarity).
    poly_residual_data_var_list = [x + "_polyfit_residuals" for x in data_var_list]
    ds_var = (fitted_polys[poly_residual_data_var_list] / ds_in.time.size).rename(
        dict(zip(poly_residual_data_var_list, data_var_list, strict=True))
    )
    return ds_mean, ds_var


def _ensemble_mean_var_polyfit_multiple_realizations(
    ds_in: xr.Dataset, *, deg: int
) -> tuple[xr.Dataset, xr.Dataset]:
    # Wrapper to extend _ensemble_mean_var_polyfit to multiple realizations.
    # Flatten observations from different realizations
    ds_in_stacked = ds_in.stack(dim={"time_realization": ("time", "realization")})
    ds_in_flattened = (
        ds_in_stacked.swap_dims(time_realization="flattened_time")
        .drop_vars(["time"])
        .rename(flattened_time="time")
        .assign_coords(
            time=ds_in_stacked["time"].values,
        )
    )
    ds_mean_flattened, ds_var = _ensemble_mean_var_polyfit(ds_in_flattened, deg=deg)
    # ds_mean_flattened has repeated time coordinates, so need to unstack
    ds_mean_flattened = (
        ds_mean_flattened.swap_dims(time="time_realization")
        .drop_vars("time")
        .assign_coords(time_realization=ds_in_stacked["time_realization"])
    )
    ds_mean = ds_mean_flattened.unstack("time_realization").isel(
        realization=0, drop=True
    )
    return ds_mean, ds_var


def _ensemble_mean_var_splinefit(
    ds_in: xr.Dataset, lam: float | None = None
) -> tuple[xr.Dataset, xr.Dataset]:
    # Estimate ensemble mean by fitting a spline to each time series.
    # Deal with cases where the dataset includes a realization coordinate
    if "realization" in ds_in.dims:
        return _ensemble_mean_var_splinefit_multiple_realizations(ds_in, lam=lam)
    # Get vector of numeric time values
    time_vec = _ensure_numeric(ds_in.time).to_numpy()
    # Rescale time vector to [0, 1] for smoothing spline
    x_vec = (time_vec - time_vec.min()) / (time_vec.max() - time_vec.min())

    def _splinefit(data_vec):
        spl = scipy.interpolate.make_smoothing_spline(
            x_vec,
            data_vec,
            lam=lam,
        )
        return spl(x_vec)

    ds_mean = xr.apply_ufunc(
        _splinefit,
        ds_in,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        exclude_dims={"time"},
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
        dask_gufunc_kwargs={"output_sizes": {"time": ds_in.time.size}},
    )
    ds_mean = ds_mean.assign_coords(
        time=ds_in.time,
    )
    ds_var = ((ds_mean - ds_in) ** 2).mean(dim="time")
    return ds_mean, ds_var


def _ensemble_mean_var_splinefit_multiple_realizations(
    ds_in: xr.Dataset, lam: float | None = None
) -> tuple[xr.Dataset, xr.Dataset]:
    # Wrapper to extend _ensemble_mean_var_splinefit to multiple realizations.
    # scipy.interpolate.make_smoothing_spline doesn't work with non-monotonic time
    # co-ordinates, so fit spline to mean of realizations (but compute variance
    # using all realizations).
    ds_mean, _ = _ensemble_mean_var_splinefit(ds_in.mean(dim="realization"), lam=lam)
    ds_var = ((ds_mean - ds_in) ** 2).mean(dim=["time", "realization"])
    return ds_mean, ds_var
