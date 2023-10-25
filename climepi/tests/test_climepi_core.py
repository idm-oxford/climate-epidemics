import numpy as np
import xarray as xr
import xarray.testing as xrt

from climepi import ClimEpiDatasetAccessor  # noqa


def test_annual_mean():
    """Test the annual_mean method of the ClimEpiDatasetAccessor class."""
    time_lb = xr.cftime_range(start="2000-01-01", periods=24, freq="MS")
    time_rb = xr.cftime_range(start="2000-02-01", periods=24, freq="MS")
    time_bnds = np.array([time_lb, time_rb]).T
    temp = np.array([np.arange(1, 25), np.arange(25, 49), np.arange(49, 73)])
    ds = xr.Dataset(
        {"temp": (("hello", "time"), temp), "time_bnds": (("time", "bnds"), time_bnds)},
        coords={
            "hello": np.array(["there", "general", "kenobi"]),
            "time": xr.cftime_range(start="2000-01-01", periods=24, freq="MS"),
            "bnds": [1, 2],
        },
    )
    ds.time.attrs.update(bounds="time_bnds")
    ds_am = ds.climepi.annual_mean("temp")
    temp_am_exp = np.array(
        [
            np.sum(
                temp[:, 0:12]
                * np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
                axis=1,
            )
            / 366,
            np.sum(
                temp[:, 12:24]
                * np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
                axis=1,
            )
            / 365,
        ]
    ).T
    ds_am_exp = xr.Dataset(
        {
            "temp": (("hello", "time"), temp_am_exp),
        },
        coords={
            "hello": np.array(["there", "general", "kenobi"]),
            "time": xr.cftime_range(start="2000-01-01", periods=2, freq="YS"),
            "bnds": [1, 2],
        },
    )
    xrt.assert_allclose(ds_am, ds_am_exp)


def test_ensemble_mean():
    """Test the ensemble_mean method of the ClimEpiDatasetAccessor class."""
    temp = np.array([[1, 2, 3], [4, 5, 6]])
    ds = xr.Dataset(
        {"temp": (("realization", "hello"), temp)},
        coords={
            "realization": np.array([1, 2]),
            "hello": np.array(["there", "general", "kenobi"]),
        },
    )
    ds_mean = ds.climepi.ensemble_mean("temp")
    temp_mean_exp = np.mean(temp, axis=0)
    ds_mean_exp = xr.Dataset(
        {"temp": ("hello", temp_mean_exp)},
        coords={"hello": np.array(["there", "general", "kenobi"])},
    )
    xrt.assert_allclose(ds_mean, ds_mean_exp)


def test_ensemble_percentiles():
    """
    Test the ensemble_percentiles method of the ClimEpiDatasetAccessor
    class.
    """
    temp = np.array([[1, 2, 3], [4, 5, 6]])
    ds = xr.Dataset(
        {"temp": (("realization", "hello"), temp)},
        coords={
            "realization": np.array([1, 2]),
            "hello": np.array(["there", "general", "kenobi"]),
        },
    )
    percentiles = [0, 25, 50, 75, 100]
    ds_percentiles = ds.climepi.ensemble_percentiles("temp", percentiles)
    temp_percentiles_exp = np.percentile(temp, percentiles, axis=0).T
    ds_percentiles_exp = xr.Dataset(
        {"temp": (("hello", "percentile"), temp_percentiles_exp)},
        coords={
            "hello": np.array(["there", "general", "kenobi"]),
            "percentile": percentiles,
        },
    )
    xrt.assert_allclose(ds_percentiles, ds_percentiles_exp)


def test_ensemble_mean_std_max_min():
    """
    Test the ensemble_mean_std_max_min method of the ClimEpiDatasetAccessor
    class.
    """
    temp = np.array([[1, 2, 3], [4, 5, 6]])
    ds = xr.Dataset(
        {"temp": (("realization", "hello"), temp)},
        coords={
            "realization": np.array([1, 2]),
            "hello": np.array(["there", "general", "kenobi"]),
        },
    )
    ds_stat = ds.climepi.ensemble_mean_std_max_min("temp")
    temp_mean_exp = np.mean(temp, axis=0)
    temp_std_exp = np.std(temp, axis=0)
    temp_max_exp = np.max(temp, axis=0)
    temp_min_exp = np.min(temp, axis=0)
    temp_stat_exp = np.array(
        [temp_mean_exp, temp_std_exp, temp_max_exp, temp_min_exp]
    ).T
    ds_stat_exp = xr.Dataset(
        {"temp": (("hello", "ensemble_statistic"), temp_stat_exp)},
        coords={
            "hello": np.array(["there", "general", "kenobi"]),
            "ensemble_statistic": np.array(["mean", "std", "max", "min"]),
        },
    )
    xrt.assert_allclose(ds_stat, ds_stat_exp)


def test_ensemble_stats():
    """ "Test the ensemble_stats method of the ClimEpiDatasetAccessor class."""
    temp = np.array([[1, 2, 3], [4, 5, 6]])
    ds = xr.Dataset(
        {"temp": (("realization", "hello"), temp)},
        coords={
            "realization": np.array([1, 2]),
            "hello": np.array(["there", "general", "kenobi"]),
        },
    )
    ds_stat = ds.climepi.ensemble_stats("temp", conf_level=90)
    temp_mean_exp = np.mean(temp, axis=0)
    temp_std_exp = np.std(temp, axis=0)
    temp_max_exp = np.max(temp, axis=0)
    temp_min_exp = np.min(temp, axis=0)
    temp_lower_exp = np.percentile(temp, 5, axis=0)
    temp_median_exp = np.median(temp, axis=0)
    temp_upper_exp = np.percentile(temp, 95, axis=0)
    temp_stat_exp = np.array(
        [
            temp_mean_exp,
            temp_std_exp,
            temp_max_exp,
            temp_min_exp,
            temp_lower_exp,
            temp_median_exp,
            temp_upper_exp,
        ]
    ).T
    ds_stat_exp = xr.Dataset(
        {"temp": (("hello", "ensemble_statistic"), temp_stat_exp)},
        coords={
            "hello": np.array(["there", "general", "kenobi"]),
            "ensemble_statistic": np.array(
                ["mean", "std", "max", "min", "lower", "median", "upper"]
            ),
        },
    )
    xrt.assert_allclose(ds_stat, ds_stat_exp)


def test_copy_bnds_from():
    """Test the copy_bnds_from method of the ClimEpiDatasetAccessor class."""
    # Create a dataset to copy bounds from
    time_lb = xr.cftime_range(start="2000-01-01", periods=24, freq="MS")
    time_rb = xr.cftime_range(start="2000-02-01", periods=24, freq="MS")
    time_bnds = np.array([time_lb, time_rb]).T
    temp = np.array([np.arange(1, 25), np.arange(25, 49), np.arange(49, 73)])
    ds_from = xr.Dataset(
        {"temp": (("hello", "time"), temp), "time_bnds": (("time", "bnds"), time_bnds)},
        coords={
            "hello": np.array(["there", "general", "kenobi"]),
            "time": xr.cftime_range(start="2000-01-01", periods=24, freq="MS"),
            "bnds": [1, 2],
        },
    )
    ds_from.time.attrs.update(bounds="time_bnds")

    # Create a dataset to copy bounds to
    ds_to = xr.Dataset(
        {"temp": (("hello", "time"), temp)},
        coords={
            "hello": np.array(["there", "general", "kenobi"]),
            "time": xr.cftime_range(start="2000-01-01", periods=24, freq="MS"),
        },
    )

    # Copy the bounds
    ds_to.climepi.copy_bnds_from(ds_from)

    # Check that the bounds were copied correctly
    assert "time_bnds" in ds_to.data_vars
    assert ds_to.time.attrs["bounds"] == "time_bnds"
    assert "lon_bnds" not in ds_to.data_vars
    xrt.assert_allclose(ds_to.time_bnds, ds_from.time_bnds)
