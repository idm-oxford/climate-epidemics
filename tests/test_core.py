"""
Unit tests for the _core module of the climepi package.

The ClimEpiDatasetAccessor class is tested in this module.
"""

import cftime
import geoviews
import hvplot.xarray  # noqa
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt
from holoviews.element.comparison import Comparison as hvt
from scipy.stats import norm

from climepi import ClimEpiDatasetAccessor, epimod
from climepi.testing.fixtures import generate_dataset


def test__init__():
    """Test the __init__ method of the ClimEpiDatasetAccessor class."""
    ds = generate_dataset()
    accessor = ClimEpiDatasetAccessor(ds)
    xrt.assert_identical(accessor._obj, ds)


def test_run_epi_model():
    """
    Test the run_epi_model method of the ClimEpiDatasetAccessor class.

    Only test that the method returns the same result as the run method of the EpiModel
    class. The run method of the EpiModel class is tested in the test module for the
    epimod subpackage.
    """
    ds = generate_dataset()
    epi_model = epimod.SuitabilityModel(temperature_range=(15, 20))
    result = ds.climepi.run_epi_model(epi_model)
    expected = epi_model.run(ds)
    xrt.assert_identical(result, expected)


class TestSelGeo:
    """Class for tests of the sel_geo method of ClimEpiDatasetAccessor."""

    @pytest.mark.parametrize("location", ["Miami", "Cape Town"])
    @pytest.mark.parametrize("lon_0_360", [False, True])
    def test_sel_geo_main(self, location, lon_0_360):
        """
        Main test.

        Checks cases where the location's longitude is outside the dataset's longitude
        grid, in which case the closer of the left/right edges of the grid should be
        selected.
        """
        ds1 = xr.Dataset(
            {
                "hello": (("lat", "lon", "there"), np.random.rand(9, 72, 2)),
            },
            coords={"lat": np.arange(0, 90, 10), "lon": np.arange(-180, 180, 5)},
        )
        ds2 = ds1.sel(lon=slice(160, 180))
        ds3 = ds1.sel(lon=slice(-180, -160))
        if location == "Miami":
            # True lat = 25.8, lon = -80.2
            lat_expected = 30
            lon_expected1 = -80
            lon_expected2 = 175
            lon_expected3 = -160
        elif location == "Cape Town":
            # True lat = -33.9, lon = 18.4
            lat_expected = 0
            lon_expected1 = 20
            lon_expected2 = 160
            lon_expected3 = -180
        for ds, lon_expected in zip(
            [ds1, ds2, ds3], [lon_expected1, lon_expected2, lon_expected3], strict=True
        ):
            if lon_0_360:
                ds["lon"] = np.sort(ds["lon"].values % 360)
                lon_expected = lon_expected % 360
            result = ds.climepi.sel_geo(location=location)
            npt.assert_equal(result.lat.values, lat_expected)
            npt.assert_equal(result.lon.values, lon_expected)
            npt.assert_equal(result["location"].values, location)
            npt.assert_equal(
                result["hello"].values,
                ds["hello"].sel(lon=lon_expected, lat=lat_expected).values,
            )

    @pytest.mark.parametrize("location_list", [["Miami", "Cape Town"], ["Miami"]])
    def test_sel_geo_location_list(self, location_list):
        """Test with a list of locations."""
        ds = (
            generate_dataset(data_var="beamer", extra_dims={"covers": 2})
            .drop_vars(("lat_bnds", "lon_bnds"))
            .assign_coords(
                lat=np.array([-90, -30, 30, 90]), lon=np.array([90, 120, 150, 180])
            )
        )
        ds["beamer"].values = np.random.rand(*ds["beamer"].shape)
        result = ds.climepi.sel_geo(location=location_list)
        assert "location" in result.dims
        npt.assert_equal(result["location"].values, location_list)
        assert "location" not in result["time_bnds"].dims
        xrt.assert_identical(result["time_bnds"], ds["time_bnds"])
        for location in location_list:
            if location == "Miami":
                # True lat = 25.8, lon = -80.2
                lat_expected = 30
                lon_expected = 180
            elif location == "Cape Town":
                # True lat = -33.9, lon = 18.4
                lat_expected = -30
                lon_expected = 90
            xrt.assert_identical(
                result.sel(location=location, drop=True),
                ds.sel(lat=lat_expected, lon=lon_expected, drop=True),
            )


@pytest.mark.parametrize("frequency", ["yearly", "monthly", "daily"])
class TestTemporalGroupAverage:
    """Class for testing the temporal_group_average method of ClimEpiDatasetAccessor."""

    def test_temporal_group_average(self, frequency):
        """
        Main test.

        Focuses on the centering of the time values (which is added to the underlying
        xcdat temporal.group_average method).
        """
        time_lb = xr.cftime_range(start="2001-01-01", periods=365, freq="D")
        time_rb = xr.cftime_range(start="2001-01-02", periods=365, freq="D")
        time_bnds = xr.DataArray(np.array([time_lb, time_rb]).T, dims=("time", "bnds"))
        time = time_bnds.mean(dim="bnds")
        temperature_values_in = np.arange(365)
        ds = xr.Dataset(
            {
                "temperature": (("time"), temperature_values_in),
                "time_bnds": time_bnds,
            },
            coords={"time": time},
        )
        ds.time.attrs.update(bounds="time_bnds")
        ds["time"].encoding.update(calendar="standard")
        result = ds.climepi.temporal_group_average(frequency=frequency)
        time_index_result = result.get_index("time")
        temperature_values_result = result.temperature.values
        if frequency == "yearly":
            # Note no centering is performed when the time-averaged data has a single
            # time value
            temperature_values_expected = np.array([np.mean(temperature_values_in)])
            time_index_expected = xr.cftime_range(start="2001-01-01", periods=1)
        elif frequency == "monthly":
            temperature_values_expected = np.array(
                [
                    np.mean(
                        temperature_values_in[month_start : month_start + month_length]
                    )
                    for month_start, month_length in zip(
                        np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]),
                        np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
                        strict=True,
                    )
                ]
            )
            ds_time_bnds_expected = xr.Dataset(
                {
                    "time_bnds": (
                        ("time", "bnds"),
                        np.array(
                            [
                                xr.cftime_range(
                                    start="2001-01-01", periods=12, freq="MS"
                                ),
                                xr.cftime_range(
                                    start="2001-02-01", periods=12, freq="MS"
                                ),
                            ]
                        ).T,
                    ),
                },
            )
            ds_time_bnds_expected["time"] = ds_time_bnds_expected["time_bnds"].mean(
                dim="bnds"
            )
            time_index_expected = ds_time_bnds_expected.get_index("time")
        elif frequency == "daily":
            temperature_values_expected = temperature_values_in
            ds_time_bnds_expected = ds[["time_bnds"]]
            time_index_expected = ds.get_index("time")
        npt.assert_allclose(temperature_values_result, temperature_values_expected)
        assert time_index_result.equals(time_index_expected)
        if frequency in ["monthly", "daily"]:
            xrt.assert_allclose(
                result["time_bnds"],
                ds_time_bnds_expected["time_bnds"],
            )
        else:
            assert "time_bnds" not in result

    def test_temporal_group_average_varlist(self, frequency):
        """Test with a list of data variables."""
        data_vars = ["temperature", "precipitation"]
        ds = generate_dataset(data_var=data_vars)
        result = ds.climepi.temporal_group_average(frequency=frequency)
        for data_var in data_vars:
            expected = ds[[data_var, "time_bnds"]].climepi.temporal_group_average(
                frequency=frequency
            )
            xrt.assert_identical(result[data_var], expected[data_var])

    def test_temporal_group_average_datatypes(self, frequency):
        """Test with different data types."""
        ds_bool = generate_dataset(data_var="temperature", dtype=bool)
        ds_int = ds_bool.copy()
        ds_int["temperature"] = ds_int["temperature"].astype(int)
        ds_float = ds_int.copy()
        ds_float["temperature"] = ds_float["temperature"].astype(float)
        result_bool = ds_bool.climepi.temporal_group_average(frequency=frequency)
        result_int = ds_int.climepi.temporal_group_average(frequency=frequency)
        result_float = ds_float.climepi.temporal_group_average(frequency=frequency)
        xrt.assert_identical(result_bool, result_int)
        xrt.assert_identical(result_bool, result_float)


def test_yearly_average():
    """
    Test the yearly_average method of the ClimEpiDatasetAccessor class.

    Since this method is a thin wrapper around the temporal_group_average method, only
    test that this method returns the same result as calling temporal_group_average
    directly.
    """
    ds = generate_dataset()
    result = ds.climepi.yearly_average()
    expected = ds.climepi.temporal_group_average(frequency="yearly")
    xrt.assert_identical(result, expected)


def test_monthly_average():
    """
    Test the monthly_average method of the ClimEpiDatasetAccessor class.

    Since this method is a thin wrapper around the temporal_group_average method, only
    test that this method returns the same result as calling temporal_group_average
    directly.
    """
    ds = generate_dataset()
    result = ds.climepi.monthly_average()
    expected = ds.climepi.temporal_group_average(frequency="monthly")
    xrt.assert_identical(result, expected)


class TestMonthsSuitable:
    """Class for testing the months_suitable method of ClimEpiDatasetAccessor."""

    def test_months_suitable(self):
        """Main test."""
        time_lb = xr.cftime_range(start="2001-01-01", periods=24, freq="MS")
        time_rb = xr.cftime_range(start="2001-02-01", periods=24, freq="MS")
        time_bnds = xr.DataArray(np.array([time_lb, time_rb]).T, dims=("time", "bnds"))
        time = time_bnds.mean(dim="bnds")
        suitability_values_in = np.random.rand(24, 2)
        ds = xr.Dataset(
            {
                "suitability": (("time", "kenobi"), suitability_values_in),
                "time_bnds": time_bnds,
            },
            coords={"time": time},
        )
        ds.time.attrs.update(bounds="time_bnds")
        ds["time"].encoding.update(calendar="standard")
        suitability_threshold = 0.5
        result = ds.climepi.months_suitable(suitability_threshold=suitability_threshold)
        months_suitable_values_result = result.months_suitable.values
        months_suitable_values_expected = np.array(
            [
                np.sum(suitability_values_in[:12, :] > suitability_threshold, axis=0),
                np.sum(suitability_values_in[12:, :] > suitability_threshold, axis=0),
            ]
        )
        npt.assert_allclose(
            months_suitable_values_result, months_suitable_values_expected
        )
        assert (
            result.months_suitable.attrs["long_name"]
            == "Months where suitability > 0.5"
        )

    def test_months_suitable_var_names(self):
        """Test with different data variable names present in the dataset."""
        data_vars = ["suitability", "also_suitability", "temperature"]
        ds = generate_dataset(data_var=data_vars)
        ds["suitability"].values = np.random.rand(*ds["suitability"].shape)
        ds["also_suitability"].values = ds["suitability"].values
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        suitability_threshold = 0.2
        result1 = ds.climepi.months_suitable(
            suitability_threshold=suitability_threshold
        )
        result2 = ds.climepi.months_suitable(
            suitability_threshold=suitability_threshold,
            suitability_var_name="also_suitability",
        )
        xrt.assert_allclose(result1["months_suitable"], result2["months_suitable"])
        result3 = ds[["also_suitability", "time_bnds"]].climepi.months_suitable(
            suitability_threshold=suitability_threshold,
        )
        xrt.assert_allclose(result1["months_suitable"], result3["months_suitable"])
        result4 = ds.climepi.months_suitable(
            suitability_threshold=suitability_threshold,
            suitability_var_name="temperature",
        )
        with pytest.raises(AssertionError):
            xrt.assert_allclose(result1["months_suitable"], result4["months_suitable"])
        with pytest.raises(ValueError):
            ds[
                ["also_suitability", "temperature", "time_bnds"]
            ].climepi.months_suitable(suitability_threshold=suitability_threshold)


class TestEnsembleStats:
    """Class for testing the ensemble_stats method of ClimEpiDatasetAccessor."""

    def test_ensemble_stats(self):
        """
        Main test.

        Since the method requires rechunking of a dask-backed dataset along the
        realization dimension, test that the method works correctly with both chunked
        and non-chunked datasets.
        """
        ds = generate_dataset(
            data_var="temperature", extra_dims={"realization": 12, "ouch": 4}
        )
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        result = ds.climepi.ensemble_stats(uncertainty_level=60)
        xrt.assert_allclose(
            result["temperature"].sel(stat="mean", drop=True),
            ds["temperature"].mean(dim="realization"),
        )
        xrt.assert_allclose(
            result["temperature"].sel(stat="std", drop=True),
            ds["temperature"].std(dim="realization"),
        )
        xrt.assert_allclose(
            result["temperature"].sel(stat="var", drop=True),
            ds["temperature"].var(dim="realization"),
        )
        xrt.assert_allclose(
            result["temperature"].sel(stat="median", drop=True),
            ds["temperature"].median(dim="realization"),
        )
        xrt.assert_allclose(
            result["temperature"].sel(stat="min", drop=True),
            ds["temperature"].min(dim="realization"),
        )
        xrt.assert_allclose(
            result["temperature"].sel(stat="max", drop=True),
            ds["temperature"].max(dim="realization"),
        )
        xrt.assert_allclose(
            result["temperature"].sel(stat="lower", drop=True),
            ds["temperature"].quantile(0.2, dim="realization").drop_vars("quantile"),
        )
        xrt.assert_allclose(
            result["temperature"].sel(stat="upper", drop=True),
            ds["temperature"].quantile(0.8, dim="realization").drop_vars("quantile"),
        )
        xrt.assert_allclose(
            result[["lon", "lat", "time", "lon_bnds", "lat_bnds", "time_bnds"]],
            ds[["lon", "lat", "time", "lon_bnds", "lat_bnds", "time_bnds"]],
        )

    def test_ensemble_stats_varlist(self):
        """Test with a list of data variables."""
        data_vars = ["temperature", "precipitation"]
        ds = generate_dataset(data_var=data_vars, extra_dims={"realization": 3})
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        ds["precipitation"].values = np.random.rand(*ds["precipitation"].shape)
        result = ds.climepi.ensemble_stats()
        for data_var in data_vars:
            xrt.assert_allclose(
                result[data_var],
                ds[[data_var]].climepi.ensemble_stats()[data_var],
            )
            xrt.assert_allclose(
                result[data_var],
                ds.climepi.ensemble_stats(data_var)[data_var],
            )

    def test_ensemble_stats_single_realization(self):
        """
        Test with a single realization.

        Uses the option to estimate internal variability (enabled by default; only
        tests that this gives the same result as the estimate_ensemble_stats method,
        which is tested separately).
        """
        ds1 = generate_dataset(data_var="temperature")
        ds1["temperature"].values = np.random.rand(*ds1["temperature"].shape)
        ds2 = ds1.copy()
        ds2["temperature"] = ds2["temperature"].expand_dims("realization")
        ds3 = ds1.copy()
        ds3["realization"] = "googly"
        ds3 = ds3.set_coords("realization")
        result1 = ds1.climepi.ensemble_stats()
        result2 = ds2.climepi.ensemble_stats()
        result3 = ds3.climepi.ensemble_stats()
        expected = ds1.climepi.estimate_ensemble_stats()
        xrt.assert_allclose(result1, expected)
        xrt.assert_allclose(result2, expected)
        xrt.assert_allclose(result3, expected)

    def test_ensemble_stats_single_realization_no_estimation(self):
        """Test with a single realization without estimating internal variability."""
        ds1 = generate_dataset(data_var="temperature")
        ds1["temperature"].values = np.random.rand(*ds1["temperature"].shape)
        ds2 = ds1.copy()
        ds2["temperature"] = ds2["temperature"].expand_dims("realization")
        ds3 = ds1.copy()
        ds3["realization"] = "googly"
        ds3 = ds3.set_coords("realization")
        result1 = ds1.climepi.ensemble_stats(estimate_internal_variability=False)
        result2 = ds2.climepi.ensemble_stats(estimate_internal_variability=False)
        result3 = ds3.climepi.ensemble_stats(estimate_internal_variability=False)
        xrt.assert_allclose(result1, result2)
        xrt.assert_allclose(result1, result3)
        for stat in ["mean", "median", "min", "max", "lower", "upper"]:
            xrt.assert_allclose(
                result1["temperature"].sel(stat=stat, drop=True),
                ds1["temperature"],
            )
        for stat in ["std", "var"]:
            npt.assert_allclose(
                result1["temperature"].sel(stat=stat, drop=True).values,
                0,
            )


class TestEstimateEnsembleStats:
    """Class for testing the estimate_ensemble_stats method of ClimEpiDatasetAccessor."""

    def test_estimate_ensemble_stats(self):
        """
        Main test.

        This test is based on estimating ensemble stats from a temperature time series
        made up of normally distributed noise added to a polynomial (matching the
        underlying assumptions of the estimate_ensemble_stats method). This is repeated
        multiple times to ensure there is no systematic bias in the estimated ensemble
        statistics.
        """
        time = xr.cftime_range(start="2001-01-01", periods=10000, freq="MS")
        days_from_start = cftime.date2num(time, "days since 2001-01-01")
        mean_theoretical = (
            0.0000000000123 * days_from_start**3
            - 0.00000257 * days_from_start**2
            - 0.326 * days_from_start
            - 259.29
        )
        std_theoretical = 0.734
        repeats = 100
        mean_result_sum = np.zeros_like(mean_theoretical)
        std_result_sum = np.zeros_like(mean_theoretical)
        for repeat in range(repeats):
            temperature_values_in = np.random.normal(
                loc=mean_theoretical, scale=std_theoretical
            )
            ds = xr.Dataset(
                {
                    "temperature": ("time", temperature_values_in),
                },
                coords={"time": time},
            )
            ds["time"].encoding.update(calendar="standard")
            result = ds.climepi.estimate_ensemble_stats(
                uncertainty_level=80, polyfit_degree=3
            )
            if repeat == 0:
                # Just check for the first repeat that the results match those obtained
                # by directly applying numpy's polynomial fitting method.
                polyfit_for_expected_values = np.polynomial.Polynomial.fit(
                    days_from_start, temperature_values_in, 3, full=True
                )
                mean_expected = polyfit_for_expected_values[0](days_from_start)
                var_expected = polyfit_for_expected_values[1][0][0] / len(
                    days_from_start
                )
                std_expected = var_expected**0.5
                lower_expected = norm.ppf(0.1, loc=mean_expected, scale=std_expected)
                upper_expected = norm.ppf(0.9, loc=mean_expected, scale=std_expected)
                npt.assert_allclose(
                    result["temperature"].sel(stat="mean", drop=True).values,
                    mean_expected,
                )
                npt.assert_allclose(
                    result["temperature"].sel(stat="std", drop=True).values,
                    std_expected,
                )
                npt.assert_allclose(
                    result["temperature"].sel(stat="var", drop=True).values,
                    var_expected,
                )
                npt.assert_allclose(
                    result["temperature"].sel(stat="lower", drop=True).values,
                    lower_expected,
                )
                npt.assert_allclose(
                    result["temperature"].sel(stat="upper", drop=True).values,
                    upper_expected,
                )
            mean_result_sum += result["temperature"].sel(stat="mean", drop=True).values
            std_result_sum += result["temperature"].sel(stat="std", drop=True).values
        mean_result_avg = mean_result_sum / repeats
        std_result_avg = std_result_sum / repeats
        var_result_avg = std_result_avg**2
        lower_result_avg = norm.ppf(0.1, loc=mean_result_avg, scale=std_result_avg)
        upper_result_avg = norm.ppf(0.9, loc=mean_result_avg, scale=std_result_avg)
        var_theoretical = std_theoretical**2
        lower_theoretical = norm.ppf(0.1, loc=mean_theoretical, scale=std_theoretical)
        upper_theoretical = norm.ppf(0.9, loc=mean_theoretical, scale=std_theoretical)
        rtol_theoretical_match = 1e-2
        npt.assert_allclose(
            mean_result_avg,
            mean_theoretical,
            rtol=rtol_theoretical_match,
        )
        npt.assert_allclose(
            std_result_avg,
            std_theoretical,
            rtol=rtol_theoretical_match,
        )
        npt.assert_allclose(
            var_result_avg,
            var_theoretical,
            rtol=rtol_theoretical_match,
        )
        npt.assert_allclose(
            lower_result_avg,
            lower_theoretical,
            rtol=rtol_theoretical_match,
        )
        npt.assert_allclose(
            upper_result_avg,
            upper_theoretical,
            rtol=rtol_theoretical_match,
        )

    def test_estimate_ensemble_stats_varlist(self):
        """Test with a list of data variables."""
        data_vars = ["temperature", "precipitation"]
        ds = generate_dataset(data_var=data_vars, frequency="monthly")
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        ds["precipitation"].values = np.random.rand(*ds["precipitation"].shape)
        result = ds.climepi.ensemble_stats()
        for data_var in data_vars:
            xrt.assert_allclose(
                result[data_var],
                ds[[data_var]].climepi.ensemble_stats()[data_var],
            )
            xrt.assert_allclose(
                result[data_var],
                ds.climepi.ensemble_stats(data_var)[data_var],
            )

    def test_estimate_ensemble_stats_contains_realization(self):
        """
        Test with a dataset containing a realization dimension.

        Checks the method gives the expected result when the dataset contains a
        realization dimension with length 1, or a non-dimensional realization
        coordinate, and raises an error when the realization dimension has length
        greater than 1.
        """
        ds_base = generate_dataset(data_var="temperature", frequency="monthly")
        ds_base["temperature"].values = np.random.rand(*ds_base["temperature"].shape)
        ds1 = ds_base.copy()
        ds1["temperature"] = ds1["temperature"].expand_dims("realization")
        ds2 = ds_base.copy()
        ds2["realization"] = "googly"
        ds2 = ds2.set_coords("realization")
        result1 = ds1.climepi.ensemble_stats()
        result2 = ds2.climepi.ensemble_stats()
        expected = ds_base.climepi.estimate_ensemble_stats()
        xrt.assert_allclose(result1, expected)
        xrt.assert_allclose(result2, expected)
        ds3 = generate_dataset(extra_dims={"realization": 3})
        ds3["temperature"].values = np.random.rand(*ds3["temperature"].shape)
        with pytest.raises(ValueError):
            ds3.climepi.estimate_ensemble_stats()


class TestVarianceDecomposition:
    """Class for testing the variance_decomposition method of ClimEpiDatasetAccessor."""

    def test_variance_decomposition(self):
        """Main test."""
        ds = generate_dataset(
            data_var="temperature",
            extra_dims={"scenario": 6, "model": 4, "realization": 9},
        )
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        ds["temperature"].attrs.update(
            units="Hello there", long_name="General Kenobi you are a bold one"
        )
        result = ds.climepi.variance_decomposition()
        result_fractional = ds.climepi.variance_decomposition(fraction=True)
        temperature_values = (
            ds["temperature"]
            .transpose("scenario", "model", "realization", "time", ...)
            .values
        )
        var_total = np.var(temperature_values, axis=(0, 1, 2))
        var_internal = np.mean(np.var(temperature_values, axis=2), axis=(0, 1))
        var_model = np.mean(np.var(np.mean(temperature_values, axis=2), axis=1), axis=0)
        var_scenario = np.var(np.mean(temperature_values, axis=(1, 2)), axis=0)
        var_frac_internal = var_internal / var_total
        var_frac_model = var_model / var_total
        var_frac_scenario = var_scenario / var_total
        npt.assert_allclose(
            result["temperature"].sel(source="internal", drop=True).values,
            var_internal,
        )
        npt.assert_allclose(
            result["temperature"].sel(source="model", drop=True).values,
            var_model,
        )
        npt.assert_allclose(
            result["temperature"].sel(source="scenario", drop=True).values,
            var_scenario,
        )
        npt.assert_allclose(
            result_fractional["temperature"].sel(source="internal", drop=True).values,
            var_frac_internal,
        )
        npt.assert_allclose(
            result_fractional["temperature"].sel(source="model", drop=True).values,
            var_frac_model,
        )
        npt.assert_allclose(
            result_fractional["temperature"].sel(source="scenario", drop=True).values,
            var_frac_scenario,
        )
        assert result["temperature"].attrs["units"] == "(Hello there)²"
        assert (
            result["temperature"].attrs["long_name"]
            == "Variance of general Kenobi you are a bold one"
        )
        assert "units" not in result_fractional["temperature"].attrs
        assert (
            result_fractional["temperature"].attrs["long_name"]
            == "Fraction of variance"
        )
        xrt.assert_allclose(
            result[["lon", "lat", "time", "lon_bnds", "lat_bnds", "time_bnds"]],
            ds[["lon", "lat", "time", "lon_bnds", "lat_bnds", "time_bnds"]],
        )

    def test_variance_decomposition_varlist(self):
        """Test with a list of data variables."""
        data_vars = ["temperature", "precipitation"]
        ds = generate_dataset(data_var=data_vars, frequency="monthly")
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        ds["precipitation"].values = np.random.rand(*ds["precipitation"].shape)
        result = ds.climepi.variance_decomposition()
        for data_var in data_vars:
            xrt.assert_allclose(
                result[data_var],
                ds[[data_var]].climepi.variance_decomposition()[data_var],
            )
            xrt.assert_allclose(
                result[data_var],
                ds.climepi.variance_decomposition(data_var)[data_var],
            )

    def test_variance_decomposition_single_scenario_model(self):
        """Test with datasets containing a single scenario and model."""
        ds1 = generate_dataset(data_var="temperature", extra_dims={"realization": 9})
        ds1["temperature"].values = np.random.rand(*ds1["temperature"].shape)
        ds2 = ds1.copy()
        ds2["temperature"] = ds2["temperature"].expand_dims(["model", "scenario"])
        ds3 = ds1.copy()
        ds3["model"] = "googly"
        ds3["scenario"] = "flipper"
        ds3 = ds3.set_coords(["model", "scenario"])
        result1 = ds1.climepi.variance_decomposition()
        result2 = ds2.climepi.variance_decomposition()
        result3 = ds3.climepi.variance_decomposition()
        xrt.assert_allclose(result1, result2)
        xrt.assert_allclose(result1, result3)
        xrt.assert_allclose(
            result1["temperature"].sel(source="internal", drop=True),
            ds1["temperature"].var(dim="realization"),
        )
        npt.assert_allclose(
            result1["temperature"].sel(source="model", drop=True).values, 0
        )
        npt.assert_allclose(
            result1["temperature"].sel(source="scenario", drop=True).values, 0
        )


class TestUncertaintyIntervalDecomposition:
    """Class for testing the uncertainty_interval_decomposition method."""

    def test_uncertainty_interval_decomposition(self):
        """Main test."""
        ds = generate_dataset(
            data_var="temperature",
            extra_dims={"scenario": 6, "model": 4, "realization": 9},
        ).isel(lon=0, lat=0)
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        ds["temperature"] = (  # Make mean 0 at each time to simplify expected values
            ds["temperature"]
            - ds["temperature"].mean(dim=["scenario", "model", "realization"])
        )
        ds["temperature"].attrs = {"reverse": "sweep"}
        result = ds.climepi.uncertainty_interval_decomposition()
        # Check that mean and lower/upper bounds for each component are correct
        z = norm.ppf(0.95)
        da_var_decomp = ds.climepi.variance_decomposition(fraction=False)["temperature"]
        internal_upper_expected = z * np.sqrt(
            da_var_decomp.sel(source="internal").values
        )
        internal_lower_expected = -internal_upper_expected
        model_upper_expected = z * np.sqrt(
            da_var_decomp.sel(source="internal").values
            + da_var_decomp.sel(source="model").values
        )
        model_lower_expected = -model_upper_expected
        scenario_upper_expected = z * np.sqrt(da_var_decomp.sum(dim="source").values)
        scenario_lower_expected = -scenario_upper_expected
        npt.assert_allclose(
            result["temperature"].sel(level="baseline").values, 0, atol=1e-7
        )
        npt.assert_allclose(
            result["temperature"].sel(level="internal_lower").values,
            internal_lower_expected,
        )
        npt.assert_allclose(
            result["temperature"].sel(level="internal_upper").values,
            internal_upper_expected,
        )
        npt.assert_allclose(
            result["temperature"].sel(level="model_lower").values,
            model_lower_expected,
        )
        npt.assert_allclose(
            result["temperature"].sel(level="model_upper").values,
            model_upper_expected,
        )
        npt.assert_allclose(
            result["temperature"].sel(level="scenario_lower").values,
            scenario_lower_expected,
        )
        npt.assert_allclose(
            result["temperature"].sel(level="scenario_upper").values,
            scenario_upper_expected,
        )
        # Check handling of attributes and bounds
        assert result["temperature"].attrs == {"reverse": "sweep"}
        xrt.assert_identical(ds["time_bnds"], result["time_bnds"])

    def test_uncertainty_interval_decomposition_varlist(self):
        """Test with a list of data variables."""
        data_vars = ["temperature", "precipitation"]
        ds = generate_dataset(data_var=data_vars, frequency="monthly")
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        ds["precipitation"].values = np.random.rand(*ds["precipitation"].shape)
        result = ds.climepi.uncertainty_interval_decomposition()
        for data_var in data_vars:
            xrt.assert_allclose(
                result[data_var],
                ds[[data_var]].climepi.uncertainty_interval_decomposition()[data_var],
            )
            xrt.assert_allclose(
                result[data_var],
                ds.climepi.uncertainty_interval_decomposition(data_var)[data_var],
            )

    def test_uncertainty_interval_decomposition_internal_only(self):
        """Test in case where only internal variability is present."""
        ds = generate_dataset(
            data_var="temperature",
            extra_dims={"realization": 231},
        ).isel(lon=0, lat=0)
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        ds["temperature"] = (  # Make mean 0 at each time to simplify expected values
            ds["temperature"] - ds["temperature"].mean(dim="realization")
        )
        result = ds.climepi.uncertainty_interval_decomposition()["temperature"]
        # Test that mean and lower/upper bounds for each component are correct
        lower_expected = ds["temperature"].quantile(0.05, dim="realization").values
        upper_expected = ds["temperature"].quantile(0.95, dim="realization").values
        npt.assert_allclose(
            result.sel(level="baseline").values,
            0,
            atol=1e-7,
        )
        npt.assert_allclose(
            result.sel(level="internal_lower").values,
            lower_expected,
        )
        npt.assert_allclose(
            result.sel(level="internal_upper").values,
            upper_expected,
        )
        xrt.assert_equal(
            result.sel(level="model_lower", drop=True),
            result.sel(level="internal_lower", drop=True),
        )
        xrt.assert_equal(
            result.sel(level="model_upper", drop=True),
            result.sel(level="internal_upper", drop=True),
        )
        xrt.assert_equal(
            result.sel(level="scenario_lower", drop=True),
            result.sel(level="internal_lower", drop=True),
        )
        xrt.assert_equal(
            result.sel(level="scenario_upper", drop=True),
            result.sel(level="internal_upper", drop=True),
        )

    def test_uncertainty_interval_decomposition_model_only(self):
        """Test with only model uncertainty (not estimating internal variability)."""
        ds = generate_dataset(
            data_var="temperature",
            extra_dims={"model": 17},
        ).isel(lon=0, lat=0)
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        ds["temperature"] = (  # Make mean 0 at each time to simplify expected values
            ds["temperature"] - ds["temperature"].mean(dim="model")
        )
        result = ds.climepi.uncertainty_interval_decomposition(
            estimate_internal_variability=False
        )["temperature"]
        # Test that mean and lower/upper bounds for each component are correct
        lower_expected = ds["temperature"].quantile(0.05, dim="model").values
        upper_expected = ds["temperature"].quantile(0.95, dim="model").values
        npt.assert_allclose(
            result.sel(level=["baseline", "internal_lower", "internal_upper"]).values,
            0,
            atol=1e-7,
        )
        npt.assert_allclose(
            result.sel(level="model_lower").values,
            lower_expected,
        )
        npt.assert_allclose(
            result.sel(level="model_upper").values,
            upper_expected,
        )
        xrt.assert_equal(
            result.sel(level="scenario_lower", drop=True),
            result.sel(level="model_lower", drop=True),
        )
        xrt.assert_equal(
            result.sel(level="scenario_upper", drop=True),
            result.sel(level="model_upper", drop=True),
        )

    def test_uncertainty_interval_decomposition_scenario_only(self):
        """Test with only scenario uncertainty (not estimating internal variability)."""
        ds = generate_dataset(
            data_var="temperature",
            extra_dims={"scenario": 17},
        ).isel(lon=0, lat=0)
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        ds["temperature"] = (  # Make mean 0 at each time to simplify expected values
            ds["temperature"] - ds["temperature"].mean(dim="scenario")
        )
        result = ds.climepi.uncertainty_interval_decomposition(
            estimate_internal_variability=False
        )["temperature"]
        # Test that mean and lower/upper bounds for each component are correct
        lower_expected = ds["temperature"].quantile(0.05, dim="scenario").values
        upper_expected = ds["temperature"].quantile(0.95, dim="scenario").values
        npt.assert_allclose(
            result.sel(
                level=[
                    "baseline",
                    "internal_lower",
                    "internal_upper",
                    "model_lower",
                    "model_upper",
                ]
            ).values,
            0,
            atol=1e-7,
        )
        npt.assert_allclose(result.sel(level="scenario_lower").values, lower_expected)
        npt.assert_allclose(result.sel(level="scenario_upper").values, upper_expected)


def test_plot_time_series():
    """
    Test the plot_time_series method of the ClimEpiDatasetAccessor class.

    Since this method is a thin wrapper around hvplot.line, only test that this method
    returns the same result as calling hvplot.line directly in a simple case.
    """
    ds = generate_dataset(
        data_var=["temperature", "precipitation"], extra_dims={"realization": 3}
    )
    ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
    ds["precipitation"].values = np.random.rand(*ds["precipitation"].shape)
    kwargs = {"by": ["realization", "lat", "lon"]}
    result = ds.climepi.plot_time_series("precipitation", **kwargs)
    expected = ds["precipitation"].hvplot.line(x="time", **kwargs)
    hvt.assertEqual(result, expected)


def test_plot_map():
    """
    Test the plot_map method of the ClimEpiDatasetAccessor class.

    Since this method is a thin wrapper around hvplot.quadmesh, only test that this
    method returns the same result as calling hvplot.quadmesh directly in a simple case.
    """
    ds = generate_dataset(
        data_var=["temperature", "precipitation"], lon_0_360=False
    ).isel(time=0)
    ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
    result = ds.climepi.plot_map("temperature", rasterize=False)
    quadmesh_expected = ds["temperature"].hvplot.quadmesh(
        x="lon",
        y="lat",
        cmap="viridis",
        project=True,
        geo=True,
        rasterize=False,
    )
    hvt.compare_quadmesh(result.QuadMesh.I, quadmesh_expected)
    assert isinstance(result.Coastline.I, geoviews.element.geo.Feature)
    assert isinstance(result.Ocean.I, geoviews.element.geo.Feature)


@pytest.mark.parametrize("fraction", [True, False])
def test_plot_variance_decomposition(fraction):
    """Test the plot_variance_decomposition method of the ClimEpiDatasetAccessor class."""
    ds = generate_dataset(
        data_var="temperature",
        extra_dims={"scenario": 6, "model": 4, "realization": 9},
    ).isel(lon=0, lat=0)
    ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
    ds["temperature"].attrs.update(units="there", long_name="Hello")
    result = ds.climepi.plot_variance_decomposition(fraction=fraction)
    # Test that lower/upper bounds for each component are correct
    da_var_decomp = ds.climepi.variance_decomposition(fraction=fraction)["temperature"]
    internal_lower_expected = 0
    internal_upper_expected = da_var_decomp.sel(source="internal").values
    model_lower_expected = internal_upper_expected
    model_upper_expected = model_lower_expected + (
        da_var_decomp.sel(source="model").values
    )
    scenario_lower_expected = model_upper_expected
    scenario_upper_expected = scenario_lower_expected + (
        da_var_decomp.sel(source="scenario").values
    )
    npt.assert_allclose(
        result["Internal variability"].data.Baseline.values, internal_lower_expected
    )
    npt.assert_allclose(
        result["Internal variability"].data["Internal variability"].values,
        internal_upper_expected,
    )
    npt.assert_allclose(
        result["Model uncertainty"].data.Baseline.values,
        model_lower_expected,
    )
    npt.assert_allclose(
        result["Model uncertainty"].data["Model uncertainty"].values,
        model_upper_expected,
    )
    npt.assert_allclose(
        result["Scenario uncertainty"].data.Baseline.values,
        scenario_lower_expected,
    )
    npt.assert_allclose(
        result["Scenario uncertainty"].data["Scenario uncertainty"].values,
        scenario_upper_expected,
    )
    if fraction:
        assert result.opts["ylabel"] == "Fraction of variance"
    else:
        assert result.opts["ylabel"] == "Variance of hello (there²)"


class TestPlotUncertaintyIntervalDecomposition:
    """Class for testing the plot_uncertainty_decomposition method."""

    def test_plot_uncertainty_interval_decomposition(self):
        """Main test."""
        ds = generate_dataset(
            data_var="temperature",
            extra_dims={"scenario": 6, "model": 4, "realization": 9},
        ).isel(lon=0, lat=0)
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        result = ds.climepi.plot_uncertainty_interval_decomposition()
        assert list(result.data.keys()) == [
            ("Area", "Scenario_uncertainty"),
            ("Area", "I"),  # unlabelled upper portion due to scenario uncertainty
            ("Area", "Model_uncertainty"),
            ("Area", "II"),  # unlabelled upper portion due to model uncertainty
            ("Area", "Internal_variability"),
            ("Curve", "Mean"),
        ]
        # Test that mean and lower/upper bounds for each component are correct
        da_decomp = ds.climepi.uncertainty_interval_decomposition()["temperature"]
        npt.assert_allclose(
            result.Curve.Mean.data.temperature,
            da_decomp.sel(level="baseline"),
        )
        npt.assert_allclose(  # Portion due to internal variability
            result.Area.Internal_variability.data.lower,
            da_decomp.sel(level="internal_lower"),
        )
        npt.assert_allclose(
            result.Area.Internal_variability.data.upper,
            da_decomp.sel(level="internal_upper"),
        )
        npt.assert_allclose(  # Lower portion due to model uncertainty
            result.Area.Model_uncertainty.data.lower,
            da_decomp.sel(level="model_lower"),
        )
        npt.assert_allclose(
            result.Area.Model_uncertainty.data.upper,
            da_decomp.sel(level="internal_lower"),
        )
        npt.assert_allclose(  # Upper portion due to model uncertainty
            result.Area.II.data.lower,
            da_decomp.sel(level="internal_upper"),
        )
        npt.assert_allclose(
            result.Area.II.data.upper,
            da_decomp.sel(level="model_upper"),
        )
        npt.assert_allclose(  # Lower portion due to scenario uncertainty
            result.Area.Scenario_uncertainty.data.lower,
            da_decomp.sel(level="scenario_lower"),
        )
        npt.assert_allclose(
            result.Area.Scenario_uncertainty.data.upper,
            da_decomp.sel(level="model_lower"),
        )
        npt.assert_allclose(  # Upper portion due to scenario uncertainty
            result.Area.I.data.lower,
            da_decomp.sel(level="model_upper"),
        )
        npt.assert_allclose(
            result.Area.I.data.upper,
            da_decomp.sel(level="scenario_upper"),
        )

    def test_plot_uncertainty_interval_decomposition_internal_only(self):
        """Test in case where only internal variability is present."""
        ds = generate_dataset(
            data_var="temperature",
            extra_dims={"realization": 231},
        ).isel(lon=0, lat=0)
        ds["temperature"].values = np.random.rand(*ds["temperature"].shape)
        result = ds.climepi.plot_uncertainty_interval_decomposition()
        assert list(result.data.keys()) == [
            ("Area", "Internal_variability"),
            ("Curve", "Mean"),
        ]

    def test_plot_uncertainty_interval_decomposition_model_only(self):
        """Test with only model uncertainty (not estimating internal variability)."""
        ds = generate_dataset(
            data_var="temperature",
            extra_dims={"model": 17},
        ).isel(lon=0, lat=0)
        result = ds.climepi.plot_uncertainty_interval_decomposition(
            estimate_internal_variability=False
        )
        assert list(result.data.keys()) == [
            ("Area", "Model_uncertainty"),
            ("Area", "I"),  # unlabelled upper portion due to model uncertainty
            ("Curve", "Mean"),
        ]

    def test_plot_uncertainty_interval_decomposition_no_model(self):
        """Test with no model uncertainty, estimating internal variability."""
        ds = generate_dataset(
            data_var="temperature",
            extra_dims={"scenario": 17},
        ).isel(lon=0, lat=0)
        result = ds.climepi.plot_uncertainty_interval_decomposition()
        assert list(result.data.keys()) == [
            ("Area", "Scenario_uncertainty"),
            ("Area", "I"),  # unlabelled upper portion due to scenario uncertainty
            ("Area", "Internal_variability"),
            ("Curve", "Mean"),
        ]


def test__process_data_var_argument():
    """Test the _process_data_var_argument method of ClimEpiDatasetAccessor."""
    ds1 = generate_dataset(data_var="temperature")
    ds2 = generate_dataset(data_var=["temperature", "precipitation"])
    assert ds1.climepi._process_data_var_argument("temperature") == "temperature"
    assert ds1.climepi._process_data_var_argument(["temperature"], as_list=True) == [
        "temperature"
    ]
    assert ds1.climepi._process_data_var_argument() == "temperature"
    assert ds1.climepi._process_data_var_argument(as_list=True) == ["temperature"]
    with pytest.raises(ValueError):
        ds1.climepi._process_data_var_argument(["temperature"])
    assert ds2.climepi._process_data_var_argument("precipitation") == "precipitation"
    assert ds2.climepi._process_data_var_argument(
        ["temperature", "precipitation"], as_list=True
    ) == [
        "temperature",
        "precipitation",
    ]
    assert ds2.climepi._process_data_var_argument(as_list=True) == [
        "temperature",
        "precipitation",
    ]
    with pytest.raises(ValueError):
        ds2.climepi._process_data_var_argument()
    with pytest.raises(ValueError):
        ds2.climepi._process_data_var_argument(["temperature", "precipitation"])
    with pytest.raises(ValueError):
        ds2.climepi._process_data_var_argument(
            ("temperature", "precipitation"), as_list=True
        )
