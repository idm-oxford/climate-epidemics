"""Unit tests for the _ensemble_stats module of the climepi package."""

import cftime
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt
from scipy.interpolate import make_smoothing_spline
from scipy.stats import norm

from climepi._ensemble_stats import (
    _ensemble_mean_var_polyfit,
    _ensemble_mean_var_polyfit_multiple_realizations,
    _ensemble_mean_var_splinefit,
    _ensemble_mean_var_splinefit_multiple_realizations,
    _ensemble_stats_direct,
    _ensemble_stats_fit,
)
from climepi.testing.fixtures import generate_dataset


class TestEnsembleStatsDirect:
    """Class for testing the _ensemble_stats_direct function."""

    def test_ensemble_stats_direct(self):
        """Test the _ensemble_stats_direct function."""
        ds = generate_dataset(
            data_var=["temperature", "precipitation"],
            extra_dims={"realization": 12, "ouch": 4},
            has_bounds=False,
        )
        result = _ensemble_stats_direct(ds, uncertainty_level=60)
        xrt.assert_allclose(
            result.sel(stat="mean", drop=True),
            ds.mean(dim="realization"),
        )
        xrt.assert_allclose(
            result.sel(stat="std", drop=True),
            ds.std(dim="realization"),
        )
        xrt.assert_allclose(
            result.sel(stat="var", drop=True),
            ds.var(dim="realization"),
        )
        xrt.assert_allclose(
            result.sel(stat="median", drop=True),
            ds.median(dim="realization"),
        )
        xrt.assert_allclose(
            result.sel(stat="min", drop=True),
            ds.min(dim="realization"),
        )
        xrt.assert_allclose(
            result.sel(stat="max", drop=True),
            ds.max(dim="realization"),
        )
        xrt.assert_allclose(
            result.sel(stat="lower", drop=True),
            ds.quantile(0.2, dim="realization").drop_vars("quantile"),
        )
        xrt.assert_allclose(
            result.sel(stat="upper", drop=True),
            ds.quantile(0.8, dim="realization").drop_vars("quantile"),
        )
        xrt.assert_allclose(
            result[["lon", "lat", "time"]],
            ds[["lon", "lat", "time"]],
        )

    def test_ensemble_stats_direct_single_realization(self):
        """Test with a single realization."""
        ds = generate_dataset(
            data_var="temperature", extra_dims={"realization": 1}, has_bounds=False
        )
        result = _ensemble_stats_direct(ds, uncertainty_level=95)
        for stat in ["mean", "median", "min", "max", "lower", "upper"]:
            xrt.assert_allclose(
                result.sel(stat=stat, drop=True), ds.isel(realization=0, drop=True)
            )
        for stat in ["std", "var"]:
            npt.assert_allclose(
                result.sel(stat=stat, drop=True)["temperature"].values, 0
            )


class TestEnsembleStatsFit:
    """Class for testing the _ensemble_stats_fit function."""

    @pytest.mark.parametrize(
        "internal_variability_method", ["polyfit", "splinefit", "fakemethod"]
    )
    def test_ensemble_stats_fit(self, internal_variability_method):
        """
        Test the _ensemble_stats_fit function.

        Note that calculation of the mean and variance is done via subroutines and is tested
        separately.
        """
        ds = generate_dataset(
            data_var=["temperature", "precipitation"],
            frequency="monthly",
            extra_dims={"ouch": 4},
            has_bounds=False,
        )

        if internal_variability_method == "fakemethod":
            with pytest.raises(ValueError, match="Unknown internal_variability_method"):
                _ensemble_stats_fit(
                    ds,
                    uncertainty_level=90,
                    internal_variability_method=internal_variability_method,
                )
            return

        result = _ensemble_stats_fit(
            ds,
            uncertainty_level=90,
            internal_variability_method=internal_variability_method,
            deg=3,
            lam=0.1,
        )

        if internal_variability_method == "polyfit":
            ds_mean_expected, ds_var_expected = _ensemble_mean_var_polyfit(ds, deg=3)
        elif internal_variability_method == "splinefit":
            ds_mean_expected, ds_var_expected = _ensemble_mean_var_splinefit(
                ds, lam=0.1
            )
        else:
            raise ValueError(
                f"Unexpected internal_variability_method: {internal_variability_method}"
            )
        xrt.assert_equal(
            result.sel(stat="mean", drop=True), ds_mean_expected, check_dim_order=False
        )
        xrt.assert_equal(  # ds_var is assumed constant in time and has no time dimension
            result.sel(stat="var", drop=True).isel(time=3, drop=True),
            ds_var_expected,
            check_dim_order=False,
        )
        xrt.assert_allclose(
            result.sel(stat="std", drop=True).isel(time=-1, drop=True) ** 2,
            ds_var_expected,
        )
        xrt.assert_allclose(
            (
                result.sel(stat="upper", drop=True) - result.sel(stat="mean", drop=True)
            ).isel(time=4, drop=True),
            norm.ppf(0.95) * ds_var_expected**0.5,
        )
        xrt.assert_allclose(
            (
                result.sel(stat="lower", drop=True) - result.sel(stat="mean", drop=True)
            ).isel(time=6, drop=True),
            norm.ppf(0.05) * ds_var_expected**0.5,
        )

    def test_ensemble_stats_fit_realization_provided(self):
        """Test nothing changes if a (singleton) realization coordinate is provided."""
        ds = generate_dataset(data_var="temperature", has_bounds=False)
        kwargs = {
            "uncertainty_level": 90,
            "internal_variability_method": "polyfit",
            "deg": 3,
        }
        result1 = _ensemble_stats_fit(ds, **kwargs)
        result2 = _ensemble_stats_fit(
            ds.assign_coords({"realization": "a realization"}), **kwargs
        )
        result3 = _ensemble_stats_fit(ds.expand_dims("realization"), **kwargs)

        xrt.assert_identical(result1, result2)
        xrt.assert_identical(result1, result3)


class TestEnsembleMeanVarPolyfit:
    """Class for testing the _ensemble_mean_var_polyfit function."""

    def test_ensemble_mean_var_polyfit(self):
        """
        Main test.

        This test is based on estimating ensemble stats from a temperature time series
        made up of normally distributed noise added to a polynomial (matching the
        underlying assumptions of the estimate_ensemble_stats method). This is repeated
        multiple times to ensure there is no systematic bias in the estimated ensemble
        statistics.
        """
        time = xr.date_range(
            start="2001-01-01", periods=10000, freq="MS", use_cftime=True
        )
        days_from_start = cftime.date2num(time, "days since 2001-01-01")
        mean_theoretical = (
            0.0000000000123 * days_from_start**3
            - 0.00000257 * days_from_start**2
            - 0.326 * days_from_start
            - 259.29
        )
        std_theoretical = 0.734
        var_theoretical = std_theoretical**2
        repeats = 100
        mean_result_sum = np.zeros_like(mean_theoretical)
        var_result_sum = np.zeros_like(mean_theoretical)
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
            ds_mean, ds_var = _ensemble_mean_var_polyfit(ds, deg=5)
            if repeat == 0:
                # Just check for the first repeat that the results match those obtained
                # by directly applying numpy's polynomial fitting method.
                fitted_poly_for_expected_values = np.polynomial.Polynomial.fit(
                    days_from_start, temperature_values_in, 5, full=True
                )
                mean_expected = fitted_poly_for_expected_values[0](days_from_start)
                var_expected = fitted_poly_for_expected_values[1][0][0] / len(
                    days_from_start
                )
                npt.assert_allclose(
                    ds_mean["temperature"].values,
                    mean_expected,
                )
                npt.assert_allclose(
                    ds_var["temperature"].values,
                    var_expected,
                )
            mean_result_sum += ds_mean["temperature"].values
            var_result_sum += ds_var["temperature"].values
        mean_result_avg = mean_result_sum / repeats
        var_result_avg = var_result_sum / repeats
        rtol_theoretical_match = 1e-2
        npt.assert_allclose(
            mean_result_avg,
            mean_theoretical,
            rtol=rtol_theoretical_match,
        )
        npt.assert_allclose(
            var_result_avg,
            var_theoretical,
            rtol=rtol_theoretical_match,
        )

    def test_ensemble_mean_var_polyfit_vars_coords(self):
        """Test with multiple data variables and extra coordinates."""
        ds = generate_dataset(
            data_var=["temperature", "precipitation"],
            frequency="monthly",
            extra_dims={"ouch": 4},
            has_bounds=False,
        )
        result = _ensemble_mean_var_polyfit(ds, deg=3)
        for i in range(2):
            xrt.assert_allclose(
                result[i].isel(ouch=3),
                _ensemble_mean_var_polyfit(
                    ds.isel(ouch=3),
                    deg=3,
                )[i],
            )

    def test_ensemble_mean_var_polyfit_multiple_realizations(self):
        """
        Test with a dataset containing multiple realizations.

        Most of the functionality in this case is handled by
        _ensemble_mean_var_polyfit_multiple_realizations, so this test just checks that
        the results are the same as those obtained calling that function directly.
        """
        ds = generate_dataset(
            data_var="temperature",
            frequency="monthly",
            extra_dims={"realization": 4},
            has_bounds=False,
        )
        result = _ensemble_mean_var_polyfit(ds, deg=3)
        expected = _ensemble_mean_var_polyfit_multiple_realizations(ds, deg=3)
        for i in range(2):
            xrt.assert_allclose(
                result[i],
                expected[i],
            )

    def test_ensemble_mean_var_polyfit_non_string_data_var(self):
        """Test with a dataset where the data variable is not a string."""
        ds = generate_dataset(
            data_var=[("hello",)],
            frequency="monthly",
            has_bounds=False,
        ).isel(lat=0, lon=0, drop=True)
        with pytest.raises(ValueError, match="Data variable names must be strings."):
            _ensemble_mean_var_polyfit(ds, deg=3)


def test_ensemble_mean_var_polyfit_multiple_realizations():
    """Test for the _ensemble_mean_var_polyfit_multiple_realizations function."""
    ds = generate_dataset(
        data_var="temperature",
        frequency="monthly",
        extra_dims={"realization": 4},
        has_bounds=False,
    ).isel(lat=0, lon=0, drop=True)
    time_values = np.arange(ds.time.size)
    ds["time"] = time_values
    result = _ensemble_mean_var_polyfit_multiple_realizations(ds, deg=3)

    fitted_poly_for_expected_values = np.polynomial.Polynomial.fit(
        np.tile(time_values, 4),
        np.ravel(ds["temperature"].transpose("realization", "time")),
        3,
        full=True,
    )
    mean_expected = fitted_poly_for_expected_values[0](time_values)
    var_expected = fitted_poly_for_expected_values[1][0][0] / (4 * len(time_values))

    npt.assert_allclose(
        result[0]["temperature"].values,
        mean_expected,
    )
    npt.assert_allclose(
        result[1]["temperature"].values,
        var_expected,
    )
    assert list(result[0].dims) == ["time"]
    assert not result[1].dims


class TestEnsembleMeanVarSplinefit:
    """Class for testing the _ensemble_mean_var_splinefit function."""

    def test_ensemble_mean_var_splinefit(self):
        """Main test."""
        ds = generate_dataset(
            data_var="temperature",
            frequency="monthly",
            has_bounds=False,
        ).isel(lat=0, lon=0, drop=True)
        time_values = np.linspace(0, 1, ds.time.size)
        ds["time"] = time_values
        result = _ensemble_mean_var_splinefit(ds, lam=0.1)

        mean_expected = make_smoothing_spline(
            time_values, ds["temperature"].values, lam=0.1
        )(time_values)
        var_expected = np.mean((ds["temperature"].values - mean_expected) ** 2)

        npt.assert_allclose(
            result[0]["temperature"].values,
            mean_expected,
        )
        npt.assert_allclose(
            result[1]["temperature"].values,
            var_expected,
        )
        assert list(result[0].dims) == ["time"]
        assert not result[1].dims

    def test_ensemble_mean_var_splinefit_vars_coords(self):
        """Test with multiple data variables and extra coordinates."""
        ds = generate_dataset(
            data_var=["temperature", "precipitation"],
            frequency="monthly",
            extra_dims={"ouch": 4},
            has_bounds=False,
        )
        result = _ensemble_mean_var_splinefit(ds)
        for i in range(2):
            xrt.assert_allclose(
                result[i].isel(ouch=2),
                _ensemble_mean_var_splinefit(
                    ds.isel(ouch=2),
                )[i],
            )

    def test_ensemble_mean_var_spline_multiple_realizations(self):
        """
        Test with a dataset containing multiple realizations.

        Most of the functionality in this case is handled by
        _ensemble_mean_var_splinefit_multiple_realizations, so this test just checks
        that the results are the same as those obtained calling that function directly.
        """
        ds = generate_dataset(
            data_var="temperature",
            frequency="monthly",
            extra_dims={"realization": 4},
            has_bounds=False,
        )
        result = _ensemble_mean_var_splinefit(ds, lam=0.1)
        expected = _ensemble_mean_var_splinefit_multiple_realizations(ds, lam=0.1)
        for i in range(2):
            xrt.assert_allclose(
                result[i],
                expected[i],
            )


def test_ensemble_mean_var_splinefit_multiple_realizations():
    """Test for the _ensemble_mean_var_splinefit_multiple_realizations function."""
    ds = generate_dataset(
        data_var="temperature",
        frequency="monthly",
        extra_dims={"realization": 4},
        has_bounds=False,
    ).isel(lat=0, lon=0, drop=True)
    time_values = np.linspace(0, 1, ds.time.size)
    ds["time"] = time_values
    result = _ensemble_mean_var_splinefit_multiple_realizations(ds)

    mean_expected = make_smoothing_spline(
        time_values, ds["temperature"].mean(dim="realization").values
    )(time_values)
    var_expected = np.mean((ds["temperature"].values - mean_expected) ** 2)

    npt.assert_allclose(
        result[0]["temperature"].values,
        mean_expected,
    )
    npt.assert_allclose(
        result[1]["temperature"].values,
        var_expected,
    )
    assert list(result[0].dims) == ["time"]
    assert not result[1].dims
