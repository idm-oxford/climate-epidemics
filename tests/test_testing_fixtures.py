"""Unit tests for the fixtures module in the testing subpackage."""

import numpy as np
import numpy.testing as npt
import pytest
import xarray.testing as xrt

from climepi._xcdat import _infer_freq
from climepi.testing.fixtures import generate_dataset


@pytest.mark.parametrize("dtype", ["bool", "int", "float"])
@pytest.mark.parametrize(
    "frequency,lon_0_360,has_bounds,random",
    [
        ("daily", False, False, False),
        ("monthly", True, True, True),
        ("yearly", False, True, True),
    ],
)
def test_generate_dataset(dtype, frequency, lon_0_360, has_bounds, random):
    """Unit test for the generate_dataset fixture."""
    ds = generate_dataset(
        data_var=["swing", "miss"],
        dtype=dtype,
        frequency=frequency,
        lon_0_360=lon_0_360,
        extra_dims={"cutter": 3, "cross-seam": 2},
        has_bounds=has_bounds,
        random=random,
    )
    assert np.issubdtype(ds.miss.dtype, dtype)
    assert set(ds.data_vars) == (
        {"swing", "miss", "time_bnds", "lat_bnds", "lon_bnds"}
        if has_bounds
        else {"swing", "miss"}
    )
    assert set(ds.dims) == (
        {"time", "lat", "lon", "cutter", "cross-seam", "bnds"}
        if has_bounds
        else {"time", "lat", "lon", "cutter", "cross-seam"}
    )
    assert (
        _infer_freq(ds.time) == "day"
        if frequency == "daily"
        else "month"
        if frequency == "monthly"
        else "year"
        if frequency == "yearly"
        else None
    )
    if lon_0_360:
        assert np.all(ds.lon >= 0)
        assert np.all(ds.lon <= 360)
        if has_bounds:
            assert np.all(ds.lon_bnds >= 0)
            assert np.all(ds.lon_bnds <= 360)
    else:
        assert np.all(ds.lon >= -180)
        assert np.all(ds.lon <= 180)
        if has_bounds:
            assert np.all(ds.lon_bnds >= -180)
            assert np.all(ds.lon_bnds <= 180)
    if has_bounds:
        xrt.assert_equal(ds.time_bnds.mean(dim="bnds"), ds.time)
        if frequency == "daily":
            npt.assert_equal(
                np.mod(
                    ds.time_bnds.sel(bnds=1).dt.dayofyear.values
                    - ds.time_bnds.sel(bnds=0).dt.dayofyear.values,
                    365,
                ),
                1,
            )
        elif frequency == "monthly":
            npt.assert_equal(ds.time_bnds.dt.day.values, 1)
            npt.assert_equal(
                np.mod(
                    ds.time_bnds.sel(bnds=1).dt.month.values
                    - ds.time_bnds.sel(bnds=0).dt.month.values,
                    12,
                ),
                1,
            )
        elif frequency == "yearly":
            npt.assert_equal(ds.time_bnds.dt.dayofyear.values, 1)
            npt.assert_equal(
                ds.time_bnds.sel(bnds=1).dt.year.values
                - ds.time_bnds.sel(bnds=0).dt.year.values,
                1,
            )
        else:
            raise ValueError(f"Unexpected frequency: {frequency} provided to test.")
    if random:
        first_value = ds.miss.values.flatten()[0]
        if dtype in ["bool", "int"]:
            assert np.any(ds.miss != first_value)
        elif dtype == "float":
            assert not np.allclose(ds.miss, first_value)
        else:
            raise ValueError(f"Unexpected dtype: {dtype} provided to test.")
    else:
        npt.assert_allclose(ds.miss, 1)


def test_generate_dataset_invalid_frequency():
    """Unit test for the generate_dataset fixture with invalid frequency."""
    with pytest.raises(ValueError, match="Invalid frequency"):
        generate_dataset(frequency="hourly")
