"""
Unit tests for the utils module.
"""

import xarray.testing as xrt

from climepi.testing.fixtures import generate_dataset
from climepi.utils import (
    add_bnds_from_other,
    add_var_attrs_from_other,
    get_data_var_and_bnds,
    list_non_bnd_data_vars,
)


def test_add_var_attrs_from_other():
    ds = generate_dataset(data_var=["temperature", "precipitation"])
    ds_from = ds.drop_vars("precipitation")
    ds_from["temperature"].attrs["units"] = "K"
    ds_from["lon"].attrs["units"] = "degrees_east"
    ds["temperature"].attrs["units"] = "C"
    ds["lat"].attrs["units"] = "degrees_north"
    ds_out = add_var_attrs_from_other(ds, ds_from)
    assert ds_out["temperature"].attrs["units"] == "C"
    assert ds_out["lon"].attrs["units"] == "degrees_east"
    assert ds_out["lat"].attrs["units"] == "degrees_north"
    assert ds_out["precipitation"].attrs == {}


def test_add_bnds_from_other():
    ds = generate_dataset(has_bounds=False)
    ds_from = generate_dataset()
    ds["lon_bnds"] = ds_from["lon_bnds"] + 1
    ds["lon"].attrs["bounds"] = "incorrect label"
    ds_out = add_bnds_from_other(ds, ds_from)
    xrt.assert_identical(ds_out["time_bnds"], ds_from["time_bnds"])
    xrt.assert_identical(ds_out["lat_bnds"], ds_from["lat_bnds"])
    xrt.assert_equal(ds_out["lon_bnds"], ds_from["lon_bnds"] + 1)  # expect diff attrs
    assert ds_out["time"].attrs["bounds"] == "time_bnds"
    assert "time_bnds" not in ds.time.attrs
    assert ds_out["lat"].attrs["bounds"] == "lat_bnds"
    assert ds_out["lon"].attrs["bounds"] == "incorrect label"


def test_get_data_var_and_bnds():
    ds = generate_dataset(data_var=["temperature", "precipitation", "kenobi"])
    result1 = get_data_var_and_bnds(ds, "temperature")
    xrt.assert_identical(
        result1, ds[["temperature", "lat_bnds", "lon_bnds", "time_bnds"]]
    )
    result2 = get_data_var_and_bnds(ds, ["temperature"])
    xrt.assert_identical(
        result2, ds[["temperature", "lat_bnds", "lon_bnds", "time_bnds"]]
    )
    result3 = get_data_var_and_bnds(ds, ["temperature", "kenobi"])
    xrt.assert_identical(
        result3, ds[["temperature", "kenobi", "lat_bnds", "lon_bnds", "time_bnds"]]
    )


def test_list_non_bnd_data_vars():
    ds1 = generate_dataset(data_var="temperature")
    ds2 = generate_dataset(data_var=["temperature", "precipitation"]).drop_vars(
        "lon_bnds"
    )
    ds3 = generate_dataset(data_var=["temperature"], has_bounds=False)
    result1 = list_non_bnd_data_vars(ds1)
    result2 = list_non_bnd_data_vars(ds2)
    result3 = list_non_bnd_data_vars(ds3)
    assert result1 == ["temperature"]
    assert "lon_bnds" in ds1  # make sure input dataset actually has bounds
    assert result2 == ["temperature", "precipitation"]
    assert result3 == ["temperature"]
