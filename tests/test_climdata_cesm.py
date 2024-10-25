"""
Unit tests for the the _cesm.py module of the climdata subpackage.

The CESMDataGetter class is tested.
"""

import pathlib
import tempfile
from unittest.mock import patch

import intake_esm
import netCDF4  # noqa (avoids warning https://github.com/pydata/xarray/issues/7259)
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt

from climepi.climdata._cesm import CESMDataGetter


@pytest.mark.parametrize("frequency", ["daily", "monthly", "yearly"])
def test_find_remote_data(frequency):
    """
    Test the _find_remote_data method of the CESMDataGetter class.

    The conversion of
    the intake_esm catalog to a dataset dictionary is mocked to avoid opening the
    actual remote dataset.
    """
    remote_frequency = "monthly" if frequency == "yearly" else frequency

    ds = xr.Dataset(
        data_vars={
            var: xr.DataArray(np.random.rand(6, 4), dims=["time", "member_id"])
            for var in ["TREFHT", "PRECC", "PRECL"]
        },
        coords={
            "time": xr.DataArray(np.arange(6), dims="time"),
            "member_id": xr.DataArray(np.arange(4), dims="member_id"),
        },
    )

    mock_to_dataset_dict_return_value = {
        "atm." + forcing + "." + remote_frequency + "." + assumption: ds.isel(
            time=3 * (forcing == "ssp370") + np.arange(3),
            member_id=2 * (assumption == "smbb") + np.arange(2),
        )
        for forcing in ["historical", "ssp370"]
        for assumption in ["cmip6", "smbb"]
    }

    data_getter = CESMDataGetter(frequency=frequency)

    with patch.object(
        intake_esm.core.esm_datastore,
        "to_dataset_dict",
        return_value=mock_to_dataset_dict_return_value,
        autospec=True,
    ) as mock_to_dataset_dict:
        data_getter._find_remote_data()

    mock_to_dataset_dict.assert_called_once()
    call_catalog_subset = mock_to_dataset_dict.call_args.args[0]
    call_kwargs = mock_to_dataset_dict.call_args.kwargs
    assert isinstance(call_catalog_subset, intake_esm.core.esm_datastore)
    assert sorted(call_catalog_subset.df.path.tolist()) == sorted(
        [
            "s3://ncar-cesm2-lens/atm/"
            + f"{remote_frequency}/cesm2LE-{forcing}-{assumption}-{var}.zarr"
            for forcing in ["historical", "ssp370"]
            for assumption in ["cmip6", "smbb"]
            for var in ["TREFHT", "PRECC", "PRECL"]
        ]
    )
    assert call_kwargs == {"storage_options": {"anon": True}}
    xrt.assert_identical(data_getter._ds, ds)


@pytest.mark.parametrize("year_mode", ["single", "multiple"])
@pytest.mark.parametrize(
    "location_mode",
    ["single_named", "multiple_named", "grid_lon_0_360", "grid_lon_180_180"],
)
def test_subset_remote_data(year_mode, location_mode):
    """Test the _subset_remote_data method of the CESMDataGetter class."""
    time_lb = xr.cftime_range(start="2001-01-01", periods=36, freq="MS")
    time_rb = xr.cftime_range(start="2001-02-01", periods=36, freq="MS")
    time_bnds = xr.DataArray(np.array([time_lb, time_rb]).T, dims=("time", "nbnd"))
    time = time_bnds.mean(dim="nbnd")
    ds_all = xr.Dataset(
        data_vars={
            "gus": xr.DataArray(
                np.random.rand(36, 4, 3, 5), dims=["time", "member_id", "lat", "lon"]
            ),
        },
        coords={
            "time": time,
            "time_bnds": time_bnds,
            "member_id": xr.DataArray(["id1", "id2", "id3", "id4"], dims="member_id"),
            "lat": xr.DataArray([-30, 15, 60], dims="lat"),
            "lon": xr.DataArray([0, 50, 180, 230, 359], dims="lon"),
        },
    )

    if year_mode == "single":
        years = 2002
        time_inds_expected = slice(12, 24)
    elif year_mode == "multiple":
        years = [2002, 2003]
        time_inds_expected = slice(12, 36)
    if location_mode == "single_named":
        locations = "Los Angeles"
        lat_range = None
        lon_range = None
    elif location_mode == "multiple_named":
        locations = ["Los Angeles", "Tokyo"]
        lat_range = None
        lon_range = None
    elif location_mode == "grid_lon_0_360":
        locations = None
        lat_range = [10, 60]
        lon_range = [15, 240]
        lat_inds_expected = [1, 2]
        lon_inds_expected = [1, 2, 3]
    elif location_mode == "grid_lon_180_180":
        locations = None
        lat_range = [-20, 30]
        lon_range = [-30, 60]
        lat_inds_expected = [1]
        lon_inds_expected = [0, 1, 4]
    subset = {
        "years": years,
        "realizations": [0, 2],
        "locations": locations,
        "lat_range": lat_range,
        "lon_range": lon_range,
    }
    data_getter = CESMDataGetter(frequency="monthly", subset=subset)
    data_getter._ds = ds_all
    data_getter._subset_remote_data()
    if location_mode in ["grid_lon_0_360", "grid_lon_180_180"]:
        xrt.assert_identical(
            data_getter._ds,
            ds_all.isel(
                time=time_inds_expected,
                lat=lat_inds_expected,
                lon=lon_inds_expected,
                member_id=[0, 2],
            ),
        )
    else:
        xrt.assert_identical(
            data_getter._ds,
            ds_all.isel(time=time_inds_expected, member_id=[0, 2]).climepi.sel_geo(
                np.atleast_1d(locations).tolist()
            ),
        )


@patch.object(xr.Dataset, "to_netcdf", autospec=True)
def test_download_remote_data(mock_to_netcdf):
    """
    Unit test for the _download_remote_data method of the CESMDataGetter class.

    The download is mocked to avoid actually downloading the remote data.
    """
    ds = xr.Dataset(data_vars={"chris": xr.DataArray(np.random.rand(6), dims=["ball"])})
    data_getter = CESMDataGetter()
    data_getter._temp_save_dir = pathlib.Path(".")
    data_getter._ds = ds
    data_getter._download_remote_data()
    assert data_getter._temp_file_names == ["temp_data.nc"]
    mock_to_netcdf.assert_called_once_with(
        ds, pathlib.Path("temp_data.nc"), compute=False
    )
    mock_to_netcdf.return_value.compute.assert_called_once_with()


def test_open_temp_data():
    """
    Unit test for the _open_temp_data method of the CESMDataGetter class.

    Checks that chunking is preserved when the temporary dataset is opened.
    """
    time_lb = xr.cftime_range(
        start="2001-01-01", periods=12, freq="MS", calendar="noleap"
    )
    time_rb = xr.cftime_range(
        start="2001-02-01", periods=12, freq="MS", calendar="noleap"
    )
    time_bnds = xr.DataArray(np.array([time_lb, time_rb]).T, dims=("time", "nbnd"))
    time = time_bnds.mean(dim="nbnd")
    ds = xr.Dataset(
        data_vars={
            "mark": xr.DataArray(np.random.rand(12, 4), dims=["time", "member_id"]),
        },
        coords={
            "time": time,
            "time_bnds": time_bnds,
            "member_id": xr.DataArray(["id1", "id2", "id3", "id4"], dims="member_id"),
        },
    )
    ds.time.attrs = {"bounds": "time_bnds"}
    ds.time.encoding = {"calendar": "noleap", "units": "days since 2000-01-01"}
    ds["mark"] = ds["mark"].chunk({"time": 1, "member_id": 2})

    with tempfile.TemporaryDirectory() as _temp_dir:
        temp_save_dir = pathlib.Path(_temp_dir)
        ds.to_netcdf(temp_save_dir / "temp_data.nc")

        data_getter = CESMDataGetter()
        data_getter._ds = ds
        data_getter._temp_save_dir = temp_save_dir
        data_getter._temp_file_names = ["temp_data.nc"]
        data_getter._open_temp_data()
        xrt.assert_identical(data_getter._ds, ds)
        xrt.assert_identical(data_getter._ds_temp, ds)
        assert data_getter._ds.chunks.mapping == {
            "time": (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            "member_id": (2, 2),
        }
        assert (
            "nbnd" in data_getter._ds_temp.chunks
        )  # time_bnds chunked in _ds_temp but not in _ds
        data_getter._ds_temp.close()


@pytest.mark.parametrize("frequency", ["monthly", "yearly"])
def test_process_data(frequency):
    """Test the _process_data method of the CESMDataGetter class."""
    time_lb = xr.cftime_range(
        start="2001-01-01", periods=36, freq="MS", calendar="noleap"
    )
    time_rb = xr.cftime_range(
        start="2001-02-01", periods=36, freq="MS", calendar="noleap"
    )
    time_bnds_in = xr.DataArray(np.array([time_lb, time_rb]).T, dims=("time", "nbnd"))
    time_in = time_bnds_in.mean(dim="nbnd").assign_attrs(bounds="time_bnds")
    time_in.encoding = {"calendar": "noleap", "units": "days since 2000-01-01"}
    ds_unprocessed = xr.Dataset(
        data_vars={
            "TREFHT": xr.DataArray(
                np.random.rand(36, 2, 1, 1), dims=["time", "member_id", "lat", "lon"]
            ),
            "PRECC": xr.DataArray(
                np.random.rand(36, 2, 1, 1), dims=["time", "member_id", "lat", "lon"]
            ),
            "PRECL": xr.DataArray(
                np.random.rand(36, 2, 1, 1), dims=["time", "member_id", "lat", "lon"]
            ),
        },
        coords={
            "time": time_in,
            "time_bnds": time_bnds_in,
            "member_id": xr.DataArray(["id1", "id2"], dims="member_id"),
            "lat": xr.DataArray([30], dims="lat"),
            "lon": xr.DataArray([150], dims="lon"),
        },
    )
    data_getter = CESMDataGetter(frequency=frequency, subset={"realizations": [0, 1]})
    data_getter._ds = ds_unprocessed
    data_getter._process_data()
    ds_processed = data_getter._ds
    # Check changes in dimensions/coords
    npt.assert_equal(
        ds_processed.temperature.dims,
        ["scenario", "model", "time", "realization", "lat", "lon"],
    )
    npt.assert_equal(ds_processed["realization"].values, [0, 1])
    npt.assert_equal(ds_processed["scenario"].values, "ssp370")
    npt.assert_equal(ds_processed["model"].values, "cesm2")
    assert "time_bnds" not in ds_processed.coords
    npt.assert_equal(
        ds_processed.time_bnds.dims,
        ["time", "bnds"],
    )
    # Check unit conversion and (if necessary) temporal averaging
    if frequency == "yearly":
        ds_unprocessed_avg = ds_unprocessed.climepi.yearly_average()
    else:
        ds_unprocessed_avg = ds_unprocessed
    temp_vals_expected = ds_unprocessed_avg["TREFHT"].values - 273.15
    prec_vals_expected = (
        ds_unprocessed_avg["PRECC"].values + ds_unprocessed_avg["PRECL"].values
    ) * 8.64e7
    npt.assert_allclose(
        ds_processed.temperature.values.squeeze(), temp_vals_expected.squeeze()
    )
    npt.assert_allclose(
        ds_processed.precipitation.values.squeeze(), prec_vals_expected.squeeze()
    )
    # Check attributes
    assert ds_processed.temperature.attrs["long_name"] == "Temperature"
    assert ds_processed.temperature.attrs["units"] == "°C"
    assert ds_processed.precipitation.attrs["long_name"] == "Precipitation"
    assert ds_processed.precipitation.attrs["units"] == "mm/day"
    assert ds_processed.time.attrs == {
        "bounds": "time_bnds",
        "long_name": "Time",
        "axis": "T",
    }
    assert ds_processed.lon.attrs == {
        "long_name": "Longitude",
        "units": "°E",
        "axis": "X",
        "bounds": "lon_bnds",
    }
    assert ds_processed.lat.attrs == {
        "long_name": "Latitude",
        "units": "°N",
        "axis": "Y",
        "bounds": "lat_bnds",
    }
    # Check addition of longitude and latitude bounds using known grid spacing
    npt.assert_allclose(
        ds_processed.lon_bnds.values.squeeze(), [150 - 0.625, 150 + 0.625]
    )
    npt.assert_allclose(
        ds_processed.lat_bnds.values.squeeze(), [30 - 180 / 382, 30 + 180 / 382]
    )
