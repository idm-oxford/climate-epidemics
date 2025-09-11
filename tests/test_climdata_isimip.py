"""
Unit tests for the the _isimip.py module of the climdata subpackage.

The ISIMIPDataGetter class is tested.
"""

import itertools
import pathlib
import time
from unittest.mock import patch

import geopy
import numpy as np
import numpy.testing as npt
import pandas as pd
import pooch
import pytest
import xarray as xr
import xarray.testing as xrt

import climepi.climdata
from climepi.climdata._isimip import ISIMIPDataGetter
from climepi.testing.fixtures import generate_dataset


def test_init():
    """Test the __init__ method of the ISIMIPDataGetter class."""
    data_getter = ISIMIPDataGetter(
        frequency="daily",
        subset={
            "years": [2015, 2021],
            "models": ["mri-esm2-0", "ukesm1-0-ll"],
            "locations": ["gabba", "mcg"],
            "lons": [153, 144],
            "lats": [-27, -37],
        },
        subset_check_interval=15,
    )
    for attr, value in (
        ("data_source", "isimip"),
        ("remote_open_possible", False),
        ("_frequency", "daily"),
        (
            "_subset",
            {
                "years": [2015, 2021],
                "scenarios": ["ssp126", "ssp370", "ssp585"],
                "models": ["mri-esm2-0", "ukesm1-0-ll"],
                "realizations": [0],
                "locations": ["gabba", "mcg"],
                "lons": [153, 144],
                "lats": [-27, -37],
                "lon_range": None,
                "lat_range": None,
            },
        ),
        ("_subset_check_interval", 15),
        ("_max_subset_wait_time", 20),
        ("_client_results", None),
    ):
        assert getattr(data_getter, attr) == value, f"Attribute {attr} is not {value}."


@patch("requests.Session", autospec=True)
def test_find_remote_data(mock_session):
    """Test the _find_remote_data method of the ISIMIPDataGetter class."""
    # Set up mock methods

    def mock_json():
        if (
            mock_session.return_value.get.call_args[0][0]
            == "https://data.isimip.org/api/v1/files/"
        ):
            model = "mri-esm2-0"
            next_ = "fake_url"
        elif mock_session.return_value.get.call_args[0][0] == "fake_url":
            model = "ukesm1-0-ll"
            next_ = None
        else:
            raise ValueError(
                f"Unexpected URL {mock_session.return_value.get.call_args[0][0]}."
            )
        response = {
            "results": [
                {
                    "specifiers": {
                        "model": model,
                        "start_year": start_year,
                        "end_year": end_year,
                    }
                }
                for start_year, end_year in zip(
                    [2015, 2021, 2031, 2041, 2051],
                    [2020, 2030, 2040, 2050, 2060],
                    strict=True,
                )
            ],
            "next": next_,
        }
        return response

    mock_session.return_value.get.return_value.json = mock_json

    # Set up DataGetter and run _find_remote_data

    data_getter = ISIMIPDataGetter(
        frequency="daily",
        subset={
            "years": [2015, 2021, 2046],
            "models": ["mri-esm2-0", "ukesm1-0-ll"],
        },
    )
    data_getter._find_remote_data()

    # Check results
    assert data_getter._client_results == [
        {
            "specifiers": {
                "model": model,
                "start_year": start_year,
                "end_year": end_year,
            }
        }
        for model in ["mri-esm2-0", "ukesm1-0-ll"]
        for start_year, end_year in zip(
            [2015, 2021, 2041], [2020, 2030, 2050], strict=True
        )
    ]


@patch("requests.Session", autospec=True)
@patch.object(climepi.climdata._isimip, "geocode", autospec=True)
@pytest.mark.parametrize(
    "location_mode",
    [
        "single_named",
        "multiple_named",
        "grid_lon_0_360",
        "grid_lon_180_180",
        "grid_all_lon",
        "global",
        "invalid_single",
        "invalid_multiple",
    ],
)
@pytest.mark.parametrize("times_out", [False, True])
def test_subset_remote_data(mock_geocode, mock_session, location_mode, times_out):
    """
    Test the _subset_remote_data method of the ISIMIPDataGetter class.

    Checks that the method correctly monitors the status of remote subsetting jobs.
    """
    # Set up mock methods (including simulating a delay in the subsetting process)

    subset_check_interval = 0.01
    max_subset_wait_time = 1
    if times_out:
        time_to_finish = 2
    else:
        time_to_finish = 0.5
    st = None  # start time to set below

    def mock_json():
        json = mock_session.return_value.post.call_args[1]["json"]
        paths = json["paths"]
        bbox = json["bbox"]
        id_ = f"{paths[0]}_to_{paths[-1]}_bbox_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"

        if time.time() - st > time_to_finish:
            status = "finished"
        else:
            status = "waiting"
        response = {
            "id": id_,
            "status": status,
            "job_url": f"https://files.isimip.org/api/v2/{id_}",
        }
        return response

    mock_session.return_value.post.return_value.json = mock_json

    def _mock_geocode(location):
        if location == "Los Angeles":
            lon = -118
            lat = 34
        elif location == "Melbourne":
            lon = 144
            lat = -37
        else:
            raise ValueError(f"Unexpected location {location}.")
        return geopy.Location(location, geopy.Point(latitude=lat, longitude=lon), {})

    mock_geocode.side_effect = _mock_geocode

    # Get inputs and expected bbox values used in subsetting requests

    if location_mode == "single_named":
        # Use a single named location with provided lons and lats
        locations = ["Gabba"]
        lons = [153]
        lats = [-27]
        lat_range = None
        lon_range = None
        id_suffixes_expected = ["bbox_-27_-27_153_153"]
    elif location_mode == "multiple_named":
        # Use multiple named locations with geocoding to get lons and lats
        locations = ["Los Angeles", "Melbourne"]
        lons = None
        lats = None
        lat_range = None
        lon_range = None
        id_suffixes_expected = ["bbox_34_34_-118_-118", "bbox_-37_-37_144_144"]
    elif location_mode == "grid_lon_0_360":
        # Use a grid of lon/lat values with lon in 0-360 range
        locations = None
        lons = None
        lats = None
        lat_range = (10, 60)
        lon_range = (15, 240)
        id_suffixes_expected = ["bbox_10_60_15_-120"]
    elif location_mode == "grid_lon_180_180":
        # Use a grid of lon/lat values with lon in -180-180 range
        locations = None
        lons = None
        lats = None
        lat_range = None
        lon_range = (-30, 60)
        id_suffixes_expected = ["bbox_-90_90_-30_60"]
    elif location_mode == "grid_all_lon":
        # Use a grid of lon/lat values with all longitudes
        locations = None
        lons = None
        lats = None
        lat_range = (-30, 60)
        lon_range = None
        id_suffixes_expected = ["bbox_-30_60_-180_180"]
    elif location_mode == "global":
        # Use all global data
        locations = None
        lons = None
        lats = None
        lat_range = None
        lon_range = None
        id_suffixes_expected = ["should_not_subset"]
    elif location_mode == "invalid_single":
        # Invalid location mode (lons provided but not lats)
        locations = ["Gabba"]
        lons = [153]
        lats = None
        lat_range = None
        lon_range = None
        id_suffixes_expected = "should_not_matter"
    elif location_mode == "invalid_multiple":
        # Invalid location mode with a list of locations (lats provided but not lons)
        locations = ["Los Angeles", "Melbourne"]
        lons = None
        lats = [-27, -37]
        lat_range = None
        lon_range = None
        id_suffixes_expected = "should_not_matter"
    else:
        raise ValueError(f"Unexpected location_mode {location_mode}.")

    job_ids_expected = [
        f"{x}_to_{y}_{id_suffix}"
        for id_suffix in id_suffixes_expected
        for x, y in zip([0, 300, 600, 900], [299, 599, 899, 999], strict=True)
    ]

    # Set up DataGetter

    data_getter = ISIMIPDataGetter(
        frequency="daily",
        subset={
            "years": [2015, 2021, 2046],
            "models": ["mri-esm2-0", "ukesm1-0-ll"],
            "locations": locations,
            "lons": lons,
            "lats": lats,
            "lat_range": lat_range,
            "lon_range": lon_range,
        },
        subset_check_interval=subset_check_interval,
        max_subset_wait_time=max_subset_wait_time,
    )
    client_results_in = [{"path": x} for x in range(1000)]
    data_getter._client_results = client_results_in

    # Run _subset_remote_data and check results
    st = time.time()
    if location_mode in ["invalid_single", "invalid_multiple"]:
        with pytest.raises(
            ValueError,
            match="Either both, or neither, of 'lons' and 'lats' must be provided.",
        ):
            data_getter._subset_remote_data()
    elif times_out and location_mode != "global":
        if location_mode == "multiple_named":
            match = "Subsetting for at least one location timed out."
        else:
            match = "\n".join(
                [f"https://files.isimip.org/api/v2/{x}" for x in job_ids_expected]
            )
        with pytest.raises(TimeoutError, match=match):
            data_getter._subset_remote_data()
    else:
        data_getter._subset_remote_data()
        assert (
            data_getter._client_results == client_results_in
            if location_mode == "global"
            else [{"id": x, "status": "finished"} for x in job_ids_expected]
        )

    # Check that requests sessions are closed

    if location_mode in ["global", "invalid_single", "invalid_multiple"]:
        mock_session.assert_not_called()
    elif location_mode == "multiple_named":
        assert mock_session.return_value.close.call_count == 2
    else:
        mock_session.return_value.close.assert_called_once()


@patch.object(pooch, "create", autospec=True)
@patch.object(pathlib.Path, "unlink", autospec=True)
@patch("zipfile.ZipFile", autospec=True)
@pytest.mark.parametrize("data_subsetted", [False, True])
def test_download_remote_data(mock_zipfile, mock_unlink, mock_create, data_subsetted):
    """Test the _download_remote_data method of the ISIMIPDataGetter class."""
    temp_save_dir = pathlib.Path("gully")

    if data_subsetted:
        base_url = "https://files.isimip.org/api/v1/output"
    else:
        base_url = "https://files.isimip.org"

    # Set up mock methods

    def mock_namelist():
        zip_file_name = str(mock_zipfile.call_args[0][0].as_posix()).rsplit(
            "/", maxsplit=1
        )[-1]
        namelist = [
            f"{zip_file_name.split('.')[0]}_file_{x}.nc" for x in [1, 2, 3, 4]
        ] + ["not_a_nc_file.txt"]
        return namelist

    # (Note __enter__ needed as zipfile.ZipFile is used as a context manager)
    mock_zipfile.return_value.__enter__.return_value.namelist = mock_namelist

    def mock_fetch_side_effect(*args, **kwargs):
        return temp_save_dir / args[0]

    mock_fetch = mock_create.return_value.fetch
    mock_fetch.side_effect = mock_fetch_side_effect

    # Set up DataGetter and run _download_remote_data

    data_getter = ISIMIPDataGetter()
    data_getter._temp_save_dir = temp_save_dir
    if data_subsetted:
        data_getter._client_results = [
            {"file_name": x, "file_url": base_url + x}
            for x in ["/batch_1.zip", "/batch_2.zip"]
        ]
    else:
        data_getter._client_results = [
            {"name": x, "file_url": base_url + x}
            for x in ["/file_1.nc", "/file_2.nc", "/file_3.nc"]
        ]
    data_getter._download_remote_data()

    # Check results

    if data_subsetted:
        assert data_getter._temp_file_names == [
            f"batch_{x}_file_{y}.nc" for x in [1, 2] for y in [1, 2, 3, 4]
        ]
        for batch in [1, 2]:
            mock_create.assert_any_call(
                base_url=base_url,
                path=temp_save_dir,
                registry={f"batch_{batch}.zip": None},
                retry_if_failed=5,
            )
            mock_fetch.assert_any_call(
                f"batch_{batch}.zip",
                progressbar=True,
            )
            mock_zipfile.return_value.__enter__.return_value.extractall.assert_any_call(
                path=temp_save_dir,
                members=[f"batch_{batch}_file_{y}.nc" for y in [1, 2, 3, 4]],
            )
            mock_unlink.assert_any_call(temp_save_dir / f"batch_{batch}.zip")
    else:
        assert data_getter._temp_file_names == ["file_1.nc", "file_2.nc", "file_3.nc"]
        for file_no in [1, 2, 3]:
            mock_create.assert_any_call(
                base_url=base_url,
                path=temp_save_dir,
                registry={f"file_{file_no}.nc": None},
                retry_if_failed=5,
            )
            mock_fetch.assert_any_call(
                f"file_{file_no}.nc",
                progressbar=True,
            )
        mock_zipfile.return_value.__enter__.return_value.extractall.assert_not_called()
        mock_unlink.assert_not_called()


@patch.object(xr, "open_mfdataset", autospec=True)
def test_open_temp_data(mock_open_mfdataset):
    """
    Test the _open_temp_data method of the ISIMIPDataGetter class.

    Focuses on checking the preprocessing of the opened dataset.
    """
    # Set up mock open_mfdataset method (which subsets, preprocesses and combines an
    # input dataset to simulate the opening of multiple files)

    scenarios = ["overcast", "sunny"]
    models = ["bouncer", "inswinger", "length"]
    data_vars = ["tas", "pr"]
    time_subsets = [  # split to test fix for some times not centered in middle of day
        "times1",
        "times2",
    ]
    ds_in = generate_dataset(
        data_var=data_vars,
        extra_dims={
            "scenario": scenarios,
            "model": models,
            "realization": [0],
        },
        frequency="daily",
        has_bounds=False,
    ).isel(lon=0, lat=0, drop=True)

    def mock_open_mfdataset_side_effect(paths, **kwargs):
        ds_list = []
        for (scenario, model, var, time_subset), path in zip(
            itertools.product(scenarios, models, data_vars, time_subsets),
            paths,
            strict=True,
        ):
            path = str(path.as_posix())
            assert scenario in path
            assert model in path
            assert var in path
            assert time_subset in path
            ds_curr_init = ds_in[[var]].sel(
                scenario=scenario, model=model, realization=0, drop=True
            )
            if time_subset == "times1":
                ds_curr_init = ds_curr_init.isel(time=slice(None, 3))
            else:
                ds_curr_init = ds_curr_init.isel(time=slice(3, None))
            if scenario == "overcast":
                ds_curr_init["time"] = ds_curr_init["time"] - pd.Timedelta("12h")
                ds_curr_init["time"].attrs = ds_in["time"].attrs
            ds_curr_init.encoding["source"] = path
            ds_curr = kwargs["preprocess"](ds_curr_init)
            ds_list.append(ds_curr)
        return xr.combine_by_coords(ds_list)

    mock_open_mfdataset.side_effect = mock_open_mfdataset_side_effect

    # Set up DataGetter and run _open_temp_data

    data_getter = ISIMIPDataGetter(subset={"scenarios": scenarios, "models": models})
    data_getter._temp_file_names = [
        f"{model}_r1i1p1f1_w5e5_{scenario}_{var}_{time_subset}.nc"
        for scenario, model, var, time_subset in itertools.product(
            scenarios, models, data_vars, time_subsets
        )
    ]
    data_getter._temp_save_dir = pathlib.Path("cover")

    data_getter._open_temp_data()

    # Check results

    xrt.assert_identical(data_getter._ds_temp, ds_in)
    xrt.assert_identical(data_getter._ds, ds_in)


@pytest.mark.parametrize("frequency", ["daily", "monthly", "yearly"])
def test_process_data(frequency):
    """Test the _process_data method of the ISIMIPDataGetter class."""
    # Set up unprocessed dataset

    time_lb = xr.date_range(
        start="2015-01-01", periods=730, freq="D", calendar="noleap", use_cftime=True
    )
    time_rb = xr.date_range(
        start="2015-01-02", periods=730, freq="D", calendar="noleap", use_cftime=True
    )
    time_bnds_in = xr.DataArray(  # not in unprocessed dataset
        np.array([time_lb, time_rb]).T,
        dims=("time", "bnds"),
        attrs={"xcdat_bounds": True},
    )
    time_in = time_bnds_in.mean(dim="bnds").assign_attrs(bounds="time_bnds")
    time_in.encoding = {"calendar": "noleap", "units": "days since 2000-01-01"}
    ds_unprocessed = xr.Dataset(
        data_vars={
            "tas": xr.DataArray(np.random.rand(730, 2, 2), dims=["time", "lat", "lon"]),
            "pr": xr.DataArray(np.random.rand(730, 2, 2), dims=["time", "lat", "lon"]),
        },
        coords={
            "time": time_in,
            "lat": xr.DataArray([-27, 30], dims="lat"),
            "lon": xr.DataArray([0, 150], dims="lon"),
        },
    )

    # Set up DataGetter and run _process_data

    locations = ["Brisbane"]
    lon = [150]
    lat = [-27]
    data_getter = ISIMIPDataGetter(
        frequency=frequency,
        subset={"locations": locations, "lon": lon, "lat": lat},
    )
    data_getter._ds = ds_unprocessed
    data_getter._process_data()
    ds_processed = data_getter._ds

    # Check processed dataset

    npt.assert_allclose(ds_processed.lon.values.squeeze(), [150])
    npt.assert_allclose(ds_processed.lat.values.squeeze(), [-27])
    npt.assert_allclose(
        ds_processed.lon_bnds.values.squeeze(), [150 - 0.25, 150 + 0.25]
    )
    npt.assert_allclose(
        ds_processed.lat_bnds.values.squeeze(), [-27 - 0.25, -27 + 0.25]
    )
    assert "time_bnds" in ds_processed
    assert "tas" not in ds_processed
    assert "pr" not in ds_processed
    assert ds_processed.temperature.attrs["long_name"] == "Temperature"
    assert ds_processed.temperature.attrs["units"] == "°C"
    assert ds_processed.precipitation.attrs["long_name"] == "Precipitation"
    assert ds_processed.precipitation.attrs["units"] == "mm/day"
    assert ds_processed.time.attrs["long_name"] == "Time"
    assert ds_processed.time.attrs["bounds"] == "time_bnds"
    if frequency == "daily":
        xrt.assert_equal(
            ds_processed["time_bnds"], time_bnds_in.assign_coords(time=time_in)
        )
        temperature_values_expected = (
            ds_unprocessed["tas"].isel(lat=0, lon=1).values - 273.15
        ).squeeze()
        precipitation_values_expected = (
            8.64e4 * ds_unprocessed["pr"].isel(lat=0, lon=1).values
        ).squeeze()
        npt.assert_allclose(
            ds_processed.temperature.values.squeeze(), temperature_values_expected
        )
        npt.assert_allclose(
            ds_processed.precipitation.values.squeeze(), precipitation_values_expected
        )
    elif frequency == "monthly":
        assert ds_processed.time.size == 24
    elif frequency == "yearly":
        assert ds_processed.time.size == 2
    else:
        raise ValueError(f"Unexpected frequency {frequency}.")
