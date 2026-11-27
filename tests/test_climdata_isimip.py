"""
Unit tests for the the _isimip.py module of the climdata subpackage.

The ISIMIPDataGetter class is tested.
"""

import itertools
import pathlib
import shutil
import zipfile
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
    """
    Test the _find_remote_data method of the ISIMIPDataGetter class.

    Checks that daily files are collected across result pages and filtered to the
    requested years.
    """
    session = mock_session.return_value.__enter__.return_value
    pages = [
        {
            "results": [
                {
                    "specifiers": {
                        "climate_scenario": "ssp126",
                        "climate_forcing": model,
                        "climate_variable": variable,
                        "start_year": start,
                        "end_year": end,
                    }
                }
                for variable, (start, end) in itertools.product(
                    ("tas", "pr"), ((2015, 2020), (2021, 2030), (2031, 2040))
                )
            ],
            "next": next_url,
        }
        for model, next_url in [("mri-esm2-0", "next_page"), ("ukesm1-0-ll", None)]
    ]
    session.get.return_value.json.side_effect = pages
    getter = ISIMIPDataGetter(
        subset={
            "years": [2015, 2021],
            "models": ["mri-esm2-0", "ukesm1-0-ll"],
            "scenarios": ["ssp126"],
        },
    )
    getter._find_remote_data()

    assert getter._client_results == [
        result
        for page in pages
        for result in page["results"]
        if result["specifiers"]["end_year"] != 2040
    ]
    first, second = session.get.call_args_list
    assert first.args == ("https://data.isimip.org/api/v1/files/",)
    assert first.kwargs["params"]["time_step"] == "daily"
    assert second.args == ("next_page",)


@patch("requests.Session", autospec=True)
@patch("climepi._core.geocode", autospec=True)
@pytest.mark.parametrize(
    ("subset", "operations"),
    [
        (
            {"locations": ["Gabba"], "lons": [153], "lats": [-27]},
            [{"operation": "cutout_point", "point": [-26.75, 153.25]}],
        ),
        (
            {"locations": ["Los Angeles", "Melbourne"]},
            [
                {"operation": "cutout_point", "point": [34.25, -117.75]},
                {"operation": "cutout_point", "point": [-36.75, 144.25]},
            ],
        ),
        (
            {"lon_range": (15, 240), "lat_range": (10, 60)},
            [{"operation": "cutout_bbox", "bbox": [15.25, -120.25, 10.25, 59.75]}],
        ),
        (
            {"lat_range": (-30, 60), "lon_range": (0, 360)},
            [{"operation": "cutout_bbox", "bbox": [-179.75, 179.75, -29.75, 59.75]}],
        ),
        (
            {"lon_range": (0.25000001, 0.24999999), "lat_range": (49, 51)},
            [{"operation": "cutout_bbox", "bbox": [0.75, -0.25, 49.25, 50.75]}],
        ),
        ({}, []),
    ],
)
@pytest.mark.parametrize("status", ["finished", "timeout", "failed"])
def test_subset_remote_data(mock_geocode, mock_session, subset, operations, status):
    """
    Test the _subset_remote_data method of the ISIMIPDataGetter class.

    Checks that each subsetting job is submitted once and then polled at its status URL.
    """
    mock_geocode.side_effect = [
        geopy.Location("Los Angeles", geopy.Point(34, -118), {}),
        geopy.Location("Melbourne", geopy.Point(-37, 144), {}),
    ]
    session = mock_session.return_value.__enter__.return_value
    payloads = [
        {"paths": paths, "operations": [operation]}
        for operation in operations
        for paths in [["%02d" % i for i in range(32)], ["32"]]
    ]
    endpoint = "https://files.isimip.org/api/v2"
    job_urls = [f"{endpoint}/{i}" for i in range(len(payloads))]
    pending_jobs = [{"status": "queued", "job_url": url} for url in job_urls]
    final_jobs = [
        {"status": status, "job_url": url, "file_url": f"{url}.zip"} for url in job_urls
    ]
    session.post.return_value.json.side_effect = pending_jobs
    session.get.return_value.json.side_effect = final_jobs
    getter = ISIMIPDataGetter(
        subset=subset,
        subset_check_interval=0,
        max_subset_wait_time=0 if status == "timeout" else 20,
    )
    catalog_results = [{"path": f"{i:02d}"} for i in range(33)]
    getter._client_results = catalog_results
    if operations and status == "timeout":
        with pytest.raises(TimeoutError, match=job_urls[-1]) as exc:
            getter._subset_remote_data()
        assert "subset_method='local'" in str(exc.value)
        session.get.assert_not_called()
    elif operations and status == "failed":
        with pytest.raises(RuntimeError, match=job_urls[0]):
            getter._subset_remote_data()
    else:
        getter._subset_remote_data()
        assert getter._client_results == (final_jobs if operations else catalog_results)
    assert [call.kwargs["json"] for call in session.post.call_args_list] == payloads
    assert all(call.args == (endpoint,) for call in session.post.call_args_list)
    if operations and status != "timeout":
        assert [call.args[0] for call in session.get.call_args_list] == job_urls


@pytest.mark.parametrize(
    "subset",
    [
        {"locations": ["A"], "lons": [1]},
        {"locations": ["A", "B"], "lons": [1, 2], "lats": [3, None]},
    ],
)
def test_subset_remote_data_incomplete_coords(subset):
    """Test _subset_remote_data with incomplete location coordinates."""
    getter = ISIMIPDataGetter(subset=subset)
    with pytest.raises(ValueError, match="must also be lists|both .* must be provided"):
        getter._subset_remote_data()


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
        subset={"locations": locations, "lons": lon, "lats": lat},
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


@pytest.mark.parametrize(
    ("source", "model", "scenario", "year", "filename_pattern"),
    [
        ("isimip3a", "gswp3-w5e5", "counterclim", 2019, "gswp3-w5e5_counterclim_{var}"),
        ("isimip4a", "gswp3-era5", "obsclim", 2022, "gswp3-era5_obsclim_{var}"),
        (
            "isimip4b",
            "ec-earth3-esm-1-1",
            "esm-scen7-h",
            2022,
            "ec-earth3-esm-1-1_r1i1p1f1_era5_esm-scen7-h_{var}",
        ),
    ],
)
def test_retrieval_subset_methods(
    source, model, scenario, year, filename_pattern, tmp_path, monkeypatch
):
    """
    Test retrieval with local and remote subsetting through get_climate_data.

    Checks that both methods give the same result for a range crossing the
    antimeridian (which NCO cutouts return with the western segment first).
    """
    paths = []
    cutout_archive = tmp_path / "cutout.zip"
    cutout_dir = tmp_path / "cutout"
    cutout_dir.mkdir()
    time_coord = xr.date_range(f"{year}-01-01T12:00:00", periods=3, use_cftime=True)
    for var in ("tas", "pr"):
        path = tmp_path / (filename_pattern.format(var=var) + "_global_daily.nc")
        values = np.arange(36, dtype=float).reshape(3, 3, 4)
        grid = xr.Dataset(
            {var: (("time", "lat", "lon"), values)},
            coords={
                "lon": ("lon", [-179.75, -0.25, 0.25, 179.75], {"axis": "X"}),
                "lat": ("lat", [1.25, 0.75, 0.25], {"axis": "Y"}),
                "time": ("time", time_coord, {"axis": "T"}),
            },
        )
        grid.time.encoding["units"] = f"hours since {year}-01-01"
        grid.to_netcdf(path)
        paths.append(path)

        # NCO writes the western segment first for ranges crossing the dateline.
        for index, time_slice in enumerate((slice(0, 1), slice(1, None))):
            cutout_path = cutout_dir / f"{path.stem}_{index}.nc"
            grid.isel(lon=[3, 0], lat=[1, 2], time=time_slice).to_netcdf(cutout_path)
            with zipfile.ZipFile(cutout_archive, "a") as archive:
                archive.write(cutout_path, arcname=cutout_path.name)

    def find_data(getter):
        getter._client_results = [{"file_url": str(path)} for path in paths]

    def download_file(getter, url, known_hash):
        return pathlib.Path(shutil.copy(url, getter._temp_save_dir))

    monkeypatch.setattr(ISIMIPDataGetter, "_find_remote_data", find_data)
    monkeypatch.setattr(ISIMIPDataGetter, "_download_file", download_file)
    subset = {
        "years": [year],
        "lon_range": (179, -179),
        "lat_range": (0, 1),
        "scenarios": [scenario],
        "models": [model],
    }
    save_dir = tmp_path / "processed"

    with patch.object(ISIMIPDataGetter, "_subset_remote_data", autospec=True) as remote:
        with climepi.climdata.get_climate_data(
            source, "daily", subset, save_dir, subset_method="local"
        ) as ds:
            assert ds.lon.values.tolist() == [-179.75, 179.75]
            assert ds.lat.values.tolist() == [0.75, 0.25]
            assert ds.scenario.values.tolist() == [scenario]
            assert ds.model.values.tolist() == [model]
            npt.assert_allclose(
                ds.temperature.values.squeeze()[0],
                [[4, 7], [8, 11]] - np.float64(273.15),
            )
            expected = ds.load()
        with climepi.climdata.get_climate_data(
            source, "daily", subset, save_dir
        ) as cached:
            xrt.assert_identical(cached, expected)
        remote.assert_not_called()

    def subset_remote_data(getter):
        getter._client_results = [{"file_url": str(cutout_archive)}]

    monkeypatch.setattr(ISIMIPDataGetter, "_subset_remote_data", subset_remote_data)
    with climepi.climdata.get_climate_data(
        source, "daily", subset, tmp_path / "processed_remote"
    ) as remote:
        xrt.assert_identical(remote, expected)
