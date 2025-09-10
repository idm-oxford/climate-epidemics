"""
Unit tests for the _data_getter_class module of the climdata subpackage.

The ClimateDataGetter class is tested.
"""

import itertools
import pathlib
import tempfile
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import dask.array as da
import netCDF4  # noqa (avoids warning https://github.com/pydata/xarray/issues/7259)
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt

import climepi  # noqa
from climepi.climdata._data_getter_class import CACHE_DIR, ClimateDataGetter
from climepi.testing.fixtures import generate_dataset


def test_init():
    """Test the __init__ method of the ClimateDataGetter class."""
    data_getter = ClimateDataGetter(
        frequency="daily",
        subset={"models": ["googly"], "locations": ["gabba"]},
    )
    for attr, value in (
        ("data_source", ""),
        ("remote_open_possible", False),
        ("available_years", []),
        ("available_scenarios", []),
        ("available_models", []),
        ("available_realizations", []),
        ("lon_res", None),
        ("lat_res", None),
        ("_frequency", "daily"),
        (
            "_subset",
            {
                "years": [],
                "scenarios": [],
                "models": ["googly"],
                "realizations": [],
                "locations": ["gabba"],
                "lons": None,
                "lats": None,
                "lon_range": None,
                "lat_range": None,
            },
        ),
        ("_temp_save_dir", None),
        ("_temp_file_names", None),
        ("_ds_temp", None),
        ("_save_dir", CACHE_DIR),
        ("_file_name_da", None),
        ("_file_names", None),
    ):
        assert getattr(data_getter, attr) == value, f"Attribute {attr} is not {value}."


@pytest.mark.parametrize(
    "years,year_str_expected,warning_expected",
    [
        ([2015], "2015", False),
        ([2015, 2016], "2015_to_2016", False),
        (np.arange(2015, 2200), "2015_to_2199", False),
        ([2015, 2016, 2018, 2100], "2015_2016_2018_2100", False),
        (np.arange(2015, 2200, step=2), "2015_by_2_to_2199", False),
        (
            np.append(np.arange(2015, 2026), 3000),
            "_".join([str(y) for y in np.append(np.arange(2015, 2026), 3000)]),
            True,
        ),
    ],
)
@pytest.mark.parametrize(
    "locations,lon_range,lat_range,loc_strs_expected",
    [
        (None, None, None, ("all",)),
        (["gabba"], None, None, ("gabba",)),
        (["mcg"], None, None, ("mcg",)),
        (["gabba", "mcg", "waca"], None, [8, 15], ("gabba", "mcg", "waca")),
        (None, [8, 15], None, ("lon_8_to_15",)),
        (None, None, [7, 44], ("lat_7_to_44",)),
        (None, [7.5, 72], [-6, 17.33], ("lon_7_5_to_72_lat_m6_to_17_33",)),
    ],
)
def test_file_name_da(
    years,
    year_str_expected,
    warning_expected,
    locations,
    lon_range,
    lat_range,
    loc_strs_expected,
):
    """Test the file_name_da property of the ClimateDataGetter class."""
    scenarios = ["overcast", "sunny"]
    models = ["length", "inswinger", "bouncer"]
    realizations = np.arange(1, 3)
    data_getter = ClimateDataGetter(
        subset={
            "scenarios": scenarios,
            "models": models,
            "realizations": realizations,
            "years": years,
            "locations": locations,
            "lon_range": lon_range,
            "lat_range": lat_range,
        },
    )
    data_getter.data_source = "broad"
    assert data_getter._file_name_da is None
    with pytest.warns(UserWarning) if warning_expected else nullcontext():
        result = data_getter.file_name_da
    xrt.assert_identical(data_getter._file_name_da, result)
    if locations is not None:
        location_dim_expected = np.atleast_1d(locations).tolist()
    else:
        location_dim_expected = list(loc_strs_expected)
    npt.assert_equal(result["location"].values, np.array(location_dim_expected))
    npt.assert_equal(result["scenario"].values, np.array(scenarios))
    npt.assert_equal(result["model"].values, np.array(models))
    npt.assert_equal(result["realization"].values, np.array(realizations))
    for (location, loc_str_expected), scenario, model, realization in itertools.product(
        zip(location_dim_expected, loc_strs_expected, strict=True),
        scenarios,
        models,
        realizations,
    ):
        file_name_result = result.sel(
            location=location, scenario=scenario, model=model, realization=realization
        ).item()
        file_name_expected = (
            f"broad_monthly_{year_str_expected}_{loc_str_expected}"
            + f"_{scenario}_{model}_{realization}.nc"
        )
        assert file_name_result == file_name_expected


def test_file_names():
    """Test the file_names property of the ClimateDataGetter class."""
    data_getter = ClimateDataGetter(
        frequency="monthly",
        subset={
            "scenarios": ["overcast", "sunny"],
            "models": ["length"],
            "realizations": np.arange(1, 3),
            "years": [2015, 2016, 2018, 2100],
            "locations": ["gabba", "mcg"],
        },
    )
    data_getter.data_source = "broad"
    assert data_getter._file_names is None
    result = data_getter.file_names
    assert data_getter._file_names == result
    expected = [
        "broad_monthly_2015_2016_2018_2100_" + comb + ".nc"
        for comb in [
            "gabba_overcast_length_1",
            "gabba_overcast_length_2",
            "gabba_sunny_length_1",
            "gabba_sunny_length_2",
            "mcg_sunny_length_1",
            "mcg_sunny_length_2",
            "mcg_overcast_length_1",
            "mcg_overcast_length_2",
        ]
    ]
    assert sorted(result) == sorted(expected)


@pytest.mark.parametrize("remote_open_possible", [True, False])
@pytest.mark.parametrize("download", [True, False])
@pytest.mark.parametrize("local_data_available", [True, False])
@pytest.mark.parametrize("force_remake", [True, False])
def test_get_data(remote_open_possible, download, local_data_available, force_remake):
    """Unit test for the get_data method of the ClimateDataGetter class."""
    subset = {
        "years": [2015],
        "scenarios": ["overcast"],
        "models": ["length"],
        "realizations": [1],
    }
    data_source = "warner"

    data_getter = ClimateDataGetter(subset=subset)
    data_getter.data_source = data_source
    data_getter.remote_open_possible = remote_open_possible

    def _open_local_data_side_effect():
        if not local_data_available and data_getter._ds is None:
            raise FileNotFoundError
        data_getter._ds = "variation"

    def _find_remote_data_side_effect():
        if data_getter.remote_open_possible:
            data_getter._ds = "stock"

    def _subset_remote_data_side_effect():
        if data_getter.remote_open_possible:
            data_getter._ds = "topspinner"

    def _open_temp_data_side_effect():
        data_getter._ds_temp = "googly"
        data_getter._ds = "googly"

    def _process_data_side_effect():
        data_getter._ds = "flipper"

    def _delete_temp_side_effect():
        data_getter._temp_save_dir.rmdir()
        data_getter._temp_save_dir = None
        data_getter._ds_temp = "deleted"

    data_getter._open_local_data = MagicMock(side_effect=_open_local_data_side_effect)
    data_getter._find_remote_data = MagicMock(side_effect=_find_remote_data_side_effect)
    data_getter._subset_remote_data = MagicMock(
        side_effect=_subset_remote_data_side_effect
    )
    data_getter._download_remote_data = MagicMock()
    data_getter._open_temp_data = MagicMock(side_effect=_open_temp_data_side_effect)
    data_getter._process_data = MagicMock(side_effect=_process_data_side_effect)
    data_getter._save_processed_data = MagicMock()
    data_getter._delete_temporary = MagicMock(side_effect=_delete_temp_side_effect)

    if force_remake and not download:
        with pytest.raises(ValueError):
            data_getter.get_data(download=download, force_remake=force_remake)
        call_counts_expected = {
            "_open_local_data": 0,
            "_find_remote_data": 0,
            "_subset_remote_data": 0,
            "_download_remote_data": 0,
            "_open_temp_data": 0,
            "_process_data": 0,
            "_save_processed_data": 0,
            "_delete_temporary": 0,
        }
    elif not local_data_available and not download and not remote_open_possible:
        with pytest.raises(ValueError):
            data_getter.get_data(download=download, force_remake=force_remake)
        call_counts_expected = {
            "_open_local_data": 1,
            "_find_remote_data": 0,
            "_subset_remote_data": 0,
            "_download_remote_data": 0,
            "_open_temp_data": 0,
            "_process_data": 0,
            "_save_processed_data": 0,
            "_delete_temporary": 0,
        }
    else:
        result = data_getter.get_data(download=download, force_remake=force_remake)
        if force_remake:
            assert result == "variation"
            assert data_getter._ds == "variation"
            assert data_getter._ds_temp == "deleted"
            call_counts_expected = {
                "_open_local_data": 1,
                "_find_remote_data": 1,
                "_subset_remote_data": 1,
                "_download_remote_data": 1,
                "_open_temp_data": 1,
                "_process_data": 1,
                "_save_processed_data": 1,
                "_delete_temporary": 1,
            }
        elif local_data_available:
            assert result == "variation"
            assert data_getter._ds == "variation"
            assert data_getter._ds_temp is None
            call_counts_expected = {
                "_open_local_data": 1,
                "_find_remote_data": 0,
                "_subset_remote_data": 0,
                "_download_remote_data": 0,
                "_open_temp_data": 0,
                "_process_data": 0,
                "_save_processed_data": 0,
                "_delete_temporary": 0,
            }
        elif download:
            assert result == "variation"
            assert data_getter._ds == "variation"
            assert data_getter._ds_temp == "deleted"
            call_counts_expected = {
                "_open_local_data": 2,
                "_find_remote_data": 1,
                "_subset_remote_data": 1,
                "_download_remote_data": 1,
                "_open_temp_data": 1,
                "_process_data": 1,
                "_save_processed_data": 1,
                "_delete_temporary": 1,
            }
        else:
            assert result == "flipper"
            assert data_getter._ds == "flipper"
            assert data_getter._ds_temp is None
            call_counts_expected = {
                "_open_local_data": 1,
                "_find_remote_data": 1,
                "_subset_remote_data": 1,
                "_download_remote_data": 0,
                "_open_temp_data": 0,
                "_process_data": 1,
                "_save_processed_data": 0,
                "_delete_temporary": 0,
            }
    for method, count_expected in call_counts_expected.items():
        assert getattr(data_getter, method).call_count == count_expected, (
            f"Method {method} called {getattr(data_getter, method).call_count} "
            f"times, expected {count_expected}."
        )
    assert data_getter._temp_save_dir is None


def test_open_local_data():
    """Test the _open_local_data method of the ClimateDataGetter class."""
    scenarios = ["overcast", "sunny"]
    models = ["bouncer", "inswinger", "length"]
    realizations = np.arange(1, 3)
    locations = ["mcg", "gabba"]
    ds_in = generate_dataset(
        data_var="temperature",
        extra_dims={
            "scenario": scenarios,
            "model": models,
            "realization": realizations,
            "location": locations,
        },
    )
    ds_in["temperature"].attrs["units"] = "deg_C"

    def _mock_xr_open_mfdataset(file_name_list, **kwargs):
        _locations = [str(file_name).split("_")[-4] for file_name in file_name_list]
        _scenarios = [str(file_name).split("_")[-3] for file_name in file_name_list]
        _models = [str(file_name).split("_")[-2] for file_name in file_name_list]
        _realizations = [
            int(str(file_name).split("_")[-1].split(".")[0])
            for file_name in file_name_list
        ]
        _ds_list = [
            ds_in.sel(scenario=[S], model=[M], realization=[R], location=[L]).chunk(
                {"scenario": 1, "model": 1, "realization": 1, "location": 1}
            )
            for S, M, R, L in zip(
                _scenarios, _models, _realizations, _locations, strict=True
            )
        ]
        # Note data_vars="minimal" ensures correct bounds handling
        assert kwargs["data_vars"] == "minimal"
        return xr.combine_by_coords(_ds_list, data_vars="minimal")

    data_getter = ClimateDataGetter(
        subset={
            "scenarios": scenarios,
            "models": models,
            "realizations": realizations,
            "years": [2015, 2016, 2018, 2100],
            "locations": locations,
        },
    )
    data_getter.data_source = "watto"

    with patch("xarray.open_mfdataset", _mock_xr_open_mfdataset):
        data_getter._open_local_data()

    ds_out = data_getter._ds
    xrt.assert_identical(ds_out, ds_in)

    # Check time bounds have been loaded into memory, but not other variables (the time
    # bounds are loaded to avoid encoding issues)
    assert isinstance(ds_out.time_bnds.data, np.ndarray)
    assert all(
        isinstance(ds_out[var].data, da.Array)
        for var in ["temperature", "lon_bnds", "lat_bnds"]
    )


def test_find_remote_data():
    """
    Test the _find_remote_data method of the ClimateDataGetter.

    The method needs to be implemented in subclasses so should raise a
    NotImplementedError.
    """
    data_getter = ClimateDataGetter()
    with pytest.raises(NotImplementedError):
        data_getter._find_remote_data()


def test_subset_remote_data():
    """
    Test the _subset_remote_data method of the ClimateDataGetter.

    The method needs to be implemented in subclasses so should raise a
    NotImplementedError.
    """
    data_getter = ClimateDataGetter()
    with pytest.raises(NotImplementedError):
        data_getter._subset_remote_data()


def test_download_remote_data():
    """
    Test the _download_remote_data method of the ClimateDataGetter.

    The method needs to be implemented in subclasses so should raise a
    NotImplementedError.
    """
    data_getter = ClimateDataGetter()
    with pytest.raises(NotImplementedError):
        data_getter._download_remote_data()


@patch("xarray.open_mfdataset", autospec=True)
def test_open_temp_data(mock_xr_open_mfdataset):
    """Test the _open_temp_data method of the ClimateDataGetter class."""
    data_getter = ClimateDataGetter()
    data_getter._temp_save_dir = pathlib.Path("not/a/real/path")
    data_getter._temp_file_names = [
        "temporary_1.nc",
        "temporary_2.nc",
        "temporary_3.nc",
    ]

    open_paths_expected = [
        pathlib.Path("not/a/real/path/temporary_1.nc"),
        pathlib.Path("not/a/real/path/temporary_2.nc"),
        pathlib.Path("not/a/real/path/temporary_3.nc"),
    ]

    data_getter._open_temp_data(data_vars="all")
    mock_xr_open_mfdataset.assert_called_once_with(
        open_paths_expected,
        data_vars="all",
        chunks={"realization": 1, "model": 1, "scenario": 1, "location": 1, "time": -1},
        coords="minimal",
        compat="override",
    )

    assert data_getter._ds_temp == mock_xr_open_mfdataset.return_value
    assert data_getter._ds == mock_xr_open_mfdataset.return_value.copy.return_value


@pytest.mark.parametrize(
    "location_mode",
    ["single_named", "multiple_named", "single_lat_lon", "grid_lat_lon"],
)
@pytest.mark.parametrize("lon_res_set", [True, False])
def test_process_data(location_mode, lon_res_set):
    """
    Test the _process_data method of the ClimateDataGetter class.

    Focus on checking that longitude/latitude bounds correctly are added to the
    dataset.
    """
    # Set up the input dataset
    lon_res = 0.8
    lat_res = 0.15
    if location_mode in ["single_named", "single_lat_lon"]:
        no_lons = 1
        no_lats = 1
    elif location_mode in ["multiple_named", "grid_lat_lon"]:
        no_lons = 4
        no_lats = 4
    else:
        raise ValueError(f"Unknown location_mode: {location_mode}")
    if location_mode in ["single_lat_lon", "grid_lat_lon"]:
        lon_vals = 179 + lon_res * np.arange(no_lons)
        lat_vals = -20 + lat_res * np.arange(no_lats)
        ds_in = xr.Dataset(
            data_vars={
                "delivery": xr.DataArray(
                    np.random.rand(no_lons, no_lats), dims=["lon", "lat"]
                )
            },
            coords={
                "lon": xr.DataArray(lon_vals, dims="lon"),
                "lat": xr.DataArray(lat_vals, dims="lat"),
            },
        )
    elif location_mode in ["single_named", "multiple_named"]:
        assert no_lons == no_lats
        lon_vals = np.random.choice(np.arange(179), no_lons, replace=False)
        lat_vals = np.random.choice(np.arange(-20, 40), no_lats, replace=False)
        location_vals = np.random.choice(
            ["gabba", "mcg", "waca", "scg"], no_lons, replace=False
        )
        ds_in = xr.Dataset(
            data_vars={
                "delivery": xr.DataArray(np.random.rand(no_lons), dims=["location"])
            },
            coords={
                "location": location_vals,
                "lon": xr.DataArray(lon_vals, dims=["location"]),
                "lat": xr.DataArray(lat_vals, dims=["location"]),
            },
        )
    else:
        raise ValueError(f"Unknown location_mode: {location_mode}")
    ds_in["lon"].attrs = {"long_name": "Longitude", "units": "degrees_east"}
    ds_in["lat"].attrs = {"long_name": "Latitude", "units": "degrees_north"}
    # Run the _process_data method
    data_getter = ClimateDataGetter()
    data_getter.lat_res = lat_res
    if lon_res_set:
        data_getter.lon_res = lon_res
    data_getter._ds = ds_in
    data_getter._process_data()
    ds_out = data_getter._ds
    # Check the output dataset
    lat_bnds_vals_expected = np.stack(
        [lat_vals - lat_res / 2, lat_vals + lat_res / 2], axis=1
    )
    lon_bnds_vals_expected = np.stack(
        [lon_vals - lon_res / 2, lon_vals + lon_res / 2], axis=1
    )
    npt.assert_allclose(ds_out["lat_bnds"], lat_bnds_vals_expected)
    if lon_res_set or location_mode == "grid_lat_lon":
        npt.assert_allclose(ds_out["lon_bnds"], lon_bnds_vals_expected)
        xrt.assert_equal(ds_out.drop_vars(["lon_bnds", "lat_bnds"]), ds_in)
    else:
        assert "lon_bnds" not in ds_out
        xrt.assert_equal(ds_out.drop_vars("lat_bnds"), ds_in)
    # Check attributes of the output dataset
    assert ds_out["lat"].attrs == {
        "long_name": "Latitude",
        "units": "°N",
        "bounds": "lat_bnds",
    }
    assert ds_out["lon"].attrs["units"] == "°E"
    assert ds_out["lon"].attrs["long_name"] == "Longitude"
    if lon_res_set:
        assert ds_out["lon"].attrs["bounds"] == "lon_bnds"
        assert ds_out["lon_bnds"].attrs == {}
    elif location_mode == "grid_lat_lon":
        assert ds_out["lon"].attrs["bounds"] == "lon_bnds"
        assert ds_out["lon_bnds"].attrs == {"xcdat_bounds": "True"}
    else:
        assert "bounds" not in ds_out["lon"].attrs


@patch.object(pathlib.Path, "mkdir", autospec=True)
@patch.object(xr, "save_mfdataset", autospec=True)
@pytest.mark.parametrize("named_locations", [True, False])
def test_save_processed_data(mock_save_mfdataset, mock_mkdir, named_locations):
    """Test the _save_processed_data method of the ClimateDataGetter class."""
    scenarios = ["overcast", "sunny"]
    models = ["inswinger", "length"]
    realizations = np.arange(1, 3)
    ds = generate_dataset(
        data_var="temperature",
        extra_dims={
            "scenario": scenarios,
            "model": models,
            "realization": realizations,
        },
    )
    if named_locations:
        locations = ["lords", "gabba"]
        ds = ds.climepi.sel_geo(locations)

    save_dir = "outside/edge"

    data_getter = ClimateDataGetter(
        frequency="monthly",
        subset={
            "scenarios": scenarios,
            "models": models,
            "realizations": realizations,
            "years": [2015, 2016, 2018, 2100],
            "locations": locations if named_locations else None,
        },
        save_dir=save_dir,
    )
    data_getter.data_source = "broad"
    data_getter._ds = ds

    data_getter._save_processed_data()

    mock_mkdir.assert_called_once_with(
        pathlib.Path(save_dir), parents=True, exist_ok=True
    )

    if named_locations:
        mock_save_mfdataset.assert_called_once_with(
            [
                ds.sel(scenario=[s], model=[m], realization=[r], location=[ll])
                for s, m, r, ll in itertools.product(
                    scenarios, models, realizations, locations
                )
            ],
            [
                pathlib.Path(
                    "outside/edge/broad_monthly_2015_2016_2018_2100_"
                    f"{ll}_{s}_{m}_{r}.nc"
                )
                for s, m, r, ll in itertools.product(
                    scenarios, models, realizations, locations
                )
            ],
            compute=False,
        )
    else:
        mock_save_mfdataset.assert_called_once_with(
            [
                ds.sel(scenario=[s], model=[m], realization=[r])
                for s, m, r in itertools.product(scenarios, models, realizations)
            ],
            [
                pathlib.Path(
                    f"outside/edge/broad_monthly_2015_2016_2018_2100_all_{s}_{m}_{r}.nc"
                )
                for s, m, r in itertools.product(scenarios, models, realizations)
            ],
            compute=False,
        )


def test_delete_temporary():
    """Test the _delete_temporary method of the ClimateDataGetter class."""
    temp_save_dir = pathlib.Path(tempfile.mkdtemp(suffix="_climepi_test"))
    temp_file_names = [
        "temporary_0.nc",
        "temporary_1.nc",
        "temporary_2.nc",
    ]
    temp_file_paths = [temp_save_dir / f for f in temp_file_names]
    for file_no, file_path in enumerate(temp_file_paths):
        ds = xr.Dataset(
            data_vars={"delivery": (["number"], np.random.rand(1))},
            coords={"number": ("number", [file_no])},
        )
        ds.to_netcdf(file_path)
    data_getter = ClimateDataGetter()
    data_getter._temp_save_dir = temp_save_dir
    data_getter._temp_file_names = temp_file_names
    data_getter._ds_temp = xr.open_mfdataset(temp_file_paths)

    assert any(temp_save_dir.iterdir())
    data_getter._delete_temporary()
    assert not temp_save_dir.is_dir()
    assert data_getter._temp_save_dir is None
    assert data_getter._temp_file_names is None
    assert data_getter._ds_temp is None
