"""
Unit tests for the ClimateDataGetter class in the _data_getter_class module of the
climdata subpackage.
"""

import itertools
import logging
import pathlib
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt

from climepi.climdata._data_getter_class import CACHE_DIR, ClimateDataGetter
from climepi.testing.fixtures import generate_dataset


def test_init():
    """
    Test the __init__ method of the ClimateDataGetter class.
    """
    data_getter = ClimateDataGetter(
        frequency="daily",
        subset={"models": ["googly"], "location": "gabba"},
    )
    for attr, value in (
        ("data_source", None),
        ("remote_open_possible", False),
        ("available_years", None),
        ("available_scenarios", None),
        ("available_models", None),
        ("available_realizations", None),
        ("lon_res", None),
        ("lat_res", None),
        ("_frequency", "daily"),
        (
            "_subset",
            {
                "years": None,
                "scenarios": None,
                "models": ["googly"],
                "realizations": None,
                "location": "gabba",
                "lon_range": None,
                "lat_range": None,
            },
        ),
        ("_temp_save_dir", CACHE_DIR / "temp"),
        ("_temp_file_names", None),
        ("_ds_temp", None),
        ("_save_dir", CACHE_DIR),
        ("_file_name_dict", None),
        ("_file_names", None),
    ):
        assert getattr(data_getter, attr) == value, f"Attribute {attr} is not {value}."


@pytest.mark.parametrize(
    "years,year_str_expected,warning",
    [
        ([2015], "2015_to_2015", False),
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
    "location,lon_range,lat_range,loc_str_expected",
    [
        ("gabba", None, None, "gabba"),
        ("Trent Bridge", [8, 15], None, "Trent_Bridge"),
        (None, [8, 15], None, "lon_8_to_15"),
        (None, None, [7, 44], "lat_7_to_44"),
        (None, [7, 72], [6, 17], "lon_7_to_72_lat_6_to_17"),
    ],
)
def test_file_name_dict(
    years, year_str_expected, warning, location, lon_range, lat_range, loc_str_expected
):
    """
    Test the _file_name_dict method of the ClimateDataGetter class.
    """
    scenarios = ["overcast", "sunny"]
    models = ["length", "inswinger", "bouncer"]
    realizations = np.arange(1, 3)
    data_getter = ClimateDataGetter(
        subset={
            "scenarios": scenarios,
            "models": models,
            "realizations": realizations,
            "years": years,
            "location": location,
            "lon_range": lon_range,
            "lat_range": lat_range,
        },
    )
    data_getter.data_source = "broad"
    assert data_getter._file_name_dict is None
    if warning:
        with pytest.warns(UserWarning):
            result = data_getter.file_name_dict
    else:
        result = data_getter.file_name_dict
    assert data_getter._file_name_dict == result
    for scenario, model, realization in itertools.product(
        scenarios, models, realizations
    ):
        file_name_result = result[scenario][model][realization]
        file_name_expected = (
            f"broad_monthly_{year_str_expected}_{loc_str_expected}"
            + f"_{scenario}_{model}_{realization}.nc"
        )
        assert file_name_result == file_name_expected, (
            f"File name is {file_name_result}, expected filename is "
            f"{file_name_expected}."
        )


def test_file_names():
    """
    Test the file_names method of the ClimateDataGetter class.
    """
    data_getter = ClimateDataGetter(
        frequency="monthly",
        subset={
            "scenarios": ["overcast", "sunny"],
            "models": ["length", "inswinger"],
            "realizations": np.arange(1, 3),
            "years": [2015, 2016, 2018, 2100],
            "location": "gabba",
        },
    )
    data_getter.data_source = "broad"
    assert data_getter._file_names is None
    result = data_getter.file_names
    assert data_getter._file_names == result
    expected = [
        "broad_monthly_2015_2016_2018_2100_gabba_" + comb + ".nc"
        for comb in [
            "overcast_length_1",
            "overcast_length_2",
            "overcast_inswinger_1",
            "overcast_inswinger_2",
            "sunny_length_1",
            "sunny_length_2",
            "sunny_inswinger_1",
            "sunny_inswinger_2",
        ]
    ]
    assert sorted(result) == sorted(expected)


class TestGetData:
    """
    Class to test the get_data method of the ClimateDataGetter class.
    """

    subset = {
        "years": [2015],
        "scenarios": ["overcast"],
        "models": ["length"],
        "realizations": [1],
    }
    data_source = "warner"

    def test_get_data_already_downloaded(self):
        """
        Test the get_data method of the ClimateDataGetter class retrieved locally
        available data if already downloaded (_open_local_data is tested in detail
        separately).
        """
        data_getter = ClimateDataGetter(subset=self.subset)
        data_getter.data_source = self.data_source

        def _side_effect():
            data_getter._ds = ["flipper"]

        data_getter._open_local_data = MagicMock(side_effect=_side_effect)
        result = data_getter.get_data()
        assert result == ["flipper"]
        assert data_getter._ds == ["flipper"]

    @pytest.mark.parametrize("remote_open_possible", [True, False])
    @pytest.mark.parametrize("download", [True, False])
    def test_get_data(self, remote_open_possible, download):
        """
        Test the get_data method of the ClimateDataGetter works correctly when data is
        not already downloaded for both download=True and download=False cases, and when
        remote_open_possible is True and False.
        """

        data_getter = ClimateDataGetter(subset=self.subset)
        data_getter.data_source = self.data_source
        data_getter.remote_open_possible = remote_open_possible

        def _open_local_data_side_effect():
            if data_getter._ds is None:
                raise FileNotFoundError
            data_getter._ds = ["variation"]

        def _find_remote_data_side_effect():
            if data_getter.remote_open_possible:
                data_getter._ds = ["stock"]

        def _subset_remote_data_side_effect():
            if data_getter.remote_open_possible:
                data_getter._ds = ["topspinner"]

        def _open_temp_data_side_effect():
            data_getter._ds_temp = ["googly"]
            data_getter._ds = ["googly"]

        def _process_data_side_effect():
            data_getter._ds = ["flipper"]

        data_getter._open_local_data = MagicMock(
            side_effect=_open_local_data_side_effect
        )
        data_getter._find_remote_data = MagicMock(
            side_effect=_find_remote_data_side_effect
        )
        data_getter._subset_remote_data = MagicMock(
            side_effect=_subset_remote_data_side_effect
        )
        data_getter._download_remote_data = MagicMock()
        data_getter._open_temp_data = MagicMock(side_effect=_open_temp_data_side_effect)
        data_getter._process_data = MagicMock(side_effect=_process_data_side_effect)
        data_getter._save_processed_data = MagicMock()
        data_getter._delete_temporary = MagicMock()
        if not download and not remote_open_possible:
            with pytest.raises(ValueError):
                data_getter.get_data(download=download)
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
            result = data_getter.get_data(download=download)
            if download:
                assert result == ["variation"]
                assert data_getter._ds == ["variation"]
                assert data_getter._ds_temp == ["googly"]
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
                assert result == ["flipper"]
                assert data_getter._ds == ["flipper"]
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


def test_open_local_data():
    """
    Test the _open_local_data method of the ClimateDataGetter class.
    """
    scenarios = ["overcast", "sunny"]
    models = ["bouncer", "inswinger", "length"]
    realizations = np.arange(1, 3)
    ds_in = generate_dataset(
        data_var="temperature",
        extra_dims={
            "scenario": scenarios,
            "model": models,
            "realization": realizations,
        },
    )
    ds_in["temperature"].attrs["units"] = "deg_C"
    ds_in["temperature"].values = np.random.rand(*ds_in["temperature"].shape)

    def _mock_xcdat_open_mfdataset(file_name_list, **kwargs):
        _scenarios = [str(file_name).split("_")[-3] for file_name in file_name_list]
        _models = [str(file_name).split("_")[-2] for file_name in file_name_list]
        _realizations = [
            int(str(file_name).split("_")[-1].split(".")[0])
            for file_name in file_name_list
        ]
        _ds_list = [
            ds_in.sel(scenario=[s], model=[m], realization=[r]).chunk(
                {"scenario": 1, "model": 1, "realization": 1}
            )
            for s, m, r in zip(_scenarios, _models, _realizations)
        ]
        # Note data_vars="minimal" is default in xcdat.open_mfdataset but not in
        # xarray.open_mfdataset (the xcdat version is used in _open_local_data, with
        # data_vars="minimal" ensuring correct bounds handling)
        return xr.combine_by_coords(_ds_list, data_vars="minimal")

    data_getter = ClimateDataGetter(
        subset={
            "scenarios": scenarios,
            "models": models,
            "realizations": realizations,
            "years": [2015, 2016, 2018, 2100],
            "location": "gabba",
        },
    )
    data_getter.data_source = "watto"

    with patch("xarray.open_mfdataset", _mock_xcdat_open_mfdataset):
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
    Test that the _find_remote_data method of the ClimateDataGetter (which needs to be
    implemented in subclasses) raises a NotImplementedError.
    """
    data_getter = ClimateDataGetter()
    with pytest.raises(NotImplementedError):
        data_getter._find_remote_data()


def test_subset_remote_data():
    """
    Test that the _subset_remote_data method of the ClimateDataGetter (which needs to be
    implemented in subclasses) raises a NotImplementedError.
    """
    data_getter = ClimateDataGetter()
    with pytest.raises(NotImplementedError):
        data_getter._subset_remote_data()


def test_download_remote_data():
    """
    Test that the _download_remote_data method of the ClimateDataGetter (which needs to
    be implemented in subclasses) raises a NotImplementedError.
    """
    data_getter = ClimateDataGetter()
    with pytest.raises(NotImplementedError):
        data_getter._download_remote_data()


@pytest.mark.parametrize("include_data_vars_kwarg", [True, False])
def test_open_temp_data(include_data_vars_kwarg):
    """
    Test the _open_temp_data method of the ClimateDataGetter class.
    """
    data_getter = ClimateDataGetter()
    data_getter._temp_save_dir = pathlib.Path("not/a/real/path")
    data_getter._temp_file_names = [
        "temporary_1.nc",
        "temporary_2.nc",
        "temporary_3.nc",
    ]

    def _mock_xr_open_mfdataset(file_path_list, **kwargs):
        return [str(x) + "_" + kwargs["data_vars"] for x in file_path_list]

    if include_data_vars_kwarg:
        with patch("xarray.open_mfdataset", _mock_xr_open_mfdataset):
            data_getter._open_temp_data(data_vars="all")
        expected = [
            "not/a/real/path/temporary_1.nc_all",
            "not/a/real/path/temporary_2.nc_all",
            "not/a/real/path/temporary_3.nc_all",
        ]
    else:
        with patch("xarray.open_mfdataset", _mock_xr_open_mfdataset):
            data_getter._open_temp_data()
        expected = [
            "not/a/real/path/temporary_1.nc_minimal",
            "not/a/real/path/temporary_2.nc_minimal",
            "not/a/real/path/temporary_3.nc_minimal",
        ]
    assert data_getter._ds_temp == expected
    assert data_getter._ds == expected


class TestProcessData:
    """
    Class to test the _process_data method of the ClimateDataGetter class.
    """

    def test_process_data_main(self):
        """
        Main test for the _process_data method of the ClimateDataGetter class. Focuses
        on checking that longitude/latitude bounds are added to the dataset, and that
        longitude values are converted to the range -180 to 180 if necessary.
        """
        lon_res = 0.8
        lat_res = 0.15
        ds_in = xr.Dataset(
            data_vars={
                "delivery": xr.DataArray(np.random.rand(4, 5), dims=["lon", "lat"])
            },
            coords={
                "lon": xr.DataArray(179 + lon_res * np.arange(4), dims="lon"),
                "lat": xr.DataArray(-20 + lat_res * np.arange(5), dims="lat"),
            },
        )
        ds_in["lon"].attrs = {"long_name": "Longitude", "units": "degrees_east"}
        ds_in["lat"].attrs = {"long_name": "Latitude", "units": "degrees_north"}
        # Case where lon_res and lat_res are not set, so xcdat add_missing_bounds is
        # used to add bounds to the longitude and latitude coordinates
        data_getter1 = ClimateDataGetter()
        data_getter1._ds = ds_in
        data_getter1._process_data()
        ds_out1 = data_getter1._ds
        assert "lon_bnds" in ds_out1 and "lat_bnds" in ds_out1
        npt.assert_allclose(
            ds_out1["lon"].values, np.array([-179.4, -178.6, 179, 179.8])
        )
        assert ds_out1["lon"].attrs == {
            "units": "°E",
            "long_name": "Longitude",
            "bounds": "lon_bnds",
        }
        assert ds_out1["lon_bnds"].attrs == {"xcdat_bounds": "True"}
        # Case where lon_res and lat_res are set, so bounds are calculated manually
        data_getter2 = ClimateDataGetter()
        data_getter2.lon_res = lon_res
        data_getter2.lat_res = lat_res
        data_getter2._ds = ds_in
        data_getter2._process_data()
        ds_out2 = data_getter2._ds
        xrt.assert_identical(
            ds_out1["delivery"],
            ds_out2["delivery"],
        )
        xrt.assert_allclose(ds_out1["lon_bnds"], ds_out2["lon_bnds"])
        assert ds_out2["lon_bnds"].attrs == {}

    @pytest.mark.parametrize("lon_option", ["dim", "non_dim"])
    def test_process_data_singleton_lon(self, caplog, lon_option):
        """
        Test the _process_data method of the ClimateDataGetter class when the longitude
        coordinate is a singleton (i.e. only one longitude value).
        """
        if lon_option == "dim":
            ds_in = xr.Dataset(
                data_vars={
                    "delivery": xr.DataArray(np.random.rand(1, 5), dims=["lon", "lat"])
                },
                coords={
                    "lon": xr.DataArray([345], dims="lon"),
                    "lat": xr.DataArray(np.arange(5), dims="lat"),
                },
            )
        elif lon_option == "non_dim":
            ds_in = xr.Dataset(
                data_vars={"delivery": xr.DataArray(np.random.rand(5), dims=["lat"])},
                coords={
                    "lon": 345,
                    "lat": xr.DataArray(np.arange(5), dims="lat"),
                },
            )
        ds_in["lon"].attrs = {"long_name": "Longitude", "units": "degrees_east"}
        ds_in["lat"].attrs = {"long_name": "Latitude", "units": "degrees_north"}
        data_getter1 = ClimateDataGetter()
        data_getter1.lon_res = 0.8
        data_getter1._ds = ds_in
        data_getter1._process_data()
        ds_out1 = data_getter1._ds
        assert "lon_bnds" in ds_out1 and "lat_bnds" in ds_out1
        npt.assert_allclose(ds_out1.lon.values, -15)
        npt.assert_allclose(ds_out1["lon_bnds"].values, np.array([[-15.4, -14.6]]))
        assert ds_out1["lon"].attrs == {
            "units": "°E",
            "long_name": "Longitude",
            "bounds": "lon_bnds",
        }
        assert ds_out1["lon_bnds"].attrs == {}
        # Case where lon_res is not set, so bounds cannot be calculated
        data_getter2 = ClimateDataGetter()
        data_getter2._ds = ds_in
        with caplog.at_level(logging.WARNING):
            data_getter2._process_data()
        assert "Cannot generate bounds" in caplog.text
        ds_out2 = data_getter2._ds
        assert "lon_bnds" not in ds_out2


def test_save_processed_data():
    """
    Test the _save_processed_data method of the ClimateDataGetter class.
    """
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
    data_getter = ClimateDataGetter(
        frequency="monthly",
        subset={
            "scenarios": scenarios,
            "models": models,
            "realizations": realizations,
            "years": [2015, 2016, 2018, 2100],
            "location": "gabba",
        },
    )
    data_getter.data_source = "broad"
    data_getter._save_dir = pathlib.Path("outside/edge")
    data_getter._ds = ds

    to_netcdf_called_datasets = []
    to_netcdf_called_paths = []

    def _mock_to_netcdf(ds, *args, **kwargs):
        to_netcdf_called_datasets.append(ds)
        to_netcdf_called_paths.append(str(args[0]))

    with patch.object(xr.Dataset, "to_netcdf", new=_mock_to_netcdf):
        data_getter._save_processed_data()

    called_paths_expected = [
        "outside/edge/broad_monthly_2015_2016_2018_2100_gabba_" + comb + ".nc"
        for comb in [
            "overcast_inswinger_1",
            "overcast_inswinger_2",
            "overcast_length_1",
            "overcast_length_2",
            "sunny_inswinger_1",
            "sunny_inswinger_2",
            "sunny_length_1",
            "sunny_length_2",
        ]
    ]
    assert to_netcdf_called_paths == called_paths_expected
    for path, ds in zip(to_netcdf_called_paths, to_netcdf_called_datasets):
        assert ds.scenario == path.split("_")[-3]
        assert ds.model == path.split("_")[-2]
        assert ds.realization == int(path.split("_")[-1].split(".")[0])


def test_delete_temporary():
    """
    Test the _delete_temporary method of the ClimateDataGetter class.
    """
    data_getter = ClimateDataGetter()
    data_getter._temp_save_dir = pathlib.Path("not/a/real/path")
    data_getter._temp_file_names = [
        "temporary_1.nc",
        "temporary_2.nc",
        "temporary_3.nc",
    ]
    data_getter._ds_temp = xr.Dataset()

    mock_unlinked_paths = []

    def _mock_unlink(path, *args, **kwargs):
        mock_unlinked_paths.append(str(path))

    with patch.object(pathlib.Path, "unlink", new=_mock_unlink):
        data_getter._delete_temporary()

    assert mock_unlinked_paths == [
        "not/a/real/path/temporary_1.nc",
        "not/a/real/path/temporary_2.nc",
        "not/a/real/path/temporary_3.nc",
    ]
