"""
Unit tests for the the _cesm.py module of the climdata subpackage.

The CESMDataGetter class is tested.
"""

import pathlib
import tempfile
import time
import types
from unittest.mock import MagicMock, patch

import intake_esm
import netCDF4  # noqa (avoids warning https://github.com/pydata/xarray/issues/7259)
import numpy as np
import numpy.testing as npt
import pooch
import pytest
import requests
import siphon.catalog
import xarray as xr
import xarray.testing as xrt
from git import InvalidGitRepositoryError, Repo

from climepi.climdata._cesm import (
    ARISEDataGetter,
    CESMDataGetter,
    GLENSDataGetter,
    LENS2DataGetter,
    _preprocess_arise_dataset,
    _preprocess_glens_dataset,
)
from climepi.testing.fixtures import generate_dataset


def _get_git_branch():
    path = pathlib.Path(__file__).parents[1]
    try:
        repo = Repo(path, search_parent_directories=True)
        return repo.active_branch.name
    except InvalidGitRepositoryError:
        print("Warning: Not in a git repository, using 'main' as fallback.")
        return "main"


class TestCESMDataGetter:
    """Class for testing the CESMDataGetter class."""

    def test_find_remote_data(self):
        """Test the _find_remote_data method of the CESMDataGetter class."""
        with pytest.raises(
            NotImplementedError,
            match=r"Method _find_remote_data must be implemented in a sub\(sub\)class.",
        ):
            CESMDataGetter()._find_remote_data()

    @pytest.mark.parametrize("year_mode", ["single", "multiple"])
    @pytest.mark.parametrize(
        "location_mode",
        ["single_named", "multiple_named", "grid_lon_0_360", "grid_lon_180_180"],
    )
    def test_subset_remote_data(self, year_mode, location_mode):
        """Test the _subset_remote_data method of the CESMDataGetter class."""
        time_lb = xr.date_range(
            start="2001-01-01", periods=36, freq="MS", use_cftime=True
        )
        time_rb = xr.date_range(
            start="2001-02-01", periods=36, freq="MS", use_cftime=True
        )
        time_bnds = xr.DataArray(np.array([time_lb, time_rb]).T, dims=("time", "nbnd"))
        time = time_bnds.mean(dim="nbnd")
        ds_all = xr.Dataset(
            data_vars={
                "gus": xr.DataArray(
                    np.random.rand(36, 2, 3, 5),
                    dims=["time", "member_id", "lat", "lon"],
                ),
            },
            coords={
                "time": time,
                "time_bnds": time_bnds,
                "member_id": xr.DataArray(["id1", "id2"], dims="member_id"),
                "lat": xr.DataArray([-30, 15, 60], dims="lat"),
                "lon": xr.DataArray([0, 50, 180, 230, 359], dims="lon"),
            },
        )

        if year_mode == "single":
            years = [2002]
            time_inds_expected = slice(12, 24)
        elif year_mode == "multiple":
            years = [2002, 2003]
            time_inds_expected = slice(12, 36)
        else:
            raise ValueError(f"Invalid year_mode: {year_mode}")
        if location_mode == "single_named":
            locations = ["Los Angeles"]
            lat_range = None
            lon_range = None
            lon_inds_expected = None
            lat_inds_expected = None
        elif location_mode == "multiple_named":
            locations = ["Los Angeles", "Tokyo"]
            lat_range = None
            lon_range = None
            lon_inds_expected = None
            lat_inds_expected = None
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
        else:
            raise ValueError(f"Invalid location_mode: {location_mode}")
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
                ),
            )
        else:
            xrt.assert_identical(
                data_getter._ds,
                ds_all.isel(time=time_inds_expected).climepi.sel_geo(
                    np.atleast_1d(locations).tolist()
                ),
            )

    @patch.object(xr.Dataset, "to_netcdf", autospec=True)
    def test_download_remote_data(self, mock_to_netcdf):
        """
        Unit test for the _download_remote_data method of the CESMDataGetter class.

        The download is mocked to avoid actually downloading the remote data.
        """
        ds = xr.Dataset(
            data_vars={"chris": xr.DataArray(np.random.rand(6), dims=["ball"])}
        )
        data_getter = CESMDataGetter()
        data_getter._temp_save_dir = pathlib.Path(".")
        data_getter._ds = ds
        data_getter._download_remote_data()
        assert data_getter._temp_file_names == ["temp_data.nc"]
        mock_to_netcdf.assert_called_once_with(
            ds, pathlib.Path("temp_data.nc"), compute=False
        )
        mock_to_netcdf.return_value.compute.assert_called_once_with()

    def test_open_temp_data(self):
        """
        Unit test for the _open_temp_data method of the CESMDataGetter class.

        Checks that chunking is applied as expected (chunks size 1 for any present of
        member_id, model, scenario, and location, and a single chunk per file division
        for time).
        """
        time_lb = xr.date_range(
            start="2001-01-01",
            periods=12,
            freq="MS",
            calendar="noleap",
            use_cftime=True,
        )
        time_rb = xr.date_range(
            start="2001-02-01",
            periods=12,
            freq="MS",
            calendar="noleap",
            use_cftime=True,
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
                "member_id": xr.DataArray(
                    ["id1", "id2", "id3", "id4"], dims="member_id"
                ),
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
                "time": (12,),
                "member_id": (1, 1, 1, 1),
            }
            assert (
                "nbnd" in data_getter._ds_temp.chunks
            )  # time_bnds chunked in _ds_temp but not in _ds
            data_getter._ds_temp.close()

    @pytest.mark.parametrize("frequency", ["monthly", "yearly"])
    @pytest.mark.parametrize("ds_unprocessed_has_time_bnds", [True, False])
    def test_process_data(self, frequency, ds_unprocessed_has_time_bnds):
        """Test the _process_data method of the CESMDataGetter class."""
        time_lb = xr.date_range(
            start="2001-01-01",
            periods=36,
            freq="MS",
            calendar="noleap",
            use_cftime=True,
        )
        time_rb = xr.date_range(
            start="2001-02-01",
            periods=36,
            freq="MS",
            calendar="noleap",
            use_cftime=True,
        )
        time_bnds_in = xr.DataArray(
            np.array([time_lb, time_rb]).T, dims=("time", "nbnd"), name="time_bnds"
        )
        time_in = time_bnds_in.mean(dim="nbnd")
        if ds_unprocessed_has_time_bnds:
            time_in = time_in.assign_attrs(bounds="time_bnds")
        time_in.encoding = {"calendar": "noleap", "units": "days since 2000-01-01"}
        ds_unprocessed = xr.Dataset(
            data_vars={
                "TREFHT": xr.DataArray(
                    np.random.rand(36, 2, 1, 1),
                    dims=["time", "member_id", "lat", "lon"],
                ),
                "PRECC": xr.DataArray(
                    np.random.rand(36, 2, 1, 1),
                    dims=["time", "member_id", "lat", "lon"],
                ),
                "PRECL": xr.DataArray(
                    np.random.rand(36, 2, 1, 1),
                    dims=["time", "member_id", "lat", "lon"],
                ),
            },
            coords={
                "time": time_in,
                "member_id": xr.DataArray(["id1", "id2"], dims="member_id"),
                "lat": xr.DataArray([30], dims="lat"),
                "lon": xr.DataArray([150], dims="lon"),
                **({"time_bnds": time_bnds_in} if ds_unprocessed_has_time_bnds else {}),
            },
        )
        data_getter = CESMDataGetter(
            frequency=frequency, subset={"realizations": [0, 1]}
        )
        data_getter.available_models = ["cesm"]
        data_getter.available_scenarios = ["some_scenario"]
        data_getter._ds = ds_unprocessed
        data_getter._process_data()
        ds_processed = data_getter._ds
        # Check changes in dimensions/coords
        npt.assert_equal(
            ds_processed.temperature.dims,
            ["scenario", "model", "time", "realization", "lat", "lon"],
        )
        npt.assert_equal(ds_processed["realization"].values, [0, 1])
        npt.assert_equal(ds_processed["scenario"].values, "some_scenario")
        npt.assert_equal(ds_processed["model"].values, "cesm")
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
        # Check time bounds match supplied bounds for monthly data (note for yearly
        # data, bounds are regenerated after averaging)
        if frequency == "monthly":
            npt.assert_equal(ds_processed.time_bnds.values, time_bnds_in.values)
        # Check addition of longitude and latitude bounds using known grid spacing
        npt.assert_allclose(
            ds_processed.lon_bnds.values.squeeze(), [150 - 0.625, 150 + 0.625]
        )
        npt.assert_allclose(
            ds_processed.lat_bnds.values.squeeze(), [30 - 180 / 382, 30 + 180 / 382]
        )


class TestLENS2DataGetter:
    """Class for testing the LENS2DataGetter class."""

    @pytest.mark.parametrize("frequency", ["daily", "monthly", "yearly"])
    def test_find_remote_data(self, frequency):
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

        data_getter = LENS2DataGetter(frequency=frequency)

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
        assert call_kwargs == {
            "storage_options": {"anon": True},
            "xarray_combine_by_coords_kwargs": {
                "coords": "minimal",
                "compat": "override",
            },
        }
        xrt.assert_identical(data_getter._ds, ds)

    def test_subset_remote_data(self):
        """
        Test the _subset_remote_data method of the LENS2DataGetter class.

        Checks that the method correctly implments realization subsetting, in addition
        to the time and spatial subsetting implemented in the parent class.
        """
        time_lb = xr.date_range(
            start="2001-01-01", periods=36, freq="MS", use_cftime=True
        )
        time_rb = xr.date_range(
            start="2001-02-01", periods=36, freq="MS", use_cftime=True
        )
        time_bnds = xr.DataArray(np.array([time_lb, time_rb]).T, dims=("time", "nbnd"))
        time = time_bnds.mean(dim="nbnd")
        ds_all = xr.Dataset(
            data_vars={
                "gus": xr.DataArray(
                    np.random.rand(36, 4, 3, 5),
                    dims=["time", "member_id", "lat", "lon"],
                ),
            },
            coords={
                "time": time,
                "time_bnds": time_bnds,
                "member_id": xr.DataArray(
                    ["id1", "id2", "id3", "id4"], dims="member_id"
                ),
                "lat": xr.DataArray([-30, 15, 60], dims="lat"),
                "lon": xr.DataArray([0, 50, 180, 230, 359], dims="lon"),
            },
        )
        years = [2002, 2003]
        time_inds_expected = slice(12, 36)
        lat_range = [10, 60]
        lon_range = [15, 240]
        lat_inds_expected = [1, 2]
        lon_inds_expected = [1, 2, 3]
        subset = {
            "years": years,
            "realizations": [0, 2],
            "lat_range": lat_range,
            "lon_range": lon_range,
        }
        data_getter = LENS2DataGetter(frequency="monthly", subset=subset)
        data_getter._ds = ds_all
        data_getter._subset_remote_data()
        xrt.assert_identical(
            data_getter._ds,
            ds_all.isel(
                time=time_inds_expected,
                lat=lat_inds_expected,
                lon=lon_inds_expected,
                member_id=[0, 2],
            ),
        )


class TestARISEDataGetter:
    """Class for testing the ARISEDataGetter class."""

    @patch.object(xr, "open_mfdataset", autospec=True)
    @patch("climepi.climdata._cesm._get_data_version", return_value=_get_git_branch())
    @pytest.mark.parametrize("frequency", ["daily", "monthly", "yearly"])
    def test_find_remote_data(self, _, mock_open_mfdataset, frequency):
        """Test the _find_remote_data method of the ARISEDataGetter class."""
        data_getter = ARISEDataGetter(
            frequency=frequency,
            subset={"years": [2044, 2045, 2046], "realizations": [1, 5]},
        )
        data_getter._find_remote_data()
        urls_expected = [
            "s3://ncar-cesm2-arise/CESM2-WACCM-SSP245/metadata/fsspec/"
            "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM."
            f"{member_id}.cam.{'h1' if frequency == 'daily' else 'h0'}.{data_var}."
            f"{time_str}.json"
            for member_id in ["002", "006"]
            for data_var in ["TREFHT", "PRECT"]
            for time_str in (
                ["20350101-20441231", "20450101-20541231"]
                if frequency == "daily"
                else ["201501-206412"]
            )
        ] + [
            "s3://ncar-cesm2-arise/ARISE-SAI-1.5/metadata/fsspec/"
            "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT."
            f"{member_id}.cam.{'h1' if frequency == 'daily' else 'h0'}.{data_var}."
            f"{time_str}.json"
            for member_id in ["002", "006"]
            for data_var in ["TREFHT", "PRECT"]
            for time_str in (
                ["20350101-20441231", "20450101-20541231"]
                if frequency == "daily"
                else ["203501-206912"]
            )
        ]
        urls_called = mock_open_mfdataset.call_args.args[0]
        assert sorted(urls_called) == sorted(urls_expected)

    def test_find_remote_data_errors(self):
        """Test cases where _find_remote_data should raise an error."""
        with pytest.raises(ValueError, match="Frequency hourly is not supported."):
            data_getter = ARISEDataGetter(frequency="hourly")
            data_getter._find_remote_data()
        with pytest.raises(ValueError, match="Scenario overcast is not supported."):
            data_getter = ARISEDataGetter(
                frequency="monthly", subset={"scenarios": ["overcast"]}
            )
            data_getter._find_remote_data()


@pytest.mark.parametrize("frequency", ["daily", "monthly"])
@pytest.mark.parametrize("scenario", ["ssp245", "sai15", "overcast"])
def test_preprocess_arise_dataset(frequency, scenario):
    """Test the _preprocess_arise_dataset function."""
    ds_in = generate_dataset(data_var="TREFHT", frequency=frequency, has_bounds=True)
    if frequency == "monthly":
        # Set time to end of month for monthly data
        time_attrs = ds_in["time"].attrs.copy()
        time_encoding = ds_in["time"].encoding.copy()
        ds_in = ds_in.assign_coords(
            time=ds_in.time_bnds.isel(bnds=1).assign_attrs(time_attrs)
        )
        ds_in["time"].encoding = time_encoding
    ds_in = ds_in.assign_attrs(
        case=(
            "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.002"
            if scenario == "sai15"
            else "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.002"
            if scenario == "ssp245"
            else "some_invalid_string_to_test_error"
        ),
    )
    if scenario == "overcast":
        with pytest.raises(
            ValueError, match="Failed to parse scenario from case attribute"
        ):
            _preprocess_arise_dataset(
                ds_in, frequency=frequency, realizations=[1], years=list(range(3000))
            )
        return
    result = _preprocess_arise_dataset(
        ds_in, frequency=frequency, realizations=[0, 1], years=list(range(3000))
    )
    npt.assert_equal(result.TREFHT.member_id.values, ["002"])
    npt.assert_equal(result.TREFHT.scenario.values, [scenario])
    npt.assert_equal(
        result.TREFHT.squeeze(["member_id", "scenario"], drop=True).values,
        ds_in.TREFHT.values,
    )
    assert "time_bnds" not in result
    if frequency == "daily":
        xrt.assert_identical(result["time"], ds_in["time"])
    elif frequency == "monthly":
        npt.assert_equal(result["time"].values, ds_in["time_bnds"].isel(bnds=0).values)
        assert result["time"].encoding == ds_in["time"].encoding
        assert result["time"].attrs == ds_in["time"].attrs
    else:
        raise ValueError(f"Unexpected frequency: {frequency}")
    with pytest.raises(AssertionError, match="Unexpected member_id 002"):
        _preprocess_arise_dataset(
            ds_in, frequency=frequency, realizations=[2], years=[]
        )


class TestGLENSDataGetter:
    """Class for testing the GLENSDataGetter class."""

    @patch.object(GLENSDataGetter, "_open_local_data", autospec=True)
    @patch.object(GLENSDataGetter, "_find_remote_data", autospec=True)
    @patch.object(GLENSDataGetter, "_subset_remote_data", autospec=True)
    @patch.object(GLENSDataGetter, "_download_remote_data", autospec=True)
    @patch.object(GLENSDataGetter, "_open_temp_data", autospec=True)
    @patch.object(GLENSDataGetter, "_process_data", autospec=True)
    @patch.object(GLENSDataGetter, "_save_processed_data", autospec=True)
    @patch.object(GLENSDataGetter, "_delete_temporary", autospec=True)
    @pytest.mark.parametrize("single_scenario", [True, False])
    @pytest.mark.parametrize("already_downloaded", [True, False])
    def test_get_data(
        self,
        mock_delete_temporary,
        mock_save_processed_data,
        mock_process_data,
        mock_open_temp_data,
        mock_download_remote_data,
        mock_subset_remote_data,
        mock_find_remote_data,
        mock_open_local_data,
        single_scenario,
        already_downloaded,
    ):
        """Test the get_data method."""

        def _mock_open_local_data(obj, *args, **kwargs):
            if already_downloaded or mock_save_processed_data.call_count:
                if mock_process_data.call_count < 2:
                    # If retrieving both scenarios, reset the mock after the first
                    # scenario so FileNotFound error is raised on second scenario (but
                    # don't reset after second scenario to avoid interfering with final
                    # call to _open_local_data)
                    mock_save_processed_data.reset_mock()
                obj._ds = MagicMock()
                return
            raise FileNotFoundError("File not found")

        mock_open_local_data.side_effect = _mock_open_local_data

        data_getter = GLENSDataGetter(
            frequency="monthly",
            subset={"scenarios": ["rcp85"]} if single_scenario else None,
        )
        data_getter.get_data()

        if already_downloaded:
            mock_open_local_data.assert_called_once()
            mock_process_data.assert_not_called()
        elif single_scenario:
            assert mock_open_local_data.call_count == 2
            mock_process_data.assert_called_once()
        else:
            # _open_local_data() should be called once at start, twice for separate
            # get_data() calls for each of the two scenarios, and again at the end
            assert mock_open_local_data.call_count == 6
            assert mock_process_data.call_count == 2

    @patch.object(xr, "open_mfdataset", autospec=True)
    @patch.object(xr, "combine_by_coords", autospec=True)
    @patch.object(siphon.catalog, "TDSCatalog")
    @patch.object(time, "sleep", autospec=True)
    @pytest.mark.parametrize("frequency", ["daily", "monthly"])
    @pytest.mark.parametrize("scenario", ["rcp85", "sai"])
    def test_find_remote_data(
        self,
        mock_sleep,
        mock_catalog,
        mock_combine_by_coords,
        mock_open_mfdataset,
        frequency,
        scenario,
    ):
        """Test the _find_remote_data method of the GLENSDataGetter class."""
        data_vars = (
            ["TREFHT", "PRECT"]
            if frequency == "daily"
            else ["TREFHT", "PRECC", "PRECL"]
        )

        def _mock_catalog(url):
            if mock_catalog.call_count < 3:
                # Simulate timeouts in the first two calls
                raise requests.exceptions.HTTPError()
            _data_var = url.split(".")[-3]
            _scenario_str = url.split(".")[6]
            assert _data_var in data_vars
            assert _scenario_str in ["Control", "Feedback"]
            catalog = MagicMock()
            catalog.datasets.values.return_value = [
                types.SimpleNamespace(url_path=name)
                for name in [
                    f"datazone/glens/{_scenario_str}/atm/proc/tseries/"
                    f"{frequency}/{_data_var}/b.e15.B5505C5WCCML45BGCR.f09_g16."
                    f"{_scenario_str.lower()}.{_member_id}.cam.hX.{_data_var}."
                    f"{_year_str}.nc"
                    for _member_id in ["001", "002", "003"]
                    for _year_str in (
                        ["20100101-20191231", "20200101-20291231", "20300101-20391231"]
                        if frequency == "daily"
                        else ["201001-201912", "202001-202912", "203001-203912"]
                    )
                ]
            ]
            return catalog

        mock_catalog.side_effect = _mock_catalog

        data_getter = GLENSDataGetter(
            frequency=frequency,
            subset={
                "years": [2029, 2030],
                "realizations": [0, 2],
                "scenarios": [scenario],
            },
        )
        data_getter._find_remote_data()

        assert mock_sleep.call_count == 2  # Two retries due to HTTPError

        urls_expected = [
            f"https://data-osdf.rda.ucar.edu/ncar/rda/d651064/GLENS/{_scenario_str}"
            f"/atm/proc/tseries/{frequency}/{_data_var}/b.e15.B5505C5WCCML45BGCR."
            f"f09_g16.{_scenario_str.lower()}.{_member_id}.cam.hX.{_data_var}."
            f"{_year_str}.nc"
            for _scenario_str in [
                "Control"
                if scenario == "rcp85"
                else "Feedback"
                if scenario == "sai"
                else "Unknown"
            ]
            for _data_var in data_vars
            for _member_id in ["001", "003"]
            for _year_str in (
                ["20200101-20291231", "20300101-20391231"]
                if frequency == "daily"
                else ["202001-202912", "203001-203912"]
            )
        ]
        mock_open_mfdataset.assert_not_called()
        mock_combine_by_coords.assert_not_called()
        assert data_getter._ds is None
        assert sorted(data_getter._urls) == sorted(urls_expected)

    def test_find_remote_data_errors(self):
        """Test cases where _find_remote_data should raise an error."""
        with pytest.raises(
            ValueError, match="can only convert an array of size 1 to a Python scalar"
        ):
            data_getter = GLENSDataGetter(frequency="daily")
            data_getter._find_remote_data()
        with pytest.raises(ValueError, match="Frequency hourly is not supported."):
            data_getter = GLENSDataGetter(
                frequency="hourly", subset={"scenarios": ["rcp85"]}
            )
            data_getter._find_remote_data()
        with pytest.raises(
            ValueError,
            match="Scenario overcast and/or frequency monthly not supported.",
        ):
            data_getter = GLENSDataGetter(
                frequency="monthly", subset={"scenarios": ["overcast"]}
            )
            data_getter._find_remote_data()

    @patch.object(CESMDataGetter, "_subset_remote_data", autospec=True)
    def test_subset_remote_data(self, mock_parent_subset):
        """Test the _subset_remote_data function."""
        data_getter = GLENSDataGetter(frequency="daily")
        data_getter._subset_remote_data()
        mock_parent_subset.assert_not_called()

    @patch.object(CESMDataGetter, "_download_remote_data", autospec=True)
    @patch.object(pooch, "create", autospec=True)
    def test_download_remote_data(self, mock_pooch_create, mock_parent_download):
        """
        Test the _download_remote_data function.

        The parent method should be overriden if full_download is False.
        """
        data_getter = GLENSDataGetter(frequency="daily")
        data_getter._urls = [
            "some/url.edu/path/file1.nc",
            "some/url.edu/path/file2.nc",
            "other/url.edu/another/path/file3.nc",
        ]
        temp_save_dir = pathlib.Path("some_dir")
        data_getter._temp_save_dir = temp_save_dir
        data_getter._download_remote_data()
        mock_parent_download.assert_not_called()
        assert mock_pooch_create.call_count == 3
        for kwargs in [
            {
                "base_url": "some/url.edu/path",
                "path": temp_save_dir,
                "registry": {"file1.nc": None},
                "retry_if_failed": 5,
            },
            {
                "base_url": "some/url.edu/path",
                "path": temp_save_dir,
                "registry": {"file2.nc": None},
                "retry_if_failed": 5,
            },
            {
                "base_url": "other/url.edu/another/path",
                "path": temp_save_dir,
                "registry": {"file3.nc": None},
                "retry_if_failed": 5,
            },
        ]:
            mock_pooch_create.assert_any_call(**kwargs)
        assert data_getter._temp_file_names == [
            "file1.nc",
            "file2.nc",
            "file3.nc",
        ]

    @patch.object(CESMDataGetter, "_open_temp_data", autospec=True)
    def test_open_temp_data(self, mock_parent_open_temp_data):
        """Test the _open_temp_data function."""
        data_getter = GLENSDataGetter(frequency="daily")
        data_getter._open_temp_data()
        mock_parent_open_temp_data.assert_called_once()
        assert "preprocess" in mock_parent_open_temp_data.call_args.kwargs

    @patch.object(CESMDataGetter, "_subset_remote_data", autospec=True)
    @patch.object(CESMDataGetter, "_process_data", autospec=True)
    def test_process_data(self, mock_parent_process_data, mock_parent_subset_data):
        """Test the _process_data function."""
        data_getter = GLENSDataGetter(frequency="daily")
        data_getter._process_data()
        mock_parent_subset_data.assert_called_once()
        mock_parent_process_data.assert_called_once()


@pytest.mark.parametrize("frequency", ["daily", "monthly", "hourly"])
@pytest.mark.parametrize("scenario", ["rcp85", "sai", "overcast"])
def test_preprocess_glens_dataset(frequency, scenario):
    """Test the _preprocess_glens_dataset function."""
    ds_in = generate_dataset(
        data_var="TREFHT",
        frequency=("daily" if frequency == "hourly" else frequency),
        has_bounds=True,
    )
    # Set time to end of interval
    time_attrs = ds_in["time"].attrs.copy()
    time_encoding = ds_in["time"].encoding.copy()
    ds_in = ds_in.assign_coords(
        time=ds_in.time_bnds.isel(bnds=1).assign_attrs(time_attrs)
    )
    ds_in["time"].encoding = time_encoding
    ds_in = ds_in.assign_attrs(
        case=(
            "b.e15.B5505C5WCCML45BGCR.f09_g16.Feedback.002"
            if scenario == "sai"
            else "b.e15.B5505C5WCCML45BGCR.f09_g16.Control.002"
            if scenario == "rcp85"
            else "some_invalid_string_to_test_error"
        ),
    )
    if scenario == "overcast":
        with pytest.raises(
            ValueError, match="Failed to parse scenario from case attribute"
        ):
            _preprocess_glens_dataset(
                ds_in, frequency=frequency, realizations=[1], years=list(range(3000))
            )
        return
    if frequency == "hourly":
        with pytest.raises(ValueError, match="Frequency hourly is not supported"):
            _preprocess_glens_dataset(
                ds_in, frequency=frequency, realizations=[0, 1], years=list(range(3000))
            )
        return
    result = _preprocess_glens_dataset(
        ds_in, frequency=frequency, realizations=[0, 1], years=list(range(3000))
    )
    npt.assert_equal(result.TREFHT.member_id.values, ["002"])
    npt.assert_equal(result.TREFHT.scenario.values, [scenario])
    npt.assert_equal(
        result.TREFHT.squeeze(["member_id", "scenario"], drop=True).values,
        ds_in.TREFHT.values,
    )
    assert "time_bnds" not in result
    npt.assert_equal(result["time"].values, ds_in["time_bnds"].isel(bnds=0).values)
    assert result["time"].encoding == ds_in["time"].encoding
    assert result["time"].attrs == ds_in["time"].attrs
    with pytest.raises(AssertionError, match="Unexpected member_id 002"):
        _preprocess_glens_dataset(
            ds_in, frequency=frequency, realizations=[2], years=list(range(3000))
        )
