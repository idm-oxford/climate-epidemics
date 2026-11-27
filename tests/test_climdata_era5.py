"""
Unit tests for the _era5 module of the climdata subpackage.

The ERA5DataGetter class is tested.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climepi import climdata
from climepi.climdata._era5 import ERA5DataGetter, _open_icechunk_archive


@pytest.fixture
def era5_archive():
    """Serve synthetic hourly fields with the Icechunk archive's layout."""
    time = pd.date_range("2020-01-01", "2023-01-01", freq="h")
    precipitation = np.where(
        (time.month == 1) & (time.day == 1) & (time.hour == 0),
        0.003,
        np.where(time.hour == 0, 0.001, 0),
    )
    fields = {
        "t2m": 273.15 + time.hour.to_numpy(),
        "tp": precipitation,
    }
    dims = ("valid_time", "latitude", "longitude")
    ds = xr.Dataset(
        {
            name: (dims, np.broadcast_to(values[:, None, None], (len(time), 4, 3)))
            for name, values in fields.items()
        },
        coords={
            "valid_time": time,
            "latitude": [90, 51.75, 51.5, -90],
            "longitude": [0, 0.25, 359.75],
            "lsm": (("latitude", "longitude"), np.ones((4, 3))),
        },
    ).chunk({"valid_time": 744})
    with patch(
        "climepi.climdata._era5._open_icechunk_archive", return_value=ds
    ) as mock_open:
        yield mock_open


def test_era5_named_locations(era5_archive, tmp_path):
    """Test lazily retrieving named locations over non-consecutive years."""
    subset = {
        "years": [2020, 2022],
        "locations": ["west", "east"],
        "lats": [51.75, 51.5],
        "lons": [-0.25, 0.25],
    }
    with climdata.get_climate_data(
        "era5", "daily", subset=subset, save_dir=tmp_path, download=False
    ) as ds:
        assert ds.sizes["time"] == 366 + 365
        assert ds.temperature.chunks is not None
        assert ds.temperature.dims == ds.precipitation.dims
        assert ds.scenario.item() == "historical"
        assert ds.model.item() == "era5"
        assert ds.realization.item() == 0
        assert ds.location.values.tolist() == subset["locations"]
        assert "lsm" not in ds.coords
        assert ds.lon.values.tolist() == [359.75, 0.25]
        assert ds.lat.values.tolist() == subset["lats"]
        assert ds.temperature.attrs["units"] == "°C"
        assert ds.precipitation.attrs["units"] == "mm/day"
        assert ds.indexes["time"][0] == pd.Timestamp("2020-01-01 12:00")
        first_day = ds.isel(time=0).compute()
        np.testing.assert_allclose(first_day.temperature, 11.5)
        np.testing.assert_allclose(first_day.precipitation, 1)
        np.testing.assert_allclose(ds.sel(time="2022-12-31").precipitation.compute(), 3)
    assert 2021 not in ds.time.dt.year.values
    era5_archive.assert_called_once()
    assert not list(tmp_path.iterdir())


def test_era5_point_cache(era5_archive, tmp_path):
    """
    Test retrieving and caching data for a single grid point.

    Checks that precipitation for the last day of the year includes the following
    midnight.
    """
    subset = {
        "years": [2020],
        "locations": ["west"],
        "lats": [51.7],
        "lons": [-0.2],
    }
    with climdata.get_climate_data("era5", "daily", subset, tmp_path) as ds:
        assert ds.sizes["time"] == 366
        assert ds.lat.values.item() == 51.75 and ds.lon.values.item() == 359.75
        assert "Earthmover Icechunk" in ds.attrs["source"]
        np.testing.assert_allclose(ds.temperature, 11.5)
        np.testing.assert_allclose(ds.precipitation.isel(time=0), 1)
        np.testing.assert_allclose(ds.precipitation.isel(time=-1), 3)
    with climdata.get_climate_data("era5", "daily", subset, tmp_path) as cached:
        assert cached.sizes["time"] == 366
    era5_archive.assert_called_once()


@pytest.mark.parametrize("frequency,count", [("monthly", 12), ("yearly", 1)])
def test_era5_cache(era5_archive, tmp_path, frequency, count):
    """Test retrieving and caching averaged data for a box crossing Greenwich."""
    subset = {"years": [2020], "lon_range": (-0.25, 0.25), "lat_range": (51.5, 90)}
    names = climdata.get_climate_data_file_names("era5", frequency, subset)
    assert len(names) == 1
    with climdata.get_climate_data("era5", frequency, subset, tmp_path) as ds:
        assert ds.sizes["time"] == count
        assert ds.lon.values.tolist() == [0, 0.25, 359.75]
        assert ds.lat.values.tolist() == [51.5, 51.75, 90]
        assert ds.lat_bnds.max().compute().item() == 90
        assert ds.time.attrs["bounds"] == "time_bnds"
        assert ds.time_bnds.values[0, 0] == np.datetime64("2020-01-01")
        assert ds.time_bnds.values[-1, 1] == np.datetime64("2021-01-01")
        expected_days = 31 if frequency == "monthly" else 366
        np.testing.assert_allclose(
            ds.precipitation.isel(time=-1), 1 + 2 / expected_days
        )
    assert [path.name for path in tmp_path.iterdir()] == names
    era5_archive.reset_mock()
    with climdata.get_climate_data("era5", frequency, subset, tmp_path) as cached:
        assert cached.sizes["time"] == count
    era5_archive.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"frequency": "hourly"}, "frequency 'hourly' is not supported"),
        ({"subset": {"years": []}}, "Select ERA5 years"),
        ({"subset": {"years": [1940]}}, "Select ERA5 years"),
        ({"subset": {"scenarios": ["ssp126"]}}, "scenario 'historical'"),
    ],
)
def test_init_unavailable(kwargs, match):
    """Test the __init__ method of the ERA5DataGetter class with unavailable options."""
    with pytest.raises(ValueError, match=match):
        ERA5DataGetter(**kwargs)


def test_era5_missing_hours(era5_archive, tmp_path):
    """Test retrieving a year that the archive does not fully cover."""
    with pytest.raises(ValueError, match="did not return every requested hour"):
        climdata.get_climate_data(
            "era5", "daily", {"years": [2023]}, tmp_path, download=False
        )


@patch.object(xr, "open_zarr", autospec=True)
@patch("climepi.climdata._era5.import_module", autospec=True)
def test_open_icechunk_archive(mock_import_module, mock_open_zarr):
    """Test the _open_icechunk_archive function."""
    icechunk = mock_import_module.return_value
    repository = icechunk.Repository.open.return_value
    assert _open_icechunk_archive() == mock_open_zarr.return_value
    assert [call.args for call in mock_import_module.call_args_list] == [
        ("icechunk",),
        ("pcodec",),
    ]
    icechunk.s3_storage.assert_called_once_with(
        bucket="earthmover-icechunk-era5",
        prefix="icechunkV2",
        region="us-east-1",
        anonymous=True,
    )
    icechunk.Repository.open.assert_called_once_with(icechunk.s3_storage.return_value)
    repository.readonly_session.assert_called_once_with("main")
    mock_open_zarr.assert_called_once_with(
        repository.readonly_session.return_value.store,
        group="single/temporal",
        consolidated=False,
        chunks={},
    )
    mock_import_module.side_effect = ImportError
    with pytest.raises(ImportError, match="requires icechunk and pcodec"):
        _open_icechunk_archive()
