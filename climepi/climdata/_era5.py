"""Module for accessing and downloading ERA5 data."""

from datetime import UTC, datetime, timedelta
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from climepi._xcdat import center_times
from climepi.climdata._data_getter_class import RemoteDatasetGetter
from climepi.climdata._utils import _standardize_names_and_units


class ERA5DataGetter(RemoteDatasetGetter):
    """
    Class for accessing and downloading ERA5 reanalysis data.

    Data are taken from Earthmover's public Icechunk archive of ERA5 single-level data
    (https://registry.opendata.aws/earthmover-era5/). Hourly 2 m temperature and total
    precipitation are aggregated to daily values over UTC days. Available years start
    in 1941, with a single scenario ("historical"), model ("era5") and realization
    (labeled as 0).

    See the base class (`climepi.climdata._data_getter_class.ClimateDataGetter`) for
    further details.
    """

    data_source = "era5"
    available_models = ["era5"]
    available_scenarios = ["historical"]
    available_realizations = [0]
    # Allow time for the archive to publish the last complete calendar year.
    available_years = list(range(1941, (datetime.now(UTC) - timedelta(days=120)).year))
    lon_res = lat_res = 0.25
    remote_open_possible = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Extends the base class constructor to check that the requested frequency and
        # subset are available.
        super().__init__(*args, **kwargs)
        if self._frequency not in ("daily", "monthly", "yearly"):
            raise ValueError(f"ERA5 frequency '{self._frequency}' is not supported.")
        years = self._subset["years"]
        if not years or not set(years).issubset(self.available_years):
            raise ValueError(
                f"Select ERA5 years between {self.available_years[0]} and "
                f"{self.available_years[-1]}."
            )
        if (
            self._subset["scenarios"] != self.available_scenarios
            or self._subset["models"] != self.available_models
            or self._subset["realizations"] != self.available_realizations
        ):
            raise ValueError(
                "ERA5 uses scenario 'historical', model 'era5' and realization 0."
            )

    def _find_remote_data(self) -> None:
        # Lazily open the hourly archive, and store it in the _ds attribute.
        self._ds = _open_icechunk_archive()

    def _subset_remote_data(self) -> None:
        # Subset the hourly data to the requested location(s), compute daily values for
        # the requested years, and store the daily dataset in the _ds attribute.
        assert self._ds is not None
        # Keep the archive's spatial and time axes and skip ancillary fields. The
        # archive stores longitudes from 0 to 360 and latitudes from north to south.
        ds = (
            self._ds[["t2m", "tp"]]
            .reset_coords(drop=True)
            .rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
            .isel(lat=slice(None, None, -1))
        )
        ds = self._subset_grid(ds).transpose("time", ...)

        # Plan contiguous ranges so reads do not span unrequested years.
        years = sorted(set(self._subset["years"]))
        blocks = np.split(years, np.flatnonzero(np.diff(years) != 1) + 1)
        periods = [(int(block[0]), int(block[-1]) + 1) for block in blocks]
        parts = [
            _daily_average(
                ds.sel(time=slice(f"{start}-01-01T00:00", f"{end}-01-01T00:00")),
                start,
                end,
            )
            for start, end in periods
        ]
        daily = xr.concat(parts, dim="time", coords="minimal", join="exact")
        self._ds = daily.assign_attrs(
            source="ERA5 single levels (Copernicus Climate Change Service) via "
            "Earthmover Icechunk",
            references="https://doi.org/10.24381/cds.adbb2d47",
            acknowledgment="Contains modified Copernicus Climate Change Service "
            "information.",
        )

    def _process_data(self) -> None:
        # Extends the parent method to add metadata, dimensions and time bounds, convert
        # units and (depending on the requested data frequency) average over time.
        assert self._ds is not None
        # Daily precipitation accumulation is in metres of water equivalent.
        ds = _standardize_names_and_units(self._ds, "t2m", "tp", 1000)
        for coord, long_name, axis in (
            ("time", "Time", "T"),
            ("lon", "Longitude", "X"),
            ("lat", "Latitude", "Y"),
        ):
            ds[coord].attrs.update(long_name=long_name, axis=axis)
        ds = ds.expand_dims(
            scenario=self.available_scenarios,
            model=self.available_models,
            realization=self.available_realizations,
        )
        ds = ds.bounds.add_time_bounds(method="freq", freq="day")
        ds["time"] = center_times(ds[["time_bnds"]]).time
        if self._frequency == "monthly":
            ds = ds.climepi.monthly_average()
        elif self._frequency == "yearly":
            ds = ds.climepi.yearly_average()
        # Centered times can fall at noon, so save them with sub-daily precision.
        time_units = f"hours since {min(self._subset['years'])}-01-01"
        ds.time.encoding["units"] = time_units
        ds.time_bnds.encoding["units"] = time_units
        self._ds = ds
        super()._process_data()


def _open_icechunk_archive() -> xr.Dataset:
    try:
        icechunk = import_module("icechunk")
        import_module("pcodec")
    except ImportError as exc:
        raise ImportError(
            "ERA5 retrieval requires icechunk and pcodec. "
            "Install them with `conda install -c conda-forge icechunk pcodec`."
        ) from exc
    storage = icechunk.s3_storage(
        bucket="earthmover-icechunk-era5",
        prefix="icechunkV2",
        region="us-east-1",
        anonymous=True,
    )
    session = icechunk.Repository.open(storage).readonly_session("main")
    # Small spatial tiles and long time chunks suit requests for whole calendar years.
    return xr.open_zarr(
        session.store, group="single/temporal", consolidated=False, chunks={}
    )


def _daily_average(ds: xr.Dataset, start: int, end: int) -> xr.Dataset:
    expected = pd.date_range(f"{start}-01-01", f"{end}-01-01", freq="h")
    if not np.array_equal(ds.time.values, expected.values):
        raise ValueError("The ERA5 source did not return every requested hour.")
    # Temperature is instantaneous. Precipitation is the accumulation over the hour
    # ending at each timestamp, so midnight precipitation belongs to the previous day.
    hourly = ds.isel(time=slice(None, -1))
    hourly["tp"] = ds.tp.isel(time=slice(1, None)).assign_coords(time=hourly.time)
    windows = hourly.coarsen(time=24).construct(time=("day", "hour"))
    daily = xr.Dataset(
        {
            "t2m": windows.t2m.mean("hour", skipna=False),
            "tp": windows.tp.sum("hour", skipna=False),
        }
    )
    return daily.rename(day="time").assign_coords(time=expected.values[:-1:24])
