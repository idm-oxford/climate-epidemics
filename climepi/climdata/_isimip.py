"""Module for accessing and downloading ISIMIP data."""

# Lazy imports activated on Python 3.15+ (PEP 810); inert on earlier versions.
__lazy_modules__ = ["requests", "requests.adapters", "urllib3.util"]

import pathlib
import time
import zipfile
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from climepi._xcdat import BoundsAccessor  # noqa
from climepi.climdata._data_getter_class import ClimateDataGetter
from climepi.climdata._utils import _standardize_names_and_units


class ISIMIPDataGetter(ClimateDataGetter):
    """
    Class for accessing and downloading ISIMIP data.

    Data are taken from the ISIMIP repository (https://data.isimip.org/). Terms of use
    can be found at
    https://www.isimip.org/gettingstarted/terms-of-use/terms-use-publicly-available-isimip-data-after-embargo-period/.

    Available years that can be specified in the `subset` argument of the class
    constructor range from 2015 to 2100, and a single realization (here labeled as 0)
    is available for a variety of emissions scenarios ("ssp126", "ssp370", and "ssp585")
    and models ("gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0",
    "ukesm1-0-ll", "canesm5", "cnrm-cm6-1", "cnrm-esm2-1", "ec-earth3", and "miroc6").
    Data for an additional scenario ("ssp245") can also be requested for all ten models.
    The data must be downloaded before opening and processing
    (i.e., the `download` option in the `get_data` method must be set to `True` (the
    default value), or an error will be raised).

    See the base class (`climepi.climdata._data_getter_class.ClimateDataGetter`) for
    further details.
    """

    data_source = "isimip"
    # Files API search options, and position of the scenario in the underscore-separated
    # file names (which start with the model).
    _simulation_round = "ISIMIP3b"
    _catalog_filters = {"bias_adjustment": "w5e5"}
    _scenario_file_index = 3
    _longitude_origin = -180
    available_years = list(range(2015, 2101))
    available_scenarios = ["ssp126", "ssp370", "ssp585"]
    available_models = [
        "gfdl-esm4",
        "ipsl-cm6a-lr",
        "mpi-esm1-2-hr",
        "mri-esm2-0",
        "ukesm1-0-ll",
        "canesm5",
        "cnrm-cm6-1",
        "cnrm-esm2-1",
        "ec-earth3",
        "miroc6",
    ]
    available_realizations = [0]
    lon_res = 0.5
    lat_res = 0.5

    def __init__(
        self,
        *args: Any,
        subset_check_interval: float = 10,
        max_subset_wait_time: float = 20,
        subset_method: Literal["remote", "local"] = "remote",
        **kwargs: Any,
    ) -> None:
        # Extends the base class constructor to include options for server-side
        # subsetting, and the _client_results attribute, which stores details of the
        # remote data files.
        super().__init__(*args, **kwargs)
        if subset_method not in ("remote", "local"):
            raise ValueError("subset_method must be 'remote' or 'local'.")
        self._subset_before_download = subset_method == "remote"
        self._subset_check_interval = subset_check_interval
        self._max_subset_wait_time = max_subset_wait_time
        self._client_results: list[dict[str, Any]] | None = None

    def _find_remote_data(self) -> None:
        # Find the daily data files covering the requested years, scenarios and models,
        # and store their details in the _client_results attribute.
        years = set(self._subset["years"])
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        variables = ["tas", "pr"]
        if not years or not scenarios or not models:
            raise ValueError("Select at least one ISIMIP year, scenario, and model.")

        url = "https://data.isimip.org/api/v1/files/"
        params: dict[str, Any] = {
            "simulation_round": self._simulation_round,
            "category": "climate",
            "subcategory": "atmosphere",
            "region": "global",
            "time_step": "daily",
            "climate_variable": variables,
            "climate_scenario": scenarios,
            "climate_forcing": models,
            **self._catalog_filters,
        }
        results: list[dict[str, Any]] = []
        coverage: dict[tuple[str, str, str], set[int]] = {}
        with _request_session() as session:
            while url is not None:
                response = session.get(url, params=params, timeout=30)
                response.raise_for_status()
                page = response.json()
                for result in page["results"]:
                    specifiers = result["specifiers"]
                    file_years = years.intersection(
                        range(specifiers["start_year"], specifiers["end_year"] + 1)
                    )
                    if file_years:
                        results.append(result)
                        key = (
                            specifiers["climate_scenario"],
                            specifiers["climate_forcing"],
                            specifiers["climate_variable"],
                        )
                        coverage.setdefault(key, set()).update(file_years)
                url, params = page["next"], {}

        for scenario, model, variable in product(scenarios, models, variables):
            missing = years - coverage.get((scenario, model, variable), set())
            if missing:
                raise ValueError(
                    f"No {self._simulation_round} {variable} data for scenario "
                    f"'{scenario}', model '{model}', and year {min(missing)}. "
                    "Check the available scenarios, models, and years."
                )
        self._client_results = results

    def _remote_operations(self) -> list[dict[str, Any]]:
        # Get the Files API operations that cut out the requested location(s) or ranges
        # (none if global data are requested).
        locations = self._subset["locations"]
        lat_range = self._subset["lat_range"]
        if (
            locations is None
            and self._subset["lon_range"] is None
            and lat_range is None
        ):
            return []

        # The public ISIMIP grids have 0.5 degree spacing and cell-centered coordinates.
        grid = xr.Dataset(
            coords={
                "lon": np.arange(-180 + self.lon_res / 2, 180, self.lon_res),
                "lat": np.arange(-90 + self.lat_res / 2, 90, self.lat_res),
            }
        )
        selected = self._subset_grid(grid)
        if locations is None:
            west, east = self._get_lon_range()
            # Send grid centers so the server's coordinate rounding cannot change
            # which cells are selected. Preserve wrapping if both sides have cells.
            lons = selected.indexes["lon"]
            split = int(lons.searchsorted(west)) if west > east else 0
            if 0 < split < lons.size:
                west, east = float(lons[split]), float(lons[split - 1])
            else:
                west, east = float(lons[0]), float(lons[-1])
            south, north = selected.lat.values[[0, -1]].tolist()
            # v2 expects longitude bounds first. West > east denotes a wrapped range.
            return [{"operation": "cutout_bbox", "bbox": [west, east, south, north]}]

        # Use the same nearest-cell selection as local subsetting, including ties.
        # The Files API's point operation takes latitude before longitude.
        return [
            {"operation": "cutout_point", "point": [lat, lon]}
            for lat, lon in dict.fromkeys(
                zip(
                    selected.lat.values.tolist(),
                    selected.lon.values.tolist(),
                    strict=True,
                )
            )
        ]

    def _subset_remote_data(self) -> None:
        # Request server-side subsetting of the data files, wait for the jobs to finish,
        # and replace the _client_results attribute with the finished jobs.
        operations = self._remote_operations()
        if not operations:
            return
        assert self._client_results is not None
        paths = sorted(result["path"] for result in self._client_results)
        files_api_url = "https://files.isimip.org/api/v2"
        jobs: dict[str, dict[str, Any]] = {}
        with _request_session() as session:
            for operation in operations:
                # The v2 server limits the number of files in a job.
                for start in range(0, len(paths), 32):
                    response = session.post(
                        files_api_url,
                        json={
                            "paths": paths[start : start + 32],
                            "operations": [operation],
                        },
                        timeout=30,
                    )
                    response.raise_for_status()
                    job = response.json()
                    jobs[job["job_url"]] = job

            deadline = time.monotonic() + self._max_subset_wait_time
            while True:
                pending = []
                for url, job in jobs.items():
                    status = job["status"]
                    if status in ("queued", "started", "scheduled", "deferred"):
                        pending.append(url)
                    elif status != "finished" or not job.get("file_url"):
                        raise RuntimeError(
                            f"ISIMIP subsetting job {status}. See {url}."
                        )
                if not pending:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "ISIMIP subsetting is still in progress. Run the same request "
                        "again later, increase 'max_subset_wait_time', or use "
                        "subset_method='local' to download the original files. "
                        "Pending jobs:\n" + "\n".join(pending)
                    )
                time.sleep(min(self._subset_check_interval, remaining))
                for url in pending:
                    response = session.get(url, timeout=30)
                    response.raise_for_status()
                    jobs[url] = response.json()

        self._client_results = list(jobs.values())

    def _download_remote_data(self) -> None:
        # Download the original or subsetted data files (extracting any zip archives)
        # to the temporary directory, and store the file names in the _temp_file_names
        # attribute.
        assert self._client_results is not None
        assert self._temp_save_dir is not None
        temp_file_names = []
        for result in self._client_results:
            known_hash = None
            if result.get("checksum") and result.get("checksum_type"):
                known_hash = f"{result['checksum_type']}:{result['checksum']}"
            download_path = self._download_file(result["file_url"], known_hash)
            if download_path.suffix == ".zip":
                with zipfile.ZipFile(download_path) as archive:
                    files = [
                        name for name in archive.namelist() if name.endswith(".nc")
                    ]
                    archive.extractall(path=self._temp_save_dir, members=files)
                download_path.unlink()
                temp_file_names.extend(files)
            else:
                temp_file_names.append(download_path.name)
        self._temp_file_names = temp_file_names

    def _open_temp_data(self, **kwargs: Any) -> None:
        # Extends the parent method by defining a custom preprocess function to enable
        # the data to be opened as a single dataset, concatenating along new 'scenario'
        # and 'model' dimensions.

        def _preprocess(ds: xr.Dataset) -> xr.Dataset:
            file_name = pathlib.Path(ds.encoding["source"]).name
            # Preprocess the data to enable concatenation along the 'scenario' and
            # 'model' dimensions.
            parts = file_name.split("_")
            scenario = parts[self._scenario_file_index]
            model = parts[0]
            assert (
                scenario in self._subset["scenarios"]
                and model in self._subset["models"]
            ), (
                f"Scenario ({scenario}) or model ({model}) either not identified "
                + "correctly or not in requested data subset."
            )
            # NCO writes wrapped cutouts west first. Restore the native longitude order
            # before combining files, which requires monotonically ordered coordinates.
            if "lon" in ds.indexes and not ds.indexes["lon"].is_monotonic_increasing:
                ds = ds.sortby("lon")
            ds = ds.expand_dims(
                {
                    "scenario": np.array([scenario], dtype="object"),
                    "model": np.array([model], dtype="object"),
                    "realization": [0],
                }
            )
            # Some data have time at beginning, some at middle - set all to middle
            centered_times: xr.DataArray = ds["time"].dt.floor("D") + pd.Timedelta(
                "12h"
            )  # type: ignore[operator]
            centered_times.attrs = ds["time"].attrs
            centered_times.encoding = ds["time"].encoding
            ds["time"] = centered_times
            return ds

        kwargs = {"preprocess": _preprocess, "join": "outer", **kwargs}
        super()._open_temp_data(**kwargs)

    def _process_data(self) -> None:
        # Extends the parent method to add subsetting, time bounds, renaming, unit
        # conversion and (depending on the requested data frequency) temporal averaging.
        assert self._ds is not None
        ds_processed = self._ds.copy()
        frequency = self._frequency
        years = self._subset["years"]
        if not self._subset_before_download or self._subset["locations"] is not None:
            ds_processed = self._subset_grid(ds_processed)
        # Subset the data to the requested years
        ds_processed = ds_processed.isel(time=ds_processed.time.dt.year.isin(years))
        # Add time bounds using xcdat
        ds_processed = ds_processed.bounds.add_time_bounds(method="freq", freq="day")
        # Use capital letter for long name of time variable (for consistent plotting).
        ds_processed["time"].attrs.update(long_name="Time")
        # Convert temperature from Kelvin to Celsius, and precipitation from kg m-2 s-1
        # (equivalent to mm/s) to mm/day.
        ds_processed = _standardize_names_and_units(
            ds_processed, "tas", "pr", 60 * 60 * 24
        )
        # Convert to monthly or yearly frequency if requested.
        if frequency == "monthly":
            ds_processed = ds_processed.climepi.monthly_average()
        elif frequency == "yearly":
            ds_processed = ds_processed.climepi.yearly_average()
        self._ds = ds_processed
        super()._process_data()


class ISIMIP3aDataGetter(ISIMIPDataGetter):
    """Class for accessing and downloading ISIMIP3a GSWP3-W5E5 historical data."""

    data_source = "isimip3a"
    available_years = list(range(1901, 2020))
    available_scenarios = ["obsclim"]
    available_models = ["gswp3-w5e5"]
    _simulation_round = "ISIMIP3a"
    _catalog_filters = {"product": "InputData"}
    _scenario_file_index = 1


class ISIMIP4aDataGetter(ISIMIPDataGetter):
    """Class for accessing and downloading ISIMIP4a GSWP3-ERA5 historical data."""

    data_source = "isimip4a"
    available_years = list(range(1901, 2026))
    available_scenarios = ["obsclim"]
    available_models = ["gswp3-era5"]
    _simulation_round = "ISIMIP4a"
    _catalog_filters = {"resolution": "30arcmin"}
    _scenario_file_index = 1


class ISIMIP4bDataGetter(ISIMIPDataGetter):
    """Class for accessing and downloading ISIMIP4b CMIP7 projection data."""

    data_source = "isimip4b"
    available_years = list(range(2022, 2101))
    available_scenarios = ["esm-scen7-vl", "esm-scen7-h"]
    available_models = ["ec-earth3-esm-1-1", "ukesm1-3-ll"]
    _simulation_round = "ISIMIP4b"
    _catalog_filters = {"resolution": "30arcmin", "bias_adjustment": "era5"}


def _request_session() -> requests.Session:
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(max_retries=Retry(total=5, allowed_methods={"GET", "POST"})),
    )
    return session
