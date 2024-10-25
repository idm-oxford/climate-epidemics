"""Module for accessing and downloading ISIMIP data."""

import pathlib
import time
import zipfile
from copy import deepcopy

import numpy as np
import pandas as pd
import pooch
import requests
import xcdat  # noqa
from geopy.geocoders import Nominatim
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from climepi.climdata._data_getter_class import ClimateDataGetter


class ISIMIPDataGetter(ClimateDataGetter):
    """
    Class for accessing and downloading ISIMIP data.

    Data are taken from the ISIMIP repository
    (https://www.isimip.org/outputdata/isimip-repository/).

    Available years that can be specified in the `subset` argument of the class
    constructor range from 2015 to 2100, and a single realization (here labelled as 0)
    is available for a variety of emissions scenarios ("ssp126", "ssp370", and "ssp585")
    and models ("gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0",
    "ukesm1-0-ll", "canesm5", "cnrm-cm6-1", "cnrm-esm2-1", "ec-earth3", and "miroc6");
    note that data for an additional scenario ("ssp245") can also be requested, but are
    not retrieved by default because these data are only available for the first five of
    the ten listed models. The data must be downloaded before opening and processing
    (i.e., the `download` option in the `get_data` method must be set to `True` (the
    default value), or an error will be raised).

    See the base class (`climepi.climdata._data_getter_class.ClimateDataGetter`) for
    further details.
    """

    data_source = "isimip"
    available_years = np.arange(2015, 2101)
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
        self, *args, subset_check_interval=10, max_subset_wait_time=20, **kwargs
    ):
        # Extends the base class constructor to include the _client_results attribute,
        # which stores the results of the client requests to the ISIMIP repository, as
        # well as the _subset_check_interval and _max_subset_wait_time attributes, which
        # control the interval between checks for the completion of server-side
        # subsetting and the maximum time to wait for subsetting to complete,
        # respectively.
        super().__init__(*args, **kwargs)
        self._subset_check_interval = subset_check_interval
        self._max_subset_wait_time = max_subset_wait_time
        self._client_results = None

    def _find_remote_data(self):
        # Use the ISIMIPClient to find the available data for the requested models,
        # scenarios and years, and store a list of results (each entry of which
        # comprises a dictionary containing details of a single data file) in the
        # _client_results attribute.
        years = self._subset["years"]
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        data_url = "https://data.isimip.org/api/v1/files/"
        requests_session = requests.Session()
        requests_session.mount(
            data_url,
            HTTPAdapter(max_retries=Retry(total=5, allowed_methods={"GET", "POST"})),
        )
        response = requests_session.get(
            data_url,
            params={
                "simulation_round": "ISIMIP3b",
                "climate_variable": ["tas", "pr"],
                "climate_scenario": scenarios,
                "climate_forcing": models,
            },
        ).json()
        client_results = response["results"]
        while response["next"] is not None:
            response = requests_session.get(response["next"]).json()
            client_results.extend(response["results"])
        requests_session.close()
        # Filter the results to only include files that are within the requested years
        client_results = [
            result
            for result in client_results
            if any(
                (
                    result["specifiers"]["start_year"]
                    <= year
                    <= result["specifiers"]["end_year"]
                    for year in years
                )
            )
        ]
        self._client_results = client_results

    def _subset_remote_data(self):
        # Request server-side subsetting of the data to the requested location(s) and
        # update the _client_results attribute with the results of the subsetting.
        locations = self._subset["locations"]
        lon_range = self._subset["lon_range"]
        lat_range = self._subset["lat_range"]
        client_results = self._client_results
        subset_check_interval = self._subset_check_interval
        max_subset_wait_time = self._max_subset_wait_time
        if locations is not None:
            if isinstance(locations, list):
                return self._subset_remote_data_location_list()
            location_geopy = Nominatim(user_agent="climepi").geocode(locations)
            lat = location_geopy.latitude
            lon = location_geopy.longitude
            bbox = [lat, lat, lon, lon]
        else:
            if lon_range is None:
                lon_range = [-180, 180]
            else:
                # Ensure longitudes are in range -180 to 180 (note this works fine for
                # ranges that cross the 180 meridian).
                lon_range = ((np.array(lon_range) + 180) % 360) - 180
            if lat_range is None:
                lat_range = [-90, 90]
            bbox = [str(x) for x in list(lat_range) + list(lon_range)]
        # Request server to subset the data
        client_file_paths = [file["path"] for file in client_results]
        paths_by_cutout_request = [
            client_file_paths[i : i + 300]
            for i in range(0, len(client_file_paths), 300)
        ]
        subsetting_completed = False
        subset_start_time = time.time()
        files_api_url = "https://files.isimip.org/api/v1"
        requests_session = requests.Session()
        requests_session.mount(
            files_api_url,
            HTTPAdapter(max_retries=Retry(total=5, allowed_methods={"GET", "POST"})),
        )
        while not subsetting_completed:
            client_results_new = [
                requests_session.post(
                    files_api_url,
                    json={"task": "cutout_bbox", "paths": paths, "bbox": bbox},
                ).json()
                for paths in paths_by_cutout_request
            ]
            job_ids = [results["id"] for results in client_results_new]
            job_statuses = [results["status"] for results in client_results_new]
            if all(job_status == "finished" for job_status in job_statuses):
                subsetting_completed = True
            else:
                if time.time() - subset_start_time > max_subset_wait_time:
                    job_urls = [
                        "https://data.isimip.org/download/" + job_id
                        for job_id in job_ids
                    ]
                    requests_session.close()
                    raise TimeoutError(
                        "Subsetting of the requested data has taken longer than the "
                        f"maximum wait time of {max_subset_wait_time} seconds. The "
                        "server-side subsetting is still in progress, and re-running "
                        "the data retrieval once the subsetting has completed will "
                        "retrieve the subsetted data. The progress of the subsetting "
                        "jobs can be monitored at the following URLs:\n"
                        + "\n".join(job_urls)
                        + "\n"
                        "Alternatively, increase the 'max_subset_wait_time' argument "
                        "to wait longer for the subsetting to complete before timing "
                        "out."
                    )
                time.sleep(subset_check_interval)
        requests_session.close()
        self._client_results = client_results_new

    def _subset_remote_data_location_list(self):
        # Request server-side subsetting of the data to a list of locations
        locations = self._subset["locations"]
        client_results = []
        any_timeout_error = False
        for location_curr in locations:
            print(f"Initiating subsetting request for location: {location_curr}")
            data_getter_curr = deepcopy(self)
            data_getter_curr._subset["locations"] = location_curr
            try:
                data_getter_curr._subset_remote_data()
                client_results += data_getter_curr._client_results
            except TimeoutError as exc:
                any_timeout_error = True
                print(
                    f"{exc}\n. Continuing to initiate subsetting for remaining "
                    "locations."
                )
        if any_timeout_error:
            raise TimeoutError("Subsetting for at least one location timed out.")
        self._client_results = client_results

    def _download_remote_data(self):
        # Download the remote data to temporary files using (printing a progress bar),
        # and store the file names in the _temp_file_names attribute. Note that this
        # method currently supports downloading both .nc files (as obtained when
        # initially querying the ISIMIP repository) and .zip files (as obtained when
        # requesting server-side subsetting of the data) - in principle, this would
        # allow data to be downloaded without server-side subsetting (and subsequently
        # subsetted locally), but this functionality may not be needed (and is not
        # currently implemented).
        client_results = self._client_results
        temp_save_dir = self._temp_save_dir
        temp_file_names = []
        for results in client_results:
            file_url = results["file_url"]
            try:
                download_file_name = results["file_name"]
            except KeyError:
                download_file_name = results["name"]
            download_path_curr = pooch.retrieve(
                file_url,
                known_hash=None,
                fname=download_file_name,
                path=temp_save_dir,
                progressbar=True,
            )
            if pathlib.Path(download_path_curr).suffix == ".zip":
                with zipfile.ZipFile(download_path_curr, "r") as zip_ref:
                    temp_file_names_curr = [
                        name for name in zip_ref.namelist() if name[-3:] == ".nc"
                    ]
                    zip_ref.extractall(path=temp_save_dir, members=temp_file_names_curr)
                pathlib.Path(download_path_curr).unlink()
                temp_file_names.extend(temp_file_names_curr)
            else:
                temp_file_names.append(download_file_name)
        self._temp_file_names = temp_file_names

    def _open_temp_data(self, **kwargs):
        # Extends the parent method by defining a custom preprocess function to enable
        # the data to be opened as a single dataset, concantenating along new 'scenario'
        # and 'model' dimensions.

        def _preprocess(ds):
            file_name = ds.encoding["source"].split("/")[-1]
            # Preprocess the data to enable concatenation along the 'scenario' and
            # 'model' dimensions.
            scenario = file_name.split("_")[3]
            model = file_name.split("_")[0]
            assert (
                scenario in self._subset["scenarios"]
                and model in self._subset["models"]
            ), (
                f"Scenario ({scenario}) or model ({model}) either not identified "
                + "correctly or not in requested data subset."
            )
            ds = ds.expand_dims(
                {"scenario": [scenario], "model": [model], "realization": [0]}
            )
            # Some data have time at beginning, some at middle - set all to middle
            centered_times = ds["time"].dt.floor("D") + pd.Timedelta("12h")
            centered_times.attrs = ds["time"].attrs
            centered_times.encoding = ds["time"].encoding
            ds["time"] = centered_times
            return ds

        kwargs = {"chunks": "auto", "preprocess": _preprocess, **kwargs}
        super()._open_temp_data(**kwargs)

    def _process_data(self):
        # Extends the parent method to add subsetting, time bounds, renaming, unit
        # conversion and (depending on the requested data frequency) temporal averaging.
        ds_processed = self._ds.copy()
        frequency = self._frequency
        years = self._subset["years"]
        locations = self._subset["locations"]
        # Ensure the data are indexed by location string if locations are provided (use
        # atleast_1d to ensure "location" is made a dimension)
        if locations is not None:
            ds_processed = ds_processed.climepi.sel_geo(
                np.atleast_1d(locations).tolist()
            )
        # Subset the data to the requested years
        ds_processed = ds_processed.isel(time=ds_processed.time.dt.year.isin(years))
        # Add time bounds using xcdat
        ds_processed = ds_processed.bounds.add_time_bounds(method="freq", freq="day")
        # Use capital letter for long name of time variable (for consistent plotting).
        ds_processed["time"].attrs.update(long_name="Time")
        # Convert temperature from Kelvin to Celsius
        ds_processed["temperature"] = ds_processed["tas"] - 273.15
        ds_processed["temperature"].attrs.update(long_name="Temperature", units="°C")
        # Convert precipitation from kg m-2 s-1 (equivalent to mm/s) to mm/day.
        ds_processed["precipitation"] = ds_processed["pr"] * (60 * 60 * 24)
        ds_processed["precipitation"].attrs.update(
            long_name="Precipitation", units="mm/day"
        )
        # Drop original temperature and precipitation variables
        ds_processed = ds_processed.drop_vars(["tas", "pr"])
        # Convert to monthly or yearly frequency if requested.
        if frequency == "monthly":
            ds_processed = ds_processed.climepi.monthly_average()
        elif frequency == "yearly":
            ds_processed = ds_processed.climepi.yearly_average()
        self._ds = ds_processed
        super()._process_data()
