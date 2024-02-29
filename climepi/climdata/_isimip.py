import pathlib
import time
import zipfile

import numpy as np
import pandas as pd
import pooch
import xcdat  # noqa
from geopy.geocoders import Nominatim
from isimip_client.client import ISIMIPClient
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from climepi.climdata._data_getter_class import ClimateDataGetter

geolocator = Nominatim(user_agent="climepi")

FILES_API_URL = "https://files.isimip.org/api/v1"
requests_session = Session()
requests_session.mount(
    FILES_API_URL,
    HTTPAdapter(max_retries=Retry(total=5, allowed_methods={"GET", "POST"})),
)


class ISIMIPDataGetter(ClimateDataGetter):

    """
    Class for accessing and downloading ISIMIP data from the ISIMIP repository
    (https://www.isimip.org/outputdata/isimip-repository/). Available
    years that can be specified in the `subset` argument of the class constructor range
    from 2015 to 2100, and a single realization (here labelled as 0) is available for
    a variety of emissions scenarios ("ssp126", "ssp370", and "ssp585") and models
    ("gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll",
    "canesm5", "cnrm-cm6-1", "cnrm-esm2-1", "ec-earth3", and "miroc6"); note that data
    for an additional scenario ("ssp245") can also be requested, but are not retrieved
    by default because these data are only available for the first five of the ten
    listed models. The data must be downloaded before opening and processing (i.e.,
    the `download` option in the `get_data` method must be set to `True` (the default
    value), or an error will be raised).

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

    def __init__(self, *args, **kwargs):
        # Extends the base class constructor to include the _client_results attribute,
        # which stores the results of the client requests to the ISIMIP repository.
        super().__init__(*args, **kwargs)
        self._client_results = None

    def _find_remote_data(self):
        # Use the ISIMIPClient to find the available data for the requested models,
        # scenarios and years, and store a list of results (each entry of which
        # comprises a dictionary containing details of a single data file) in the
        # _client_results attribute.
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        years = self._subset["years"]
        response = ISIMIPClient().files(
            simulation_round="ISIMIP3b",
            climate_variable=["tas", "pr"],
            climate_scenario=scenarios,
            climate_forcing=models,
        )
        client_results = response["results"]
        while response["next"] is not None:
            response = requests_session.get(response["next"]).json()
            client_results.extend(response["results"])
        # Filter the results to only include files that are within the requested years
        # Get paths for files that are within the requested years
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
        loc_str = self._subset["loc_str"]
        lon_range = self._subset["lon_range"]
        lat_range = self._subset["lat_range"]
        client_results = self._client_results
        if loc_str is not None:
            location = geolocator.geocode(loc_str)
            lat = location.latitude
            lon = location.longitude
            bbox = [lat, lat, lon, lon]
        else:
            if lon_range is None:
                lon_range = [-180, 180]
            else:
                # Ensure longitudes are in range -180 to 180
                lon_range = ((np.array(lon_range) + 180) % 360) - 180
            if lat_range is None:
                lat_range = [-90, 90]
            bbox = [lat_range[0], lat_range[1], lon_range[0], lon_range[1]]
        # Request server to subset the data
        client_file_paths = [file["path"] for file in client_results]
        paths_by_cutout_request = [
            client_file_paths[i : i + 500]
            for i in range(0, len(client_file_paths), 500)
        ]
        subsetting_completed = False
        while not subsetting_completed:
            client_results_new = [
                requests_session.post(
                    FILES_API_URL,
                    json={"task": "cutout_bbox", "paths": paths, "bbox": bbox},
                ).json()
                for paths in paths_by_cutout_request
            ]
            job_ids = [results["id"] for results in client_results_new]
            job_statuses = [results["status"] for results in client_results_new]
            print(
                *[
                    f"Job {job_id} {job_status}"
                    for job_id, job_status in zip(job_ids, job_statuses)
                ],
                sep="\n",
            )
            if all(job_status == "finished" for job_status in job_statuses):
                subsetting_completed = True
            else:
                time.sleep(10)
        self._client_results = client_results_new

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
        # Extends the parent method to add temporal subsetting, renaming, unit conversion
        # and (depending on the requested data frequency) temporal averaging.
        ds_processed = self._ds.copy()
        frequency = self._frequency
        years = self._subset["years"]
        # Subset the data to the requested years
        ds_processed = ds_processed.isel(time=ds_processed.time.dt.year.isin(years))
        # Add time bounds using xcdat
        ds_processed = ds_processed.bounds.add_time_bounds(method="freq", freq="day")
        # Convert temperature from Kelvin to Celsius
        ds_processed["temperature"] = ds_processed["tas"] - 273.15
        ds_processed["temperature"].attrs.update(long_name="Temperature")
        ds_processed["temperature"].attrs.update(units="°C")
        # Convert precipitation from kg m-2 s-1 (equivalent to mm/s) to mm/day.
        ds_processed["precipitation"] = ds_processed["pr"] * (60 * 60 * 24)
        ds_processed["precipitation"].attrs.update(long_name="Precipitation")
        ds_processed["precipitation"].attrs.update(units="mm/day")
        ds_processed = ds_processed.drop(["tas", "pr"])
        # Convert to monthly or yearly frequency if requested.
        if frequency == "monthly":
            ds_processed = ds_processed.climepi.monthly_average()
        elif frequency == "yearly":
            ds_processed = ds_processed.climepi.yearly_average()
        self._ds = ds_processed
        super()._process_data()
