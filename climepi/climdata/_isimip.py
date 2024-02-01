import pathlib
import zipfile

import numpy as np
import pandas as pd
import pooch
import xcdat  # noqa
from geopy.geocoders import Nominatim
from isimip_client.client import ISIMIPClient

from climepi.climdata._data_getter_class import ClimateDataGetter

geolocator = Nominatim(user_agent="climepi")


class ISIMIPDataGetter(ClimateDataGetter):
    data_source = "isimip"
    available_years = np.arange(2015, 2101)
    available_scenarios = ["ssp126", "ssp245", "ssp370", "ssp585"]
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client_results = None

    def _find_remote_data(self):
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        response = ISIMIPClient().files(
            simulation_round="ISIMIP3b",
            climate_variable=["tas", "pr"],
            climate_scenario=scenarios,
            climate_forcing=models,
        )
        client_results = response["results"]
        while response["next"] is not None:
            response = ISIMIPClient(data_url="").get(response["next"])
            client_results.extend(response["results"])
        self._client_results = client_results

    def _subset_remote_data(self):
        # Subset the remotely opened dataset to the requested years, realizations and
        # location(s), and store the subsetted dataset in the _ds attribute.
        years = self._subset["years"]
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
        # Get paths for files that are within the requested years
        client_file_paths = [file["path"] for file in client_results]
        client_file_start_years = [
            file["specifiers"]["start_year"] for file in client_results
        ]
        client_file_end_years = [
            file["specifiers"]["end_year"] for file in client_results
        ]
        client_file_paths = [
            path
            for path, file_start_year, file_end_year in zip(
                client_file_paths, client_file_start_years, client_file_end_years
            )
            if any((file_start_year <= year <= file_end_year for year in years))
        ]
        # Request server to subset the data
        client_results_new = ISIMIPClient().cutout(client_file_paths, bbox, poll=10)
        self._client_results = client_results_new

    def _download_remote_data(self):
        client_results = self._client_results
        temp_save_dir = self._temp_save_dir
        temp_file_names = []
        # ISIMIPClient().download(
        #     client_results["file_url"],
        #     path=temp_save_dir,
        #     validate=False,
        #     extract=False,
        # )
        zip_path = pooch.retrieve(
            client_results["file_url"],
            known_hash=None,
            fname=client_results["file_name"],
            path=temp_save_dir,
            progressbar=True,
        )
        # zip_path = temp_save_dir / client_results["file_name"]
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            temp_file_names = [
                name for name in zip_ref.namelist() if name[-3:] == ".nc"
            ]
            zip_ref.extractall(path=temp_save_dir, members=temp_file_names)
        pathlib.Path(zip_path).unlink()
        self._temp_file_names = temp_file_names

    def _open_temp_data(self, **kwargs):
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
        # Process the remotely opened dataset, and store the processed dataset in the
        # _ds attribute.
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
