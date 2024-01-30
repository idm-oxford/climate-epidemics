import itertools
import time
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
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
        self._client_results_dict = None
        self._temp_file_scenarios = None
        self._temp_file_models = None

    def _find_remote_data(self):
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        client_results_dict = {
            scenario: {model: None for model in models} for scenario in scenarios
        }
        for scenario, model in itertools.product(scenarios, models):
            response = ISIMIPClient().files(
                simulation_round="ISIMIP3b",
                climate_variable=["tas", "pr"],
                climate_scenario=scenario,
                climate_forcing=model,
            )
            results = response["results"]
            while response["next"] is not None:
                response = ISIMIPClient(data_url="").get(response["next"])
                results.extend(response["results"])
            client_results_dict[scenario][model] = results
        self._client_results_dict = client_results_dict

    def _subset_remote_data(self):
        # Subset the remotely opened dataset to the requested years, realizations and
        # location(s), and store the subsetted dataset in the _ds attribute.
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        years = self._subset["years"]
        loc_str = self._subset["loc_str"]
        lon_range = self._subset["lon_range"]
        lat_range = self._subset["lat_range"]
        client_results_dict = self._client_results_dict
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
        paths_dict = {
            scenario: {model: None for model in models} for scenario in scenarios
        }
        for scenario, model in itertools.product(scenarios, models):
            results = client_results_dict[scenario][model]
            paths = [file["path"] for file in results]
            file_start_years = [file["specifiers"]["start_year"] for file in results]
            file_end_years = [file["specifiers"]["end_year"] for file in results]
            paths = [
                path
                for path, file_start_year, file_end_year in zip(
                    paths, file_start_years, file_end_years
                )
                if any((file_start_year <= year <= file_end_year for year in years))
            ]
            paths_dict[scenario][model] = paths
        # Request server to subset the data
        from alive_progress import alive_bar

        subsetting_completed_dict = {
            scenario: {model: False for model in models} for scenario in scenarios
        }
        no_scenario_model_combs = len(scenarios) * len(models)
        with alive_bar(no_scenario_model_combs) as progress_bar:
            while progress_bar.current < no_scenario_model_combs:
                for scenario, model in itertools.product(scenarios, models):
                    if not subsetting_completed_dict[scenario][model]:
                        paths = paths_dict[scenario][model]
                        results_new = ISIMIPClient().cutout(paths, bbox)
                        if results_new["status"] == "finished":
                            client_results_dict[scenario][model] = results_new
                            subsetting_completed_dict[scenario][model] = True
                            progress_bar()
                if progress_bar.current < no_scenario_model_combs:
                    time.sleep(10)

    def _download_remote_data(self):
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        client_results_dict = self._client_results_dict
        temp_save_dir = self._temp_save_dir
        temp_file_names = []
        for scenario, model in itertools.product(scenarios, models):
            results = client_results_dict[scenario][model]
            ISIMIPClient().download(
                results["file_url"], path=temp_save_dir, validate=False, extract=False
            )
            zip_path = temp_save_dir / results["file_name"]
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                download_file_names_curr = [
                    name for name in zip_ref.namelist() if name[-3:] == ".nc"
                ]
                zip_ref.extractall(path=temp_save_dir, members=download_file_names_curr)
            zip_path.unlink()
            download_file_paths_curr = [
                temp_save_dir / name for name in download_file_names_curr
            ]
            with xr.open_mfdataset(download_file_paths_curr, chunks="auto") as ds:
                # Preprocess the data to enable concatenation along the 'scenario' and
                # 'model' dimensions.
                ds = ds.expand_dims(
                    {"scenario": [scenario], "model": [model], "realization": [0]}
                )
                # Some data have time at beginning, some at middle - set all to middle
                ds["time"] = ds["time"].dt.floor("D") + pd.Timedelta("12h")
                temp_file_name_curr = f"{scenario}_{model}.nc"
                ds.to_netcdf(temp_save_dir / temp_file_name_curr)
            for download_file_path in download_file_paths_curr:
                download_file_path.unlink()
            temp_file_names.append(temp_file_name_curr)
        self._temp_file_names = temp_file_names

    def _process_data(self):
        # NEED TO INCLUDE YEAR SUBSETTING HERE
        # Process the remotely opened dataset, and store the processed dataset in the
        # _ds attribute.
        ds_processed = self._ds.copy()
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
        self._ds = ds_processed
        super()._process_data()
