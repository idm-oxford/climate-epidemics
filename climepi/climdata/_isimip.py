import itertools
import zipfile

import numpy as np
import xarray as xr
from isimip_client.client import ISIMIPClient

from climepi.climdata._data_getter_class import ClimateDataGetter


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
        self._client = None
        self._responses = None
        self._temp_file_scenarios = None
        self._temp_file_models = None

    def _find_remote_data(self):
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        client = ISIMIPClient()
        responses = {
            scenario: {model: None for model in models} for scenario in scenarios
        }
        for scenario, model in itertools.product(scenarios, models):
            responses[scenario][model] = client.datasets(
                simulation_round="ISIMIP3b",
                climate_variable=["tas", "pr"],
                climate_scenario=scenario,
                climate_forcing=model,
            )
        self._client = client
        self._responses = responses

    def _subset_remote_data(self):
        # Subset the remotely opened dataset to the requested years, realizations and
        # location(s), and store the subsetted dataset in the _ds attribute.
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        client = self._client
        responses = self._responses
        for scenario, model in itertools.product(scenarios, models):
            response = responses[scenario][model]
            results = response["results"]
            # ADD IN CODE TO ONLY KEEP RELEVANT YEARS
            paths = [file["path"] for result in results for file in result["files"]]
            response_new = client.cutout(paths, [-40, -40, 170, 170])
            responses[scenario][model] = response_new

    def _download_remote_data(self):
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        client = self._client
        responses = self._responses
        temp_save_dir = self._temp_save_dir
        temp_file_names = []
        for scenario, model in itertools.product(scenarios, models):
            response = responses[scenario][model]
            client.download(
                response["file_url"], path=temp_save_dir, validate=False, extract=False
            )
            zip_path = temp_save_dir / response["file_name"]
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
                # NEED TO MAKE DATA VARS HAVE SCENARIO AND MODEL COORDS FOR OPEN TO WORK?
                ds["scenario"] = scenario
                ds["model"] = model
                ds = ds.set_coords(["scenario", "model"])
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
        realizations = self._subset["realizations"]
        ds_processed = self._ds.copy()
        # Index data variables by an integer realization coordinate instead of the
        # member_id coordinate (which is a string), and add model and scenario
        # coordinates.
        ds_processed = ds_processed.swap_dims({"member_id": "realization"})
        ds_processed["realization"] = realizations
        ds_processed["scenario"] = self.available_scenarios
        ds_processed["model"] = self.available_models
        ds_processed = ds_processed.set_coords(["scenario", "model"])
        # Convert the calendar from the no-leap calendar to the proleptic_gregorian
        # calendar (avoids plotting issues).
        time_bnds_new = xr.concat(
            [
                ds_processed.isel(nbnd=nbnd)
                .convert_calendar("proleptic_gregorian", dim="time_bnds")
                .time_bnds.swap_dims({"time_bnds": "time"})
                .expand_dims("nbnd", axis=1)
                for nbnd in range(2)
            ],
            dim="nbnd",
        )
        time_bnds_new["time"] = ds_processed["time"]
        ds_processed["time_bnds"] = time_bnds_new["time_bnds"]
        time_attrs = ds_processed["time"].attrs
        time_encoding = {
            key: ds_processed["time"].encoding[key] for key in ["units", "dtype"]
        }
        ds_processed["time"] = ds_processed["time_bnds"].mean(dim="nbnd")
        ds_processed["time"].attrs = time_attrs
        ds_processed["time"].encoding = time_encoding
        # Make time bounds a data variable instead of a coordinate, and format in order
        # to match the conventions of xcdat.
        ds_processed = ds_processed.reset_coords("time_bnds")
        ds_processed["time_bnds"] = ds_processed["time_bnds"].T
        ds_processed = ds_processed.rename_dims({"nbnd": "bnds"})
        # Convert temperature from Kelvin to Celsius
        ds_processed["temperature"] = ds_processed["TREFHT"] - 273.15
        ds_processed["temperature"].attrs.update(long_name="Temperature")
        ds_processed["temperature"].attrs.update(units="°C")
        # Calculate total precipitation from convective and large-scale precipitation,
        # and convert from m/s to mm/day.
        ds_processed["precipitation"] = (
            ds_processed["PRECC"] + ds_processed["PRECL"]
        ) * (1000 * 60 * 60 * 24)
        ds_processed["precipitation"].attrs.update(long_name="Precipitation")
        ds_processed["precipitation"].attrs.update(units="mm/day")
        ds_processed = ds_processed.drop(["TREFHT", "PRECC", "PRECL"])
        self._ds = ds_processed
        super()._process_data()
