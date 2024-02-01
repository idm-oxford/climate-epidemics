"""
Module for accessing and downloading CESM LENS2 data from the aws server (see
https://ncar.github.io/cesm2-le-aws/model_documentation.html).
"""

import climepi  # noqa
import dask.diagnostics
import intake
import numpy as np
import xarray as xr

from climepi.climdata._data_getter_class import ClimateDataGetter


class CESMDataGetter(ClimateDataGetter):
    data_source = "lens2"
    available_years = np.arange(1850, 2101)
    available_scenarios = ["ssp370"]
    available_models = ["cesm2"]
    available_realizations = np.arange(100)

    def _find_remote_data(self):
        frequency = self._frequency
        if frequency == "yearly":
            frequency = "monthly"
        catalog = intake.open_esm_datastore(
            "https://raw.githubusercontent.com/NCAR/cesm2-le-aws"
            + "/main/intake-catalogs/aws-cesm2-le.json"
        )
        print("\n\n")
        catalog_subset = catalog.search(
            variable=["TREFHT", "PRECC", "PRECL"], frequency=frequency
        )
        ds_dict_in = catalog_subset.to_dataset_dict(storage_options={"anon": True})
        ds_cmip6_in = xr.concat(
            [
                ds_dict_in["atm.historical.monthly.cmip6"],
                ds_dict_in["atm.ssp370.monthly.cmip6"],
            ],
            dim="time",
        )
        ds_smbb_in = xr.concat(
            [
                ds_dict_in["atm.historical.monthly.smbb"],
                ds_dict_in["atm.ssp370.monthly.smbb"],
            ],
            dim="time",
        )
        ds_in = xr.concat([ds_cmip6_in, ds_smbb_in], dim="member_id")
        self._ds = ds_in

    def _subset_remote_data(self):
        # Subset the remotely opened dataset to the requested years, realizations and
        # location(s), and store the subsetted dataset in the _ds attribute.
        years = self._subset["years"]
        realizations = self._subset["realizations"]
        loc_str = self._subset["loc_str"]
        lon_range = self._subset["lon_range"]
        lat_range = self._subset["lat_range"]
        ds_subset = self._ds.copy()
        ds_subset = ds_subset.isel(member_id=realizations)
        ds_subset = ds_subset.isel(time=np.isin(ds_subset.time.dt.year, years))
        if loc_str is not None:
            ds_subset.climepi.modes = {"spatial": "global"}
            ds_subset = ds_subset.climepi.sel_geopy(loc_str)
        else:
            if lon_range is not None:
                # Note the remote data are stored with longitudes in the range 0 to 360.
                if lon_range[0] < 0 <= lon_range[1]:
                    ds_subset = xr.concat(
                        [
                            ds_subset.sel(lon=slice(0, lon_range[1] % 360)),
                            ds_subset.sel(lon=slice(lon_range[0] % 360, 360)),
                        ],
                        dim="lon",
                    )
                else:
                    ds_subset = ds_subset.sel(
                        lon=slice(lon_range[0] % 360, lon_range[1] % 360)
                    )
            if lat_range is not None:
                ds_subset = ds_subset.sel(lat=slice(*lat_range))
        self._ds = ds_subset

    def _download_remote_data(self):
        # Download the remote dataset to a temporary file (printing a progress bar),
        # and then open the dataset from the temporary file (using the same chunking as
        # the remote dataset). Note that using a single file (which is later deleted
        # after saving the data to separate files) streamlines the download process
        # compared to downloading each realization directly to its own final file,
        # and may be more efficient, but a single download may not be desirable if
        # internet connection is unreliable.
        temp_save_dir = self._temp_save_dir
        temp_file_name = "temporary.nc"
        temp_save_path = temp_save_dir / temp_file_name
        delayed_obj = self._ds.to_netcdf(temp_save_path, compute=False)
        with dask.diagnostics.ProgressBar():
            delayed_obj.compute()
        self._temp_file_names = [temp_file_name]

    def _open_temp_data(self, **kwargs):
        kwargs = {"chunks": self._ds.chunks.mapping, **kwargs}
        super()._open_temp_data(**kwargs)

    def _process_data(self):
        # Process the remotely opened dataset, and store the processed dataset in the
        # _ds attribute.
        realizations = self._subset["realizations"]
        frequency = self._frequency
        ds_processed = self._ds.copy()
        # Index data variables by an integer realization coordinate instead of the
        # member_id coordinate (which is a string), and add model and scenario
        # coordinates.
        ds_processed = ds_processed.swap_dims({"member_id": "realization"})
        ds_processed["realization"] = realizations
        # ds_processed["scenario"] = self.available_scenarios
        # ds_processed["model"] = self.available_models
        # ds_processed = ds_processed.set_coords(["scenario", "model"])
        ds_processed = ds_processed.expand_dims(
            {"scenario": self.available_scenarios, "model": self.available_models}
        )
        # Make time bounds a data variable instead of a coordinate, and format in order
        # to match the conventions of xcdat.
        ds_processed = ds_processed.reset_coords("time_bnds")
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
        # Take yearly averages if requested.
        if frequency == "yearly":
            ds_processed = ds_processed.climepi.yearly_average()
        self._ds = ds_processed
        super()._process_data()