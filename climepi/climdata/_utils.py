import itertools
import pathlib

import numpy as np
import pooch
import xarray as xr
import xcdat

# Cache directory for storing any temporary files created when downloading data.
# Note: the code could be improved to ensure that the temporary files are deleted if an
# error occurs, and to use a different temporary file name each time to avoid
# potential conflicts if multiple instances of the code are run simultaneously.
CACHE_DIR = pooch.os_cache("climepi")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ClimateDataGetter:
    """
    Class for accessing downloading climate projection data. The 'get_data' method
    controls the process of finding,  downloading and formatting the data. Intended to
    be subclassed for specific data sources.

    Parameters
    ----------
    subset : dict, optional
        Dictionary of data subsetting options. The following keys/values are available:
            years : list or array-like of int, optional
                Years for which to retrieve data within the available data range. If
                not provided, all years are retrieved.
            realizations : list or array-like of int, optional
                Realizations for which to retrieve data, out of the available 100
                realizations numbered 0 to 99. If not provided, all realizations are
                retrieved.
            loc_str : str, optional
                Name of a single location for which to retrieve data. If not provided,
                the 'lon_range' and 'lat_range' parameters are used instead.
            lon_range : list or array-like of float, optional
                Longitude range for which to retrieve data. Should comprise two values
                giving the minimum and maximum longitudes. Ignored if 'loc_str' is
                provided. If not provided, and 'loc_str' is also not provided, all
                longitudes are retrieved.
            lat_range : list or array-like of float, optional
                Latitude range for which to retrieve data. Should comprise two values
                giving the minimum and maximum latitudes. Ignored if 'loc_str' is
                provided. If not provided, and 'loc_str' is also not provided, all
                latitudes are retrieved.
    save_dir : str or pathlib.Path, optional
        Directory to which downloaded data are saved to and accessed from. If not
        provided, a directory within the OS cache directory is used.
    """

    datasource_name = None
    available_years = None
    available_scenarios = None
    available_models = None
    available_realizations = None

    def __init__(
        self,
        subset=None,
        save_dir=None,
    ):
        subset_in = subset or {}
        subset = {
            "years": self.available_years,
            "scenarios": self.available_scenarios,
            "models": self.available_models,
            "realizations": self.available_realizations,
            "loc_str": None,
            "lon_range": None,
            "lat_range": None,
        }
        subset.update(subset_in)
        self._subset = subset
        if save_dir is None:
            save_dir = CACHE_DIR
        self._save_dir = save_dir
        self._ds = None
        self._file_name_dict = None
        self._file_names = None

    @property
    def file_name_dict(self):
        """
        Gets a dictionary mapping each included realization to a file name for saving
        and retrieving the data for that realization (without the directory path).
        The file names are determined based on the provided data subsetting options.
        """
        if self._file_name_dict is None:
            years = self._subset["years"]
            scenarios = self._subset["scenarios"]
            models = self._subset["models"]
            realizations = self._subset["realizations"]
            loc_str = self._subset["loc_str"]
            lon_range = self._subset["lon_range"]
            lat_range = self._subset["lat_range"]
            base_name_str_list = [self.datasource_name]
            if all(np.diff(years) == 1):
                base_name_str_list.extend([f"{years[0]}", "to", f"{years[-1]}"])
            else:
                base_name_str_list.extend([f"{year}" for year in years])
            if loc_str is not None:
                base_name_str_list.append(loc_str.replace(" ", "_"))
            else:
                if lon_range is not None:
                    base_name_str_list.extend(
                        ["lon", f"{lon_range[0]}", "to", f"{lon_range[1]}"]
                    )
                if lat_range is not None:
                    base_name_str_list.extend(
                        ["lat", f"{lat_range[0]}", "to", f"{lat_range[1]}"]
                    )
            file_name_dict = {}
            for scenario, model, realization in itertools.product(
                scenarios, models, realizations
            ):
                name_str_list = base_name_str_list.copy()
                name_str_list.extend(["scenario", scenario])
                name_str_list.extend(["model", model])
                name_str_list.extend(["realization", f"{realization}.nc"])
                name_str = "_".join(name_str_list)
                file_name_dict[realization] = name_str
            self._file_name_dict = file_name_dict
        return self._file_name_dict

    @property
    def file_names(self):
        """
        Gets a list of file names for saving and retrieving the data for the included
        realizations (see 'file_name_dict' property).
        """
        if self._file_names is None:
            self._file_names = list(self.file_name_dict.values())
        return self._file_names

    def get_data(self):
        """
        Main method for retrieving data. First tries to open the data locally from
        the provided 'save_dir' directory. If the dataset is not found locally, it is
        opened and subsetted within the remote server, and then downloaded, processed
        and saved to the 'save_dir' directory.

        Returns
        -------
        xarray.Dataset
            Retrieved data (dask-backed dataset opened from the found or downloaded
            local file)
        """
        try:
            self._open_local_data()
            return self._ds
        except FileNotFoundError:
            pass
        print("Opening remote data...")
        self._find_remote_data()
        print("\n\nRemote data found.")
        print("Subsetting data...")
        self._subset_remote_data()
        print("Data subsetted.")
        print("Downloading data...")
        self._download_remote_data()
        print("Data downloaded.")
        self._open_temp_data()
        self._process_data()
        self._save_processed_data()
        print(f"Formatted data saved to {self._save_dir}")
        self._delete_temporary()
        self._open_local_data()
        return self._ds

    def _open_local_data(self):
        # Open the data from the local files (will raise FileNotFoundError if any
        # files are not found), and store the dataset in the _ds attribute.
        save_dir = pathlib.Path(self._save_dir)
        file_names = self.file_names
        _ds = xcdat.open_mfdataset(
            [save_dir / file_name for file_name in file_names], chunks={}
        )
        self._ds = _ds

    def _find_remote_data(self):
        raise NotImplementedError

    def _subset_remote_data(self):
        raise NotImplementedError

    def _download_remote_data(self):
        raise NotImplementedError

    def _open_temp_data(self):
        raise NotImplementedError

    def _process_data(self):
        # Process the remotely opened dataset, and store the processed dataset in the
        # _ds attribute.
        realizations = self._subset["realizations"]
        ds_processed = self._ds.copy()
        # Index data variables by an integer realization coordinate instead of the
        # member_id coordinate (which is a string).
        ds_processed = ds_processed.swap_dims({"member_id": "realization"})
        ds_processed["realization"] = realizations
        # Convert the longitude coordinate to the range -180 to 180 (MAY REMOVE IN
        # FUTURE)
        if ds_processed.lon.size > 1:
            ds_processed = xcdat.swap_lon_axis(ds_processed, to=(-180, 180))
        else:
            ds_processed["lon"] = ((ds_processed["lon"] + 180) % 360) - 180
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
        # Add latitude and longitude bounds.
        ds_processed = ds_processed.bounds.add_missing_bounds(axes=["X", "Y"])
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

    def _save_processed_data(self):
        # Save the data for each realization to a separate file in the 'save_dir'
        # directory.
        realizations = self._subset["realizations"]
        save_dir = pathlib.Path(self._save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name_dict = self._file_name_dict
        ds_all = self._ds
        for realization in realizations:
            ds_curr = ds_all.sel(realization=[realization])
            save_path = save_dir / file_name_dict[realization]
            ds_curr.to_netcdf(save_path)

    def _delete_temporary(self):
        # Delete the temporary file created when downloading the data (once the data
        # have been saved to separate files).
        temporary_save_path = pathlib.Path(CACHE_DIR) / "temporary.nc"
        self._ds.close()
        temporary_save_path.unlink()
