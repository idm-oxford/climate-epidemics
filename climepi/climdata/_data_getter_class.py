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
    frequency : str, optional
        Frequency of the data to retrieve. Should be one of 'daily' or 'monthly'
        (default is 'monthly').
    subset : dict, optional
        Dictionary of data subsetting options. The following keys/values are available:
            years : list or array-like of int, optional
                Years for which to retrieve data within the available data range. If
                not provided, all years are retrieved.
            realizations : list or array-like of int, optional
                Realizations for which to retrieve data. If not provided, all available
                realizations are retrieved.
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

    data_source = None
    available_years = None
    available_scenarios = None
    available_models = None
    available_realizations = None

    def __init__(self, frequency="monthly", subset=None, save_dir=None):
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
        self._frequency = frequency
        self._subset = subset
        self._ds = None
        self._temp_save_dir = pathlib.Path(CACHE_DIR) / "temp"
        self._temp_file_names = None
        if save_dir is None:
            save_dir = CACHE_DIR
        self._save_dir = pathlib.Path(save_dir)
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
            base_name_str_list = [self.data_source, self._frequency]
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
            file_name_dict = {
                scenario: {
                    model: {realization: None for realization in realizations}
                    for model in models
                }
                for scenario in scenarios
            }
            for scenario, model, realization in itertools.product(
                scenarios, models, realizations
            ):
                name_str_list = base_name_str_list.copy()
                name_str_list.extend([scenario, model, f"{realization}.nc"])
                name_str = "_".join(name_str_list)
                file_name_dict[scenario][model][realization] = name_str
            self._file_name_dict = file_name_dict
        return self._file_name_dict

    @property
    def file_names(self):
        """
        Gets a list of file names for saving and retrieving the data for the included
        realizations (see 'file_name_dict' property).
        """
        if self._file_names is None:
            scenarios = self._subset["scenarios"]
            models = self._subset["models"]
            realizations = self._subset["realizations"]
            file_name_dict = self.file_name_dict
            self._file_names = [
                file_name_dict[scenario][model][realization]
                for scenario, model, realization in itertools.product(
                    scenarios, models, realizations
                )
            ]
        return self._file_names

    def get_data(self, download=True):
        """
        Main method for retrieving data. First tries to open the data locally from
        the provided 'save_dir' directory. If the dataset is not found locally, it is
        opened and subsetted within the remote server, (optionally) downloaded,
        processed and (if downloaded) saved to the 'save_dir' directory.

        Main method for retrieving data. First tries to open the data locally from
        the provided 'save_dir' directory. If the data are not found locally, it is
        opened, subsetted and processed from the remote aws server, and then
        (optionally) downloaded to the 'save_dir' directory.

        Parameters
        ----------
        download : bool, optional
            Whether to download the data to the 'save_dir' directory if not found
            locally (default is True). If False and the data are not found locally,
            the data are only lazily opened and processed from the remote server
            (provided this is possible).

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
        print("Getting remote data info...")
        self._find_remote_data()
        print("Remote data found.")
        print("Subsetting data...")
        self._subset_remote_data()
        print("Data subsetted.")
        if download:
            print("Downloading data...")
            self._download_remote_data()
            print("Data downloaded.")
            self._open_temp_data()
        self._process_data()
        if download:
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

    def _open_temp_data(self, chunks=None):
        if chunks is None:
            chunks = {}
        temp_save_dir = self._temp_save_dir
        temp_file_names = self._temp_file_names
        temp_file_paths = [
            temp_save_dir / temp_file_name for temp_file_name in temp_file_names
        ]
        self._ds = xr.open_mfdataset(temp_file_paths, chunks=chunks)

    def _process_data(self):
        # Process the remotely opened dataset, and store the processed dataset in the
        # _ds attribute.
        ds_processed = self._ds.copy()
        # Process the remotely opened dataset, and store the processed dataset in the
        # _ds attribute.
        # Convert the longitude coordinate to the range -180 to 180 (MAY REMOVE IN
        # FUTURE)
        if ds_processed.lon.size > 1:
            ds_processed = xcdat.swap_lon_axis(ds_processed, to=(-180, 180))
        else:
            ds_processed["lon"] = ((ds_processed["lon"] + 180) % 360) - 180
        # Add latitude and longitude bounds.
        ds_processed = ds_processed.bounds.add_missing_bounds(axes=["X", "Y"])
        self._ds = ds_processed

    def _save_processed_data(self):
        # Save the data for each realization to a separate file in the 'save_dir'
        # directory.
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        realizations = self._subset["realizations"]
        save_dir = self._save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name_dict = self._file_name_dict
        ds_all = self._ds
        for scenario, model, realization in itertools.product(
            scenarios, models, realizations
        ):
            ds_curr = ds_all.sel(
                realization=[realization], scenario=[scenario], model=[model]
            )
            save_path = save_dir / file_name_dict[scenario][model][realization]
            ds_curr.to_netcdf(save_path)

    def _delete_temporary(self):
        # Delete the temporary file(s) created when downloading the data (once the data
        # have been saved to separate files).
        temp_save_dir = self._temp_save_dir
        temp_file_names = self._temp_file_names
        self._ds.close()
        for temp_file_name in temp_file_names:
            temp_save_path = temp_save_dir / temp_file_name
            temp_save_path.unlink()
