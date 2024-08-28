import itertools
import pathlib
import tempfile
import warnings

import numpy as np
import pooch
import xarray as xr
import xcdat

from climepi.utils import list_non_bnd_data_vars

# Cache directory for storing any temporary files created when downloading data.
# Note: the code could be improved to ensure that the temporary files are deleted if an
# error occurs, and to use a different temporary file name each time to avoid
# potential conflicts if multiple instances of the code are run simultaneously.
CACHE_DIR = pooch.os_cache("climepi")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ClimateDataGetter:
    """
    Class for accessing and downloading climate projection data. The 'get_data' method
    controls the process of finding,  downloading and formatting the data. Intended to
    be subclassed for specific data sources. Subclasses should define the below class
    attributes, as well as overriding and/or extending the methods as necessary.

    Class attributes
    ----------------
    data_source: str
        Name of the data source (e.g. 'lens2', 'isimip').
    remote_open_possible: bool
        Whether it is possible to lazily open the remote data as an xarray dataset (i.e.
        without first downloading the data), e.g. if the data are stored in a cloud-based
        file system.
    available_years: list or array-like of int
        Available years for which data can be retrieved.
    available_scenarios: list or array-like of str
        Available scenarios for which data can be retrieved.
    available_models: list or array-like of str
        Available models for which data can be retrieved.
    available_realizations: list or array-like of int
        Available realizations for which data can be retrieved (labelled as integers
        from 0).
    lon_res: float
        Longitude resolution of the data (degrees).
    lat_res: float
        Latitude resolution of the data (degrees).

    Parameters
    ----------
    frequency : str, optional
        Frequency of the data to retrieve. Should be one of 'daily', 'monthly' or
        'yearly' (default is 'monthly').
    subset : dict, optional
        Dictionary of data subsetting options. The following keys/values are available:
            years : list or array-like of int, optional
                Years for which to retrieve data within the available data range. If
                not provided, all years are retrieved.
            scenarios : list or array-like of str, optional
                Scenarios for which to retrieve data. If not provided, all available
                scenarios are retrieved.
            models : list or array-like of str, optional
                Models for which to retrieve data. If not provided, all available models
                are retrieved.
            realizations : list or array-like of int, optional
                Realizations for which to retrieve data. If not provided, all available
                realizations are retrieved.
            location : str or list of str, optional
                Name of one or more locations for which to retrieve data (for each
                provided location, `geopy` is used to query the corresponding longitude
                and latitude, and data for the nearest grid point are retrieved). If
                not provided, the 'lon_range' and 'lat_range' parameters are used
                instead.
            lon_range : list or array-like of float, optional
                Longitude range for which to retrieve data. Should comprise two values
                giving the minimum and maximum longitudes. Ignored if 'location' is
                provided. If not provided, and 'location' is also not provided, all
                longitudes are retrieved.
            lat_range : list or array-like of float, optional
                Latitude range for which to retrieve data. Should comprise two values
                giving the minimum and maximum latitudes. Ignored if 'location' is
                provided. If not provided, and 'location' is also not provided, all
                latitudes are retrieved.
    save_dir : str or pathlib.Path, optional
        Directory to which downloaded data are saved to and accessed from. If not
        provided, a directory within the OS cache directory is used.
    """

    data_source = None
    remote_open_possible = False
    available_years = None
    available_scenarios = None
    available_models = None
    available_realizations = None
    lon_res = None
    lat_res = None

    def __init__(self, frequency="monthly", subset=None, save_dir=None):
        subset_in = subset or {}
        subset = {
            "years": self.available_years,
            "scenarios": self.available_scenarios,
            "models": self.available_models,
            "realizations": self.available_realizations,
            "location": None,
            "lon_range": None,
            "lat_range": None,
        }
        subset.update(subset_in)
        self._frequency = frequency
        self._subset = subset
        self._ds = None
        self._temp_save_dir = None
        self._temp_file_names = None
        self._ds_temp = None
        if save_dir is None:
            save_dir = CACHE_DIR
        self._save_dir = pathlib.Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._file_name_dict = None
        self._file_names = None

    @property
    def file_name_dict(self):
        """
        Gets a dictionary mapping each scenario/model/realization combination to a file
        name for saving and retrieving the corresponding data (without the directory
        path). The file names are determined based on the provided data subsetting
        options.
        """
        if self._file_name_dict is None:
            years = self._subset["years"]
            scenarios = self._subset["scenarios"]
            models = self._subset["models"]
            realizations = self._subset["realizations"]
            location = self._subset["location"]
            lon_range = self._subset["lon_range"]
            lat_range = self._subset["lat_range"]
            base_name_str_list = [self.data_source, self._frequency]
            if all(np.diff(years) == 1):
                base_name_str_list.extend([f"{years[0]}", "to", f"{years[-1]}"])
            elif np.size(years) <= 10:
                base_name_str_list.extend([f"{year}" for year in years])
            elif all(np.diff(np.diff(years)) == 0):
                base_name_str_list.extend(
                    [f"{years[0]}", "by", f"{np.diff(years)[0]}", "to", f"{years[-1]}"]
                )
            else:
                base_name_str_list.extend([f"{year}" for year in years])
                warnings.warn(
                    "Requesting a large number of non-uniform years may lead to "
                    "invalid long file names. Consider separating the request into "
                    "smaller chunks.",
                    stacklevel=2,
                )
            if location is not None:
                base_name_str_list.append(location.replace(" ", "_"))
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
                name_str_list = base_name_str_list + [
                    scenario,
                    model,
                    f"{realization}.nc",
                ]
                name_str = "_".join(name_str_list)
                file_name_dict[scenario][model][realization] = name_str
            self._file_name_dict = file_name_dict
        return self._file_name_dict

    @property
    def file_names(self):
        """
        Gets a list of file names for saving and retrieving the data for the included
        scenario/model/realization combinations (see 'file_name_dict' property).
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

    def get_data(self, download=True, force_remake=False):
        """
        Main method for retrieving data. First tries to open the data locally from
        the provided 'save_dir' directory. If not found locally, the data are searched
        for and subsetted within the remote server, downloaded to a temporary file
        (optionally, if it is possible to lazily open the remote dataset), and then
        processed and (if downloaded) saved to the 'save_dir' directory.

        Parameters
        ----------
        download : bool, optional
            Whether to download the data to the 'save_dir' directory if not found
            locally (default is True). If False and the data are not found locally,
            the remotely held data are only lazily opened and processed (provided this
            is possible).
        force_remake : bool, optional
            Whether to force re-download and re-formatting of the data even if the data
            exist locally (default is False). Can only be used if 'download' is True.
        Returns
        -------
        xarray.Dataset
            Processed data (lazily opened from either local or remote files)
        """
        if not download and force_remake:
            raise ValueError("Cannot force remake if download is False.")
        if not force_remake:
            try:
                self._open_local_data()
                return self._ds
            except FileNotFoundError:
                pass
        if not self.remote_open_possible and not download:
            raise ValueError(
                "It is not possible to lazily load the remote data. Set download=True ",
                "to download the data.",
            )
        print("Finding data files on server...")
        self._find_remote_data()
        print("Data found.")
        print("Subsetting data...")
        self._subset_remote_data()
        print("Data subsetted.")
        if download:
            self._temp_save_dir = pathlib.Path(tempfile.mkdtemp(suffix="_climepi"))
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
        save_dir = self._save_dir
        file_names = self.file_names
        _ds = xcdat.open_mfdataset(
            [save_dir / file_name for file_name in file_names], chunks={}
        )
        if "time_bnds" in _ds:
            # Load time bounds to avoid bug saving to file caused by encoding not being
            # set [REVISIT IN FUTURE]
            _ds.time_bnds.load()
        self._ds = _ds

    def _find_remote_data(self):
        # Method for finding the data on the remote server to be implemented in
        # subclasses. The expected behaviour depends on the data source.
        raise NotImplementedError

    def _subset_remote_data(self):
        # Method for subsetting the remotely held data to be implemented in subclasses.
        # The expected behaviour depends on the data source.
        raise NotImplementedError

    def _download_remote_data(self):
        # Method for downloading the remotely held data to be implemented in subclasses.
        # Should download the data to temporary netCDF file(s) and store the file
        # name(s) in the _temp_file_names attribute.
        raise NotImplementedError

    def _open_temp_data(self, **kwargs):
        # Open the downloaded data from the temporary file(s), and store the dataset in
        # both the _ds attribute and the _ds_temp attribute (the latter is used for
        # closing the temporary file(s) before they are deleted). The 'kwargs' argument
        # is included to allow for different options to be passed to
        # xarray.open_mfdataset by subclasses which extend this method.
        kwargs = {"data_vars": "minimal", "chunks": {}, **kwargs}
        temp_save_dir = self._temp_save_dir
        temp_file_names = self._temp_file_names
        temp_file_paths = [
            temp_save_dir / temp_file_name for temp_file_name in temp_file_names
        ]
        self._ds_temp = xr.open_mfdataset(temp_file_paths, **kwargs)
        self._ds = self._ds_temp

    def _process_data(self):
        # Process the downloaded dataset, and store the processed dataset in the _ds
        # attribute. Processing common to all data sources is implemented here; this
        # method can be extended (or overridden) by subclasses to include data source-
        # specific processing.
        ds_processed = self._ds.copy()
        # Add latitude and longitude bounds (use provided resolution if available, else
        # use the xcdat `add_missing_bounds` method to infer bounds from the coordinate
        # values (which is not possible for single-value coordinates).
        lon_res = self.lon_res
        lat_res = self.lat_res
        non_bnd_data_vars = list_non_bnd_data_vars(ds_processed)
        if len(non_bnd_data_vars) == 1:
            # Assignment below fails with a length-1 list
            non_bnd_data_vars = non_bnd_data_vars[0]
        if "lon" not in ds_processed.dims:
            # Ensure lon is a dim so it appears as a dim of its bounds variable
            ds_processed[non_bnd_data_vars] = ds_processed[
                non_bnd_data_vars
            ].expand_dims("lon")
        if "lat" not in ds_processed.dims:
            ds_processed[non_bnd_data_vars] = ds_processed[
                non_bnd_data_vars
            ].expand_dims("lat")
        if "lon_bnds" not in ds_processed and lon_res is not None:
            ds_processed["lon_bnds"] = xr.concat(
                [ds_processed.lon - lon_res / 2, ds_processed.lon + lon_res / 2],
                dim="bnds",
            ).T
            ds_processed["lon"].attrs.update(bounds="lon_bnds")
        if "lat_bnds" not in ds_processed and lat_res is not None:
            ds_processed["lat_bnds"] = xr.concat(
                [ds_processed.lat - lat_res / 2, ds_processed.lat + lat_res / 2],
                dim="bnds",
            ).T
            ds_processed["lat"].attrs.update(bounds="lat_bnds")
        ds_processed = ds_processed.bounds.add_missing_bounds(axes=["X", "Y"])
        # Convert the longitude coordinate to the range -180 to 180 (MAY REMOVE IN
        # FUTURE) - note this should be done after adding bounds to avoid issues on
        # the boundaries
        ds_processed = xcdat.swap_lon_axis(ds_processed, to=(-180, 180))
        # Use degree symbol for units of latitude and longitude (for nicer plotting)
        ds_processed["lon"].attrs.update(units="°E")
        ds_processed["lat"].attrs.update(units="°N")
        self._ds = ds_processed

    def _save_processed_data(self):
        # Save the data for each scenario/model/realization combination to a separate
        # file in the 'save_dir' directory.
        scenarios = self._subset["scenarios"]
        models = self._subset["models"]
        realizations = self._subset["realizations"]
        save_dir = self._save_dir
        file_name_dict = self.file_name_dict
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
        # have been processed and saved to final files).
        temp_save_dir = self._temp_save_dir
        temp_file_names = self._temp_file_names
        self._ds_temp.close()
        for temp_file_name in temp_file_names:
            temp_save_path = temp_save_dir / temp_file_name
            temp_save_path.unlink()
        temp_save_dir.rmdir()
        self._ds_temp = None
        self._temp_save_dir = None
        self._temp_file_names = None
