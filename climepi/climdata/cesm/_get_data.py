"""
Module for accessing and downloading CESM LENS2 data from the aws server (see
https://ncar.github.io/cesm2-le-aws/model_documentation.html).
"""

import pathlib

import dask.diagnostics
import intake
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


class CESMDataGetter:
    """
    Class for accessing and (optionally) downloading CESM LENS2 data from the aws server
    (see https://ncar.github.io/cesm2-le-aws/model_documentation.html), and for
    retrieving downloaded data. The 'get_data' method controls the process of
    finding, formatting and downloading the data.

    Parameters
    ----------
    subset : dict, optional
        Dictionary of data subsetting options. The following keys/values are available:
            years : list or array-like of int, optional
                Years for which to retrieve data within the data range (1850-2100). If
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

    def __init__(
        self,
        subset=None,
        save_dir=None,
    ):
        subset_in = subset or {}
        subset = {
            "years": np.arange(1850, 2101),
            "realizations": np.arange(100),
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
        self._catalog = None
        self._file_name_dict = None
        self._file_names = None

    @property
    def catalog(self):
        """
        Gets an intake-esm catalog describing the available CESM LENS2 data.
        """
        if self._catalog is None:
            self._catalog = intake.open_esm_datastore(
                "https://raw.githubusercontent.com/NCAR/cesm2-le-aws"
                + "/main/intake-catalogs/aws-cesm2-le.json"
            )
        return self._catalog

    @property
    def file_name_dict(self):
        """
        Gets a dictionary mapping each included realization to a file name for saving
        and retrieving the data for that realization (without the directory path).
        The file names are determined based on the provided data subsetting options.
        """
        if self._file_name_dict is None:
            years = self._subset["years"]
            realizations = self._subset["realizations"]
            loc_str = self._subset["loc_str"]
            lon_range = self._subset["lon_range"]
            lat_range = self._subset["lat_range"]
            base_name_str_list = ["cesm"]
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
            for realization in realizations:
                name_str_list = base_name_str_list.copy()
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

    def get_data(self, download=False):
        """
        Main method for retrieving data. First tries to open the data locally from
        the provided 'save_dir' directory. If the data are not found locally, it is
        opened, subsetted and processed from the remote aws server, and then
        (optionally) downloaded to the 'save_dir' directory.

        Parameters
        ----------
        download : bool, optional
            Whether to download the data to the 'save_dir' directory if it is not
            found locally (default is False).

        Returns
        -------
        xarray.Dataset
            Retrieved data. If the data was found locally or downloaded with
            download=True), a dask-backed dataset opened from the local files is
            returned. Otherwise, a dask-backed dataset opened from the remote server
            (lazily subsetted and processed) is returned.
        """
        try:
            self._open_local_data()
            return self._ds
        except FileNotFoundError:
            pass
        print("Opening remote data...")
        self._open_remote_data()
        print("\n\nRemote data opened.")
        self._subset_data()
        self._process_data()
        if download:
            print("Downloading data...")
            self._download()
            print("Data downloaded.")
            self._save()
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

    def _open_remote_data(self):
        # Open all monthly data for the required data variables (temperature values
        # are contained in the 'TREFHT' variable, and precipitation values are
        # contained in the 'PRECC' and 'PRECL' variables) from the aws server.
        # Historical and future data, and data for two slightly different forcing
        # scenarios (cmip6 and smbb), are combined into a single xarray dataset, which
        # is stored in the _ds attribute.
        catalog = self.catalog
        catalog_subset = catalog.search(
            variable=["TREFHT", "PRECC", "PRECL"], frequency="monthly"
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

    def _subset_data(self):
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

    def _download(self):
        # Download the remote dataset to a temporary file (printing a progress bar),
        # and then open the dataset from the temporary file (using the same chunking as
        # the remote dataset). Note that using a single file (which is later deleted
        # after saving the data to separate files) streamlines the download process
        # compared to downloading each realization directly to its own final file,
        # and may be more efficient, but a single download may not be desirable if
        # internet connection is unreliable.
        temporary_save_path = pathlib.Path(CACHE_DIR) / "temporary.nc"
        delayed_obj = self._ds.to_netcdf(temporary_save_path, compute=False)
        with dask.diagnostics.ProgressBar():
            delayed_obj.compute()
        chunks = self._ds.chunks.mapping
        self._ds = xr.open_dataset(temporary_save_path, chunks=chunks)

    def _save(self):
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
        # has been saved to separate files).
        temporary_save_path == pathlib.Path(CACHE_DIR) / "temporary.nc"
        self._ds.close()
        temporary_save_path.unlink()


def get_cesm_data(subset=None, save_dir=None, download=False):
    """
    Function to retrieve and (optionally) download CESM LENS2 data using the
    CESMDataGetter class.

    Parameters
    ----------
    subset: dict, optional
        Dictionary of data subsetting options passed to the CESMDataGetter constructor.
        See the CESMDataGetter class documentation for details.
    save_dir : str or pathlib.Path, optional
        Directory to which downloaded data are saved to and accessed from. If not
        provided, a directory within the OS cache directory is used.
    download : bool, optional
        Whether to download the data if it is not found locally (default is False).
        Details of where downloaded data are saved to and accessed from are given in
        the CESMDataGetter class documentation.
    """
    data_getter = CESMDataGetter(save_dir=save_dir, subset=subset)
    ds_cesm = data_getter.get_data(download=download)
    return ds_cesm
