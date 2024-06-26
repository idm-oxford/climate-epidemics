import xarray as xr

from climepi.climdata._cesm import CESMDataGetter
from climepi.climdata._isimip import ISIMIPDataGetter


def get_climate_data(
    data_source,
    frequency="monthly",
    subset=None,
    save_dir=None,
    download=True,
    force_remake=False,
    max_subset_wait_time=20,
):
    """
    Function to retrieve and download climate projection data from a remote server.

    Parameters
    ----------
    data_source : str, optional
        Data source to retrieve data from. Currently supported sources are 'lens2' (for
        CESM2 LENS data) and 'isimip' (for ISIMIP data).
    frequency : str, optional
        Frequency of the data to retrieve. Should be one of 'daily', 'monthly' or
        'yearly' (default is 'monthly').
    subset: dict, optional
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
    download : bool, optional
        For CESM2 LENS data only; whether to download the data to the 'save_dir'
        directory if not found locally (default is True). If False and the data are not
        found locally, a lazily opened xarray dataset linked to the remote data is
        returned. For ISIMIP data, the data must be downloaded if not found locally.
    force_remake : bool, optional
        Whether to force re-download and re-formatting of the data even if found
        locally (default is False). Can only be used if 'download' is True.
    max_subset_wait_time : int or float, optional
        For ISIMIP data only; maximum time to wait for server-side data subsetting to
        complete, in seconds, before timing out (default is 20). Server-side subsetting
        will continue to run after this function times out, and this function can be
        re-run to check if the subsetting has completed and retrieve the subsetted data.

    Returns
    -------
    xarray.Dataset
        Formatted climate projection dataset.
    """
    if "location" in subset and isinstance(subset["location"], list):
        return _get_climate_data_location_list(
            data_source=data_source,
            frequency=frequency,
            subset=subset,
            save_dir=save_dir,
            download=download,
            force_remake=force_remake,
        )
    data_getter = _get_data_getter(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
        save_dir=save_dir,
        max_subset_wait_time=max_subset_wait_time,
    )
    ds_clim = data_getter.get_data(download=download, force_remake=force_remake)
    return ds_clim


def get_climate_data_file_names(data_source="lens2", frequency="monthly", subset=None):
    """
    Function to retrieve file names of formatted climate data files created by the
    `get_climate_data` function.

    Parameters
    ----------
    data_source : str, optional
        Data source. Currently supported sources are 'lens2' (for CESM2 LENS data) and
        'isimip' (for ISIMIP data).
    frequency : str, optional
        Frequency of the data. Should be one of 'daily', 'monthly' or 'yearly' (default
        is 'monthly').
    subset: dict, optional
        Dictionary of data subsetting options. See the docstring of `get_climate_data`
        for details.
    """
    if "location" in subset and isinstance(subset["location"], list):
        return [
            file_name
            for location_curr in subset["location"]
            for file_name in get_climate_data_file_names(
                data_source=data_source,
                frequency=frequency,
                subset={**subset, "location": location_curr},
            )
        ]
    data_getter = _get_data_getter(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
    )
    file_names = data_getter.file_names
    return file_names


def _get_data_getter(data_source, *args, max_subset_wait_time=None, **kwargs):
    if data_source == "lens2":
        data_getter = CESMDataGetter(*args, **kwargs)
    elif data_source == "isimip":
        data_getter = ISIMIPDataGetter(
            *args, max_subset_wait_time=max_subset_wait_time, **kwargs
        )
    else:
        raise ValueError(f"Data source '{data_source}' not supported.")
    return data_getter


def _get_climate_data_location_list(
    data_source,
    frequency=None,
    subset=None,
    save_dir=None,
    download=None,
    force_remake=None,
):
    ds_list = []
    for location_curr in subset["location"]:
        try:
            subset_curr = {**subset, "location": location_curr}
            ds_curr = get_climate_data(
                data_source, frequency, subset_curr, save_dir, download, force_remake
            )
            ds_curr["location"] = [location_curr]
            for data_var in ds_curr.data_vars:
                if data_var == "time_bnds":
                    continue
                if "lon" in ds_curr[data_var].dims:
                    ds_curr[data_var] = ds_curr[data_var].squeeze("lon")
                if "lat" in ds_curr[data_var].dims:
                    ds_curr[data_var] = ds_curr[data_var].squeeze("lat")
                ds_curr[data_var] = ds_curr[data_var].expand_dims("location", axis=0)
            if "lon" in ds_curr.dims:
                ds_curr["lon"] = ds_curr["lon"].swap_dims({"lon": "location"})
            if "lat" in ds_curr.dims:
                ds_curr["lat"] = ds_curr["lat"].swap_dims({"lat": "location"})
            ds_list.append(ds_curr)
        except TimeoutError as exc:
            print(f"{exc}")
            print(f"Skipping location '{location_curr}' for now.")
    # Set CF x and y coords?
    ds = xr.concat(ds_list, dim="location", data_vars="minimal")
    return ds
