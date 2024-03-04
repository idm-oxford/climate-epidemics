import xarray as xr

from climepi.climdata._cesm import CESMDataGetter
from climepi.climdata._isimip import ISIMIPDataGetter


def get_climate_data(
    data_source, frequency="monthly", subset=None, save_dir=None, download=True
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
    download : bool, optional
        Whether to download the data if it is not found locally (default is False).
        Details of where downloaded data are saved to and accessed from are given in
        the CESMDataGetter class documentation.
    """
    if "loc_str" in subset and isinstance(subset["loc_str"], list):
        return _get_climate_data_loc_list(
            data_source, frequency, subset, save_dir, download
        )
    data_getter = _get_data_getter(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
        save_dir=save_dir,
    )
    ds_clim = data_getter.get_data(download=download)
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
    if "loc_str" in subset and isinstance(subset["loc_str"], list):
        return [
            file_name
            for loc_str_curr in subset["loc_str"]
            for file_name in get_climate_data_file_names(
                data_source, frequency, {**subset, "loc_str": loc_str_curr}
            )
        ]
    data_getter = _get_data_getter(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
    )
    file_names = data_getter.file_names
    return file_names


def _get_data_getter(data_source, *args, **kwargs):
    if data_source == "lens2":
        data_getter = CESMDataGetter(*args, **kwargs)
    elif data_source == "isimip":
        data_getter = ISIMIPDataGetter(*args, **kwargs)
    else:
        raise ValueError(f"Data source '{data_source}' not supported.")
    return data_getter


def _get_climate_data_loc_list(data_source, frequency, subset, save_dir, download):
    import functools
    import multiprocessing

    if data_source == "isimip":
        # Run for each location with a timeout to ensure server-side processing is
        # requested for each location (can then cancel the Python process and restart
        # once the server-side processing is complete).
        for loc_str_curr in subset["loc_str"]:
            subset_curr = {**subset, "loc_str": loc_str_curr}
            target = functools.partial(
                get_climate_data,
                data_source,
                frequency,
                subset_curr,
                save_dir,
                download,
            )
            p = multiprocessing.Process(target=target)
            p.start()
            p.join(60)
            if p.is_alive():
                p.terminate()
    ds_list = []
    for loc_str_curr in subset["loc_str"]:
        subset_curr = {**subset, "loc_str": loc_str_curr}
        ds_curr = get_climate_data(
            data_source, frequency, subset_curr, save_dir, download
        )
        ds_curr["location"] = [loc_str_curr]
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
        # Set CF x and y coords?
    ds = xr.concat(ds_list, dim="location", data_vars="minimal")
    return ds
