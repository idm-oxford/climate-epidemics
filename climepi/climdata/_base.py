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
