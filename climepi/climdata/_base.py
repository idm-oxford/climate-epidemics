"""Module defining get_climate_data and get_climate_data_file_names functions."""

from climepi.climdata._cesm import CESMDataGetter
from climepi.climdata._isimip import ISIMIPDataGetter


def get_climate_data(
    data_source,
    frequency="monthly",
    subset=None,
    save_dir=None,
    download=True,
    force_remake=False,
    subset_check_interval=10,
    max_subset_wait_time=20,
    **kwargs,
):
    """
    Retrieve and download climate projection data from a remote server.

    Currently available data sources are CESM2 LENS (data_source='lens2') and ISIMIP
    (data_source='isimip'). CESM2 LENS data are taken from an AWS server
    (https://registry.opendata.aws/ncar-cesm2-lens/), and terms of use can be found at
    https://www.ucar.edu/terms-of-use/data. ISIMIP data are taken from the ISIMIP
    repository (https://data.isimip.org/), and terms of use can be found at
    https://www.isimip.org/gettingstarted/terms-of-use/terms-use-publicly-available-isimip-data-after-embargo-period/.

    Parameters
    ----------
    data_source : str
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
            locations : str or list of str, optional
                Name of one or more locations for which to retrieve data. If provided,
                and the 'lon' and 'lat' parameters are not provided, OpenStreetMap data
                (https://openstreetmap.org/copyright) is used to query corresponding
                longitude and latitudes, and data for the nearest grid point to each
                location are retrieved). If 'lon' and 'lat' are also provided, these are
                used to retrieve the data (the locations parameter is still used as a
                dimension coordinate in the output dataset). If not provided, the
                'lon_range' and 'lat_range' parameters are used instead.
            lon: float or list of float, optional
                Longitude(s) for which to retrieve data. If provided, both 'locations'
                and 'lat' should also be provided. If 'locations' is a list, 'lon' and
                'lat' must also be lists of the same length (if provided).
            lat: float or list of float, optional
                Latitude(s) for which to retrieve data. If provided, both 'locations'
                and 'lon' should also be provided. If 'locations' is a list, 'lon' and
                'lat' must also be lists of the same length (if provided).
            lon_range : list or array-like of float, optional
                Longitude range for which to retrieve data. Should comprise two values
                giving the minimum and maximum longitudes. Ignored if 'locations' is
                provided. If not provided, and 'locations' is also not provided, all
                longitudes are retrieved.
            lat_range : list or array-like of float, optional
                Latitude range for which to retrieve data. Should comprise two values
                giving the minimum and maximum latitudes. Ignored if 'locations' is
                provided. If not provided, and 'locations' is also not provided, all
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
    subset_check_interval : int or float, optional
        For ISIMIP data only; time interval in seconds between checks for server-side
        data subsetting completion (default is 10).
    max_subset_wait_time : int or float, optional
        For ISIMIP data only; maximum time to wait for server-side data subsetting to
        complete, in seconds, before timing out (default is 20). Server-side subsetting
        will continue to run after this function times out, and this function can be
        re-run to check if the subsetting has completed and retrieve the subsetted data.
    **kwargs
        Additional keyword arguments to pass to xarray.open_mfdataset when opening
        downloaded data files.

    Returns
    -------
    xarray.Dataset
        Formatted climate projection dataset.
    """
    data_getter = _get_data_getter(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
        save_dir=save_dir,
        subset_check_interval=subset_check_interval,
        max_subset_wait_time=max_subset_wait_time,
    )
    ds_clim = data_getter.get_data(
        download=download, force_remake=force_remake, **kwargs
    )
    return ds_clim


def get_climate_data_file_names(data_source="lens2", frequency="monthly", subset=None):
    """
    Retrieve file names of formatted climate data files.

    File names are as created by the `get_climate_data` function.

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


def _get_data_getter(
    data_source, *args, subset_check_interval=None, max_subset_wait_time=None, **kwargs
):
    if data_source == "lens2":
        data_getter = CESMDataGetter(*args, **kwargs)
    elif data_source == "isimip":
        data_getter = ISIMIPDataGetter(
            *args,
            subset_check_interval=subset_check_interval,
            max_subset_wait_time=max_subset_wait_time,
            **kwargs,
        )
    else:
        raise ValueError(f"Data source '{data_source}' not supported.")
    return data_getter
