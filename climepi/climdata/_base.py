from climepi.climdata._cesm import CESMDataGetter
from climepi.climdata._isimip import ISIMIPDataGetter


def _get_data_getter(data_source, *args, **kwargs):
    if data_source == "lens2":
        data_getter = CESMDataGetter(*args, **kwargs)
    elif data_source == "isimip":
        data_getter = ISIMIPDataGetter(*args, **kwargs)
    else:
        raise ValueError(f"Data source '{data_source}' not supported.")
    return data_getter


def get_climate_data(
    data_source="lens2", frequency="monthly", subset=None, save_dir=None, download=True
):
    """
    Function to retrieve and (optionally) download climate data.

    Parameters
    ----------
    data_source : str, optional
        Data source to retrieve data from. Currently, only 'lens2' (for CESM2 LENS data)
        is supported.
    frequency : str, optional
        Frequency of the data to retrieve. Should be one of 'daily' or 'monthly'
        (default is 'monthly').
    subset: dict, optional
        Dictionary of data subsetting options.
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


def get_climate_data_file_names(
    data_source="lens2", frequency="monthly", subset=None, save_dir=None
):
    """
    Function to retrieve and (optionally) download climate data.

    Parameters
    ----------
    data_source : str, optional
        Data source to retrieve data from. Currently, only 'lens2' (for CESM2 LENS data)
        is supported.
    frequency : str, optional
        Frequency of the data to retrieve. Should be one of 'daily' or 'monthly'
        (default is 'monthly').
    subset: dict, optional
        Dictionary of data subsetting options.
    save_dir : str or pathlib.Path, optional
        Directory to which downloaded data are saved to and accessed from. If not
        provided, a directory within the OS cache directory is used.
    """
    data_getter = _get_data_getter(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
        save_dir=save_dir,
    )
    file_names = data_getter.file_names
    return file_names
