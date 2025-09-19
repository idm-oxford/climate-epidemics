"""Module defining get_climate_data and get_climate_data_file_names functions."""

import os
from typing import Any, Literal

import xarray as xr

from climepi.climdata._cesm import ARISEDataGetter, GLENSDataGetter, LENS2DataGetter
from climepi.climdata._data_getter_class import (
    ClimateDataGetter,
    SubsetPartial,
)
from climepi.climdata._isimip import ISIMIPDataGetter


def get_climate_data(
    data_source: Literal["lens2", "arise", "glens", "isimip"],
    frequency: Literal["daily", "monthly", "yearly"] = "monthly",
    subset: SubsetPartial | None = None,
    save_dir: str | os.PathLike | None = None,
    download: bool = True,
    force_remake: bool = False,
    subset_check_interval: float = 10,
    max_subset_wait_time: float = 20,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Retrieve climate projection data.

    Remotely stored climate projection data are formatted and downloaded (if
    ``download=True``). If data have been downloaded previously (and
    ``force_remake=False``), rerunning this function will simply open the local data
    files.

    Currently available data sources are:

    - LENS2 (``data_source='lens2'``): Community Earth System Model (CESM) version 2
    Large Ensemble (LENS2) data (https://www.cesm.ucar.edu/community-projects/lens2).
    Possible values for subsetting options (items in subset) are (all ranges inclusive):

    - years: 1850–2100 (historical and future data combined for convenience).
    - scenarios: 'ssp370' (historical and SSP3-7.0).
    - models: 'cesm2'.
    - realizations: 0–99 (100 ensemble members).

    Data are available from AWS (https://registry.opendata.aws/ncar-cesm2-lens/).
    Terms of data use are available at https://www.ucar.edu/terms-of-use/data.

    - ARISE-SAI (``data_source='arise'``): CESM2 Assessing Responses and Impacts of
    Solar intervention on the Earth system with Stratospheric Aerosol Injection
    (ARISE-SAI) data (https://www.cesm.ucar.edu/community-projects/arise-sai). Possible
    values for subsetting options (all ranges inclusive):

    - years: 2035–2069 (feedback simulations), 2015–2100 (reference simulations 0–4),
        2015–2069 (reference simulations 5–9).
    - scenarios: 'sai15' (SAI-1.5 scenario with climate intervention) and 'ssp245'
        (SSP2-4.5 reference scenario without intervention).
    - models: 'cesm2'.
    - realizations: 0–9 (10 ensemble members per scenario).

    Data are available from AWS (https://registry.opendata.aws/ncar-cesm2-arise/).
    Terms of data use are available at https://www.ucar.edu/terms-of-use/data.

    - GLENS (``data_source='glens'``): CESM1 Geoengineering Large Ensemble (GLENS) data
    (https://www.cesm.ucar.edu/community-projects/glens). Possible values for
    subsetting options (all ranges inclusive):

    - years: 2020–2099 (feedback simulations), 2010–2097 (reference simulations 0–2),
        2010–2030 (reference simulations 3–19).
    - scenarios: 'sai' (feedback simulations with climate intervention) and 'rcp85'
        (RCP8.5 reference scenario without intervention).
    - models: 'cesm1'.
    - realizations: 0–19 (20 ensemble members per scenario).

    Data are available from the NSF NCAR Research Data Archive
    (https://rda.ucar.edu/datasets/d651064/).
    Terms of data use are available at https://www.ucar.edu/terms-of-use/data.

    - ISIMIP (``data_source='isimip'``): Inter-Sectoral Impact Model Intercomparison
    Project Phase 3b (ISIMIP3b) data (https://www.isimip.org/). Possible values for
    subsetting options (all ranges inclusive):

    - years: 2015–2100.
    - scenarios: 'ssp126' (SSP1-2.6), 'ssp370' (SSP3-7.0), 'ssp585' (SSP5-8.5) for all
        models, and 'ssp245' (SSP2-4.5) for the first five models listed below.
    - models: 'gfdl-esm4', 'ipsl-cm6a-lr', 'mpi-esm1-2-hr', 'mri-esm2-0', 'ukesm1-0-ll',
        'canesm5', 'cnrm-cm6-1', 'cnrm-esm2-1', 'ec-earth3', 'miroc6'. Details of models
        and bias adjustment are provided at
        https://www.isimip.org/gettingstarted/isimip3b-bias-adjustment/.
    - realizations: 0 (single ensemble member per model/scenario pair).

    Data are available from the ISIMIP repository (https://data.isimip.org/).
    Terms of data use are available at
    https://www.isimip.org/gettingstarted/terms-of-use/terms-use-publicly-available-isimip-data-after-embargo-period/.


    For each data source, global data are available. Server-side spatial subsetting can
    be performed for LENS2, ARISE, and ISIMIP data (see subsetting options below). For
    GLENS data, no server-side subsetting is performed, and downloaded full data files
    are only cleared up after all data have been downloaded and processed; splitting the
    data retrieval into multiple calls to this function may reduce the disk space
    overhead.

    Parameters
    ----------
    data_source : str
        Data source to retrieve data from. Currently supported sources are 'lens2' (for
        LENS2 data), 'arise' (ARISE data), 'glens' (GLENS data), and 'isimip' (ISIMIP
        data).
    frequency : str, optional
        Frequency of the data to retrieve. Should be one of 'daily', 'monthly' or
        'yearly' (default is 'monthly').
    subset: dict, optional
        Dictionary of data subsetting options. The following keys/values are available:
            years : list of int, optional
                Years for which to retrieve data within the available data range. If
                not provided, all years are retrieved.
            scenarios : list of str, optional
                Scenarios for which to retrieve data. If not provided, all available
                scenarios are retrieved.
            models : list of str, optional
                Models for which to retrieve data. If not provided, all available models
                are retrieved.
            realizations : list of int, optional
                Realizations for which to retrieve data (indexed from 0). If not
                provided, all available realizations are retrieved.
            locations : list of str, optional
                Name of one or more locations for which to retrieve data. If provided,
                and the 'lons' and 'lats' keys are not provided, OpenStreetMap
                data (https://openstreetmap.org/copyright) is used to query
                corresponding longitude and latitudes, and data for the nearest grid
                point to each location are retrieved). If 'lons' and 'lats' are also
                provided, these are used to retrieve the data (the locations parameter
                is still used as a dimension coordinate in the output dataset). If not
                provided, the 'lon_range' and 'lat_range' keys are used instead.
            lons : list of float, optional
                Longitude(s) for which to retrieve data. If provided, both 'locations'
                and 'lats' should also be provided, and must be lists of the same
                length.
            lats : list of float, optional
                Latitude(s) for which to retrieve data. If provided, both 'locations'
                and 'lons' should also be provided, and must be lists of the same
                length.
            lon_range : tuple of two floats, optional
                Longitude range for which to retrieve data. Should comprise two values
                giving the minimum and maximum longitudes. Ignored if 'locations' is
                provided. If not provided, and 'locations' is also not provided, all
                longitudes are retrieved.
            lat_range : tuple of two floats, optional
                Latitude range for which to retrieve data. Should comprise two values
                giving the minimum and maximum latitudes. Ignored if 'locations' is
                provided. If not provided, and 'locations' is also not provided, all
                latitudes are retrieved.
    save_dir : str or pathlib.Path, optional
        Directory to which downloaded data are saved to and accessed from. If not
        provided, a directory within the OS cache directory is used.
    download : bool, optional
        For CESM2 LENS and ARISE data only; whether to download the data to the
        ``save_dir`` directory if not found locally (default is ``True``). If ``False``
        and the data are not found locally, a lazily opened dataset linked to the remote
        data is returned. For GLENS and ISIMIP data, the data must be downloaded if not
        found locally.
    force_remake : bool, optional
        Whether to force re-download and re-formatting of the data even if found
        locally (default is ``False``). Can only be used if ``download`` is ``True``.
    subset_check_interval : float, optional
        For ISIMIP data only; time interval in seconds between checks for server-side
        data subsetting completion (default is 10).
    max_subset_wait_time : float, optional
        For ISIMIP data only; maximum time to wait for server-side data subsetting to
        complete, in seconds, before timing out (default is 20). Server-side subsetting
        will continue to run after this function times out, and this function can be
        re-run to check if the subsetting has completed and retrieve the subsetted data.
    **kwargs
        Additional keyword arguments to pass to :func:`xarray.open_mfdataset` when
        opening downloaded data files.

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


def get_climate_data_file_names(
    data_source: Literal["lens2", "arise", "glens", "isimip"],
    frequency: Literal["daily", "monthly", "yearly"] = "monthly",
    subset: SubsetPartial | None = None,
) -> list[str]:
    """
    Retrieve file names of formatted climate data files.

    File names are as created by :meth:`~climepi.climdata.get_climate_data`.

    Parameters
    ----------
    data_source : str
        Data source. Currently supported sources are 'lens2' (for CESM2 LENS data),
        'arise' (CESM2 ARISE data), 'glens' (CESM1 GLENS data), and 'isimip' (ISIMIP
        data).
    frequency : str, optional
        Frequency of the data. Should be one of 'daily', 'monthly' or 'yearly' (default
        is 'monthly').
    subset: dict, optional
        Dictionary of data subsetting options. See the docstring of
        :meth:`~climepi.climdata.get_climate_data` for details.

    Returns
    -------
    list of str
        List of file names of formatted climate data files.
    """
    data_getter = _get_data_getter(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
    )
    file_names = data_getter.file_names
    return file_names


def _get_data_getter(
    data_source: Literal["lens2", "arise", "glens", "isimip"],
    *args: Any,
    subset_check_interval: float = 10,
    max_subset_wait_time: float = 20,
    **kwargs: Any,
) -> ClimateDataGetter:
    data_getter: ClimateDataGetter
    if data_source == "lens2":
        data_getter = LENS2DataGetter(*args, **kwargs)
    elif data_source == "arise":
        data_getter = ARISEDataGetter(*args, **kwargs)
    elif data_source == "glens":
        data_getter = GLENSDataGetter(*args, **kwargs)
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
