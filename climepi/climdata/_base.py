"""Module defining get_climate_data and get_climate_data_file_names functions."""

import os
from typing import Any, Literal

import xarray as xr

from climepi.climdata._cesm import ARISEDataGetter, GLENSDataGetter, LENS2DataGetter
from climepi.climdata._data_getter_class import (
    ClimateDataGetter,
    SubsetPartial,
)
from climepi.climdata._era5 import ERA5DataGetter
from climepi.climdata._isimip import (
    ISIMIP3aDataGetter,
    ISIMIP4aDataGetter,
    ISIMIP4bDataGetter,
    ISIMIPDataGetter,
)

ClimateDataSource = Literal[
    "lens2", "arise", "glens", "isimip", "isimip3a", "isimip4a", "isimip4b", "era5"
]


def get_climate_data(
    data_source: ClimateDataSource,
    frequency: Literal["daily", "monthly", "yearly"] = "monthly",
    subset: SubsetPartial | None = None,
    save_dir: str | os.PathLike | None = None,
    download: bool = True,
    force_remake: bool = False,
    subset_check_interval: float = 10,
    max_subset_wait_time: float = 20,
    *,
    subset_method: Literal["remote", "local"] = "remote",
    **kwargs: Any,
) -> xr.Dataset:
    """
    Retrieve climate projection and reanalysis data.

    Remotely stored climate data are formatted and downloaded (if
    ``download=True``). If data have been downloaded previously (and
    ``force_remake=False``), rerunning this function will simply open the local data
    files.

    Currently available data sources are:

    - LENS2 (``data_source='lens2'``): Community Earth System Model (CESM) version 2
      Large Ensemble (LENS2) data (https://www.cesm.ucar.edu/community-projects/lens2).
      Possible values for subsetting options (items in ``subset``) are (all ranges
      inclusive):

      - years: 1850–2100 (historical and future data combined for convenience).
      - scenarios: 'ssp370' (historical and SSP3-7.0).
      - models: 'cesm2'.
      - realizations: 0–99 (100 ensemble members).

      Data are available from AWS (https://registry.opendata.aws/ncar-cesm2-lens/).
      Terms of data use are available at https://www.ucar.edu/terms-of-use/data.
    - ARISE-SAI (``data_source='arise'``): CESM2 Assessing Responses and Impacts of
      Solar intervention on the Earth system with Stratospheric Aerosol Injection
      (ARISE-SAI) data (https://www.cesm.ucar.edu/community-projects/arise-sai).
      Possible values for subsetting options (all ranges inclusive):

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
      - scenarios: 'ssp126' (SSP1-2.6), 'ssp370' (SSP3-7.0), and 'ssp585'
        (SSP5-8.5) by default. 'ssp245' (SSP2-4.5) can also be requested for all
        ten models.
      - models: 'gfdl-esm4', 'ipsl-cm6a-lr', 'mpi-esm1-2-hr', 'mri-esm2-0',
        'ukesm1-0-ll', 'canesm5', 'cnrm-cm6-1', 'cnrm-esm2-1', 'ec-earth3', 'miroc6'.
        Details of models and bias adjustment are provided at
        https://www.isimip.org/gettingstarted/isimip3b-bias-adjustment/.
      - realizations: 0 (single ensemble member per model/scenario pair).

      Data are available from the ISIMIP repository (https://data.isimip.org/).
      Terms of data use are available at
      https://www.isimip.org/gettingstarted/terms-of-use/terms-use-publicly-available-isimip-data-after-embargo-period/.
    - ISIMIP3a (``data_source='isimip3a'``): ISIMIP Phase 3a GSWP3-W5E5 historical
      climate forcing data (https://protocol.isimip.org/). Possible values for
      subsetting options (all ranges inclusive):

      - years: 1901–2019.
      - scenarios: 'obsclim' (observed climate) by default. 'counterclim' (detrended
        counterfactual climate, used for impact attribution) can also be requested.
      - models: 'gswp3-w5e5'.
      - realizations: 0.

      Data are available from the ISIMIP repository on the same terms as ISIMIP3b
      data. Citation information is given in the protocol linked above.
    - ISIMIP4a (``data_source='isimip4a'``): ISIMIP Phase 4a GSWP3-ERA5 historical
      climate forcing data (https://protocol4.isimip.org/). Possible values for
      subsetting options (all ranges inclusive):

      - years: 1901–2025.
      - scenarios: 'obsclim' (observed climate).
      - models: 'gswp3-era5'.
      - realizations: 0.

      Data are available from the ISIMIP repository on the same terms as ISIMIP3b
      data. Citation information is given in the protocol linked above.
    - ISIMIP4b (``data_source='isimip4b'``): ISIMIP Phase 4b bias-adjusted CMIP7
      projection data (https://protocol4.isimip.org/). Possible values for
      subsetting options (all ranges inclusive):

      - years: 2022–2100 (future scenarios), 1850–2021 ('esm-hist').
      - scenarios: 'esm-scen7-vl' (Very Low) and 'esm-scen7-h' (High) by default.
        'esm-hist' (historical) can also be requested.
      - models: 'ec-earth3-esm-1-1', 'ukesm1-3-ll'.
      - realizations: 0.

      Data are available from the ISIMIP repository on the same terms as ISIMIP3b
      data. Citation information is given in the protocol linked above.
    - ERA5 (``data_source='era5'``): ECMWF ERA5 global reanalysis data
      (https://doi.org/10.24381/cds.adbb2d47). Daily values are means of hourly 2 m
      temperature and totals of hourly precipitation over UTC days, and monthly and
      yearly values are averages of the daily values. Possible values for subsetting
      options (all ranges inclusive):

      - years: 1941 to the latest complete year (added about four months after the
        year ends).
      - scenarios: 'historical'.
      - models: 'era5'.
      - realizations: 0.

      Data are available from Earthmover's public Icechunk archive on AWS
      (https://registry.opendata.aws/earthmover-era5/), which also gives citation
      information and terms of use.

    For each data source, global data are available. Server-side spatial subsetting can
    be performed for LENS2, ARISE, ISIMIP, and ERA5 data (see subsetting options below).
    GLENS downloads full data files and clears them after processing. Splitting a
    GLENS retrieval into multiple calls may reduce the disk space overhead.

    Parameters
    ----------
    data_source : str
        Data source to retrieve data from. Currently supported sources are 'lens2' (for
        LENS2 data), 'arise' (ARISE-SAI data), 'glens' (GLENS data), 'isimip'
        (ISIMIP3b data), 'isimip3a', 'isimip4a', 'isimip4b', and 'era5' (ERA5
        reanalysis).
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
                giving the western and eastern bounds. For example, (170, -170)
                selects a range crossing the antimeridian. Longitudes in a range
                that crosses the edge of the data's longitude grid are returned
                increasing from the western bound (e.g. -3 to 1 for (-3, 1), even
                where the data use longitudes from 0 to 360). Ignored if
                'locations' is provided. If omitted along with 'locations', all
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
        For LENS2, ARISE-SAI, and ERA5 data, controls whether to download the data to
        the ``save_dir`` directory if not found locally (default is ``True``). If
        ``False`` and the data are not found locally, the remote dataset is opened
        lazily.
        For GLENS and ISIMIP data, the data must be downloaded if not found locally.
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
    subset_method : {'remote', 'local'}, optional
        For ISIMIP sources, spatially subset before downloading ('remote', the
        default) or download the original global files and subset locally ('local').
        Remote subsetting usually transfers much less data but must wait for the ISIMIP
        queue. Local subsetting skips the queue and can use several GB of temporary
        disk space per model, scenario and time block. Both methods use the same
        processed cache files. Global requests use the original files.
    **kwargs
        Additional keyword arguments to pass to :func:`xarray.open_mfdataset` when
        opening downloaded data files.

    Returns
    -------
    xarray.Dataset
        Formatted climate dataset.
    """
    data_getter = _get_data_getter(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
        save_dir=save_dir,
        subset_check_interval=subset_check_interval,
        max_subset_wait_time=max_subset_wait_time,
        subset_method=subset_method,
    )
    ds_clim = data_getter.get_data(
        download=download, force_remake=force_remake, **kwargs
    )
    return ds_clim


def get_climate_data_file_names(
    data_source: ClimateDataSource,
    frequency: Literal["daily", "monthly", "yearly"] = "monthly",
    subset: SubsetPartial | None = None,
) -> list[str]:
    """
    Retrieve file names of formatted climate data files.

    File names are as created by :meth:`~climepi.climdata.get_climate_data`.

    Parameters
    ----------
    data_source : str
        Data source. See :meth:`~climepi.climdata.get_climate_data` for supported
        sources.
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


def _data_getter_classes() -> dict[str, type[ClimateDataGetter]]:
    # Map each supported data source to its data getter class, in the order sources
    # are presented in the app.
    return {
        "lens2": LENS2DataGetter,
        "arise": ARISEDataGetter,
        "glens": GLENSDataGetter,
        "isimip": ISIMIPDataGetter,
        "isimip3a": ISIMIP3aDataGetter,
        "isimip4a": ISIMIP4aDataGetter,
        "isimip4b": ISIMIP4bDataGetter,
        "era5": ERA5DataGetter,
    }


def _get_data_getter(
    data_source: ClimateDataSource,
    *args: Any,
    subset_check_interval: float = 10,
    max_subset_wait_time: float = 20,
    subset_method: Literal["remote", "local"] = "remote",
    **kwargs: Any,
) -> ClimateDataGetter:
    data_getter_classes = _data_getter_classes()
    if data_source not in data_getter_classes:
        raise ValueError(f"Data source '{data_source}' not supported.")
    if data_source.startswith("isimip"):
        kwargs.update(
            subset_check_interval=subset_check_interval,
            max_subset_wait_time=max_subset_wait_time,
            subset_method=subset_method,
        )
    elif subset_method != "remote":
        raise ValueError("subset_method is only available for ISIMIP data sources.")
    return data_getter_classes[data_source](*args, **kwargs)
