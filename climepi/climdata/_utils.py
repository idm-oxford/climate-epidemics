"""Utility functions for the climdata subpackage."""

import pooch
import xarray as xr

from climepi._version import get_version_dict

FALLBACK_BRANCH = "main"


def _get_data_version() -> str:
    version = pooch.check_version(
        get_version_dict()["version"], fallback=FALLBACK_BRANCH
    )
    if version != FALLBACK_BRANCH:
        version = "v" + version
    return version


def _standardize_names_and_units(
    ds: xr.Dataset,
    temperature_var: str,
    precipitation_var: str,
    precipitation_factor: float,
) -> xr.Dataset:
    # Replace a data source's temperature (in Kelvin) and precipitation variables with
    # 'temperature' (in °C) and 'precipitation' (in mm/day) variables, where
    # precipitation_factor converts the source precipitation units to mm/day.
    return ds.assign(
        temperature=(ds[temperature_var] - 273.15).assign_attrs(
            long_name="Temperature", units="°C"
        ),
        precipitation=(ds[precipitation_var] * precipitation_factor).assign_attrs(
            long_name="Precipitation", units="mm/day"
        ),
    ).drop_vars([temperature_var, precipitation_var])
