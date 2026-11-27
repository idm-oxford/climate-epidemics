"""Utility functions for the climdata subpackage."""

import numpy as np
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


def _drop_string_width_encodings(ds: xr.Dataset) -> None:
    # Remove fixed string widths from the encoding of an opened dataset. Datasets
    # opened from several files take the width from the first file, so saving them
    # would truncate longer strings from other files (e.g. "gfdl-esm4" to "gfdl-es").
    for variable in ds.variables.values():
        encoded = variable.encoding.get("dtype")
        if encoded is not None and np.dtype(encoded).kind == "U":
            del variable.encoding["dtype"]
