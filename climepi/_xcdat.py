"""
Module defining methods/classes ported from xcdat package.

The xesmf package is mocked if it cannot be imported (simplifies use on Windows, since
xesmf/esmpy can be difficult to install on Windows but is not required for climepi).
"""

import importlib
import logging
import sys
import types
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

try:
    importlib.import_module("xesmf")
except (ImportError, KeyError):
    xesmf: Any = types.ModuleType("xesmf")
    xesmf.Regridder = None
    sys.modules["xesmf"] = xesmf
    logging.warning(
        "`xesmf` package could not be imported; using mocked version. This does not "
        "affect the functionality of `climepi` (`xesmf` is an upstream dependency of "
        "the `xcdat` package, which is used for regridding operations not required by "
        "`climepi`)."
    )

from xcdat import (  # noqa
    center_times,
    BoundsAccessor,
    TemporalAccessor,
    swap_lon_axis,
)


def _infer_freq(time_coords: xr.DataArray) -> Literal["year", "month", "day", "hour"]:
    # Infer the time frequency from the median time step. Ported from the private
    # `xcdat.temporal._infer_freq` function (xcdat v0.11.3) rather than imported, since
    # the older xcdat versions installed on Windows (where newer versions cannot be
    # installed due to their `xesmf` dependency) use the minimum rather than median time
    # step, and raise warnings with recent pandas versions.
    time_deltas = np.diff(time_coords.values).astype("timedelta64[ns]")
    median_delta = pd.to_timedelta(np.median(time_deltas))
    if median_delta < pd.Timedelta(days=1):
        return "hour"
    if median_delta < pd.Timedelta(days=21):
        return "day"
    if median_delta < pd.Timedelta(days=300):
        return "month"
    return "year"
