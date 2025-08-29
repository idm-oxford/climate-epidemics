"""
Module defining methods/classes ported from xcdat package.

The xesmf package is mocked if it cannot be imported (simplifies use on Windows, since
xesmf/esmpy can be difficult to install on Windows but is not required for climepi).
"""

import importlib
import logging
import sys
import types
from typing import Any

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
from xcdat.temporal import _infer_freq  # noqa
