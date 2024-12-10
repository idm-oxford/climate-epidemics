"""
Module defining methods/classes ported from xcdat package.

The xesmf package is mocked if it cannot be imported (simplifies use on Windows, since
xesmf/esmpy can be difficult to install on Windows but is not required for climepi).
"""

import logging
import sys
import types

try:
    import xesmf
except ImportError:
    xesmf = types.ModuleType("xesmf")
    xesmf.Regridder = None
    sys.modules["xesmf"] = xesmf
    logging.warning(
        "xesmf package could not be imported; using mocked version. This does not "
        "affect the functionality of `climepi`."
    )

from xcdat import (  # noqa
    center_times,
    BoundsAccessor,
    TemporalAccessor,
    swap_lon_axis,
)
from xcdat.temporal import _infer_freq  # noqa
