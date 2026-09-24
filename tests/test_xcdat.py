"""Unit tests for the _xcdat module of the climepi package."""

import importlib
import logging
import sys
import types
from unittest.mock import patch

import pytest
import xarray as xr

import climepi._xcdat
from climepi._xcdat import _infer_freq


def test_xesmf_import_error_handling(caplog):
    """Test that the _xcdat.py module correctly handles an ImportError for `xesmf`."""
    sys.modules.pop("xesmf", None)

    def mock_importlib_import(name, *args, **kwargs):
        if name == "xesmf":
            raise ImportError("Simulated ImportError for xesmf")
        raise ValueError(
            f"Unexpected import: {name}. Attempted imports of modules "
            "other than `xesmf` through importlib.import while being mocked may cause "
            "unexpected behavior."
        )

    with patch.object(importlib, "import_module", mock_importlib_import):
        with caplog.at_level(logging.WARNING):
            importlib.reload(climepi._xcdat)

    assert isinstance(sys.modules["xesmf"], types.ModuleType)
    assert sys.modules["xesmf"].Regridder is None
    assert "`xesmf` package could not be imported; using mocked version." in caplog.text

    importlib.reload(climepi._xcdat)


@pytest.mark.parametrize("use_cftime", [False, True])
@pytest.mark.parametrize(
    "freq,expected",
    [("h", "hour"), ("D", "day"), ("MS", "month"), ("YS", "year")],
)
def test_infer_freq(use_cftime, freq, expected):
    """Unit test for the _infer_freq function."""
    time = xr.DataArray(
        xr.date_range(start="2000", periods=12, freq=freq, use_cftime=use_cftime),
        dims="time",
    )
    assert _infer_freq(time) == expected


def test_infer_freq_irregular():
    """Test that _infer_freq uses the median (not minimum) time step."""
    time_values = xr.date_range(start="2000", periods=12, freq="MS").union(
        xr.date_range(start="2000-01-06", periods=1)
    )
    time = xr.DataArray(time_values, dims="time")
    assert _infer_freq(time) == "month"
