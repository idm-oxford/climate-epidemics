"""Unit tests for the _xcdat module of the climepi package."""

import importlib
import logging
import sys
import types
from unittest.mock import patch

import climepi._xcdat


def test_xesmf_import_error_handling(caplog):
    """Test that the _xcdat.py module correctly handles an ImportError for `xesmf`."""

    original_xesmf = sys.modules.pop("xesmf", None)
    original_import_module = importlib.import_module

    def mock_importlib_import(name, *args, **kwargs):
        if name == "xesmf":
            raise ImportError("Simulated ImportError for xesmf")
        return original_import_module(name, *args, **kwargs)

    with patch.object(importlib, "import_module", mock_importlib_import):
        with caplog.at_level(logging.WARNING):
            importlib.reload(climepi._xcdat)

    assert isinstance(sys.modules["xesmf"], types.ModuleType)
    assert sys.modules["xesmf"].Regridder is None
    assert hasattr(climepi._xcdat, "BoundsAccessor")
    assert hasattr(climepi._xcdat, "TemporalAccessor")
    assert "`xesmf` package could not be imported; using mocked version." in caplog.text

    if original_xesmf is not None:
        sys.modules["xesmf"] = original_xesmf
        importlib.reload(climepi._xcdat)
