"""Unit tests for the _xcdat module of the climepi package."""

import logging
import sys
import types
from unittest.mock import patch

LOGGER = logging.getLogger("climepi._xcdat")
LOGGER.propagate = True


def test_xesmf_import_error_handling(caplog):
    """Test that the _xcdat.py module correctly handles an ImportError for `xesmf`."""
    sys.modules.pop("xesmf", None)

    original_import = __builtins__["__import__"]

    def mock_import(name, *args, **kwargs):
        if name == "xesmf" and "xesmf" not in sys.modules:
            raise ImportError("Simulated ImportError for xesmf")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with caplog.at_level(logging.WARNING, logger="climepi._xcdat"):
            import climepi._xcdat

    assert isinstance(sys.modules["xesmf"], types.ModuleType)
    assert sys.modules["xesmf"].Regridder is None
    assert hasattr(climepi._xcdat, "BoundsAccessor")
    assert hasattr(climepi._xcdat, "TemporalAccessor")
    assert "`xesmf` package could not be imported; using mocked version." in caplog.text
