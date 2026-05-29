"""Unit tests for the lazy attribute loading in the app subpackage __init__."""

import pytest

import climepi.app._app_construction as app_construction
from climepi import app


def test_lazy_getattr():
    """Test that lazily-loaded attributes resolve to the _app_construction module."""
    assert app.run_app is app_construction.run_app
    assert app.get_logger is app_construction.get_logger


def test_lazy_getattr_invalid():
    """Test that accessing an unknown attribute raises an AttributeError."""
    missing_attr = "does_not_exist"
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(app, missing_attr)


def test_lazy_dir():
    """Test that dir() advertises the lazily-loaded attributes."""
    assert {"run_app", "get_logger"} <= set(dir(app))
