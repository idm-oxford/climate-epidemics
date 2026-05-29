"""Unit tests for the lazy attribute loading in the epimod subpackage __init__."""

import pytest

import climepi.epimod._model_fitting as model_fitting
from climepi import epimod

_LAZY_ATTRS = {
    "ParameterizedSuitabilityModel",
    "fit_temperature_response",
    "get_posterior_temperature_response",
    "plot_fitted_temperature_response",
}


def test_lazy_getattr():
    """Test that lazily-loaded attributes resolve to the _model_fitting module."""
    for name in _LAZY_ATTRS:
        assert getattr(epimod, name) is getattr(model_fitting, name)


def test_lazy_getattr_invalid():
    """Test that accessing an unknown attribute raises an AttributeError."""
    missing_attr = "does_not_exist"
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(epimod, missing_attr)


def test_lazy_dir():
    """Test that dir() advertises the lazily-loaded attributes."""
    assert _LAZY_ATTRS <= set(dir(epimod))
