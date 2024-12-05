"""Unit tests for the _geocoding module of the climepi package."""

import types
from unittest.mock import patch

import geopy

import climepi._geocoding as geocoding


def test_initialize_geocode():
    """Test the _initialize_geocode method."""
    geocoding._geocode = None
    geocoding._initialize_geocode()
    assert isinstance(geocoding._geocode, geopy.extra.rate_limiter.RateLimiter)


@patch.object(geocoding, "_geocode")
def test_geocode(mock_geocode):
    """
    Test the geocode method.

    Checks that caching works as expected.
    """
    mock_geocode.return_value = "not out"
    assert geocoding.geocode("howzat") == "not out"
    assert geocoding.geocode("howzat") == "not out"
    assert mock_geocode.call_count == 1
