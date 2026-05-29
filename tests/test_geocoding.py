"""Unit tests for the _geocoding module of the climepi package."""

from unittest.mock import patch

import geopy
import pytest

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


def test_geocode_exactly_one_false():
    """Test that requesting multiple results raises a ValueError."""
    with pytest.raises(ValueError, match="exactly_one=False"):
        geocoding.geocode("anywhere", exactly_one=False)


@patch.object(geocoding, "_geocode")
def test_geocode_unresolved_query(mock_geocode):
    """Test that an unresolvable query raises a ValueError."""
    mock_geocode.return_value = None
    with pytest.raises(ValueError, match="Could not geocode query 'nowhere-real'"):
        geocoding.geocode("nowhere-real")
