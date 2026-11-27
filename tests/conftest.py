"""Shared pytest fixtures."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from geopy import Location


@pytest.fixture(autouse=True)
def mock_sel_geo_geocoding() -> Iterator[None]:
    """Use fixed coordinates when tests select grid cells by location name."""
    coordinates = {
        "cape town": (-33.9, 18.4),
        "gabba": (-27.5, 153.0),
        "lords": (51.5, -0.2),
        "los angeles": (34.1, -118.2),
        "miami": (25.8, -80.2),
        "scg": (-33.9, 151.2),
        "tokyo": (35.7, 139.7),
    }

    def get_location(name: str, **_kwargs: object) -> Location:
        return Location(name, coordinates[name.lower()], {})

    with patch("climepi._core.geocode", new=get_location):
        yield
