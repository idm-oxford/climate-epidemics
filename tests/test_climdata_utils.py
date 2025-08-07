"""Tests for the _utils module of the climdata subpackage."""

from unittest.mock import patch

from climepi import climdata


def test_get_data_version():
    """
    Test the _get_data_version method.

    Should default to "main" for development versions.
    """
    with patch.object(
        climdata._utils, "get_versions", return_value={"version": "4.2.0"}
    ):
        assert climdata._utils._get_data_version() == "v4.2.0"
    with patch.object(
        climdata._utils, "get_versions", return_value={"version": "4.2.0+10.8dl8dh9"}
    ):
        assert climdata._utils._get_data_version() == "main"
