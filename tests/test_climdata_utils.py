"""Tests for the _utils module of the climdata subpackage."""

from unittest.mock import patch

import numpy as np
import xarray as xr

from climepi import climdata


def test_get_data_version():
    """
    Test the _get_data_version method.

    Should default to "main" for development versions.
    """
    with patch.object(
        climdata._utils, "get_version_dict", return_value={"version": "4.2.0"}
    ):
        assert climdata._utils._get_data_version() == "v4.2.0"
    with patch.object(
        climdata._utils,
        "get_version_dict",
        return_value={"version": "4.2.0+10.8dl8dh9"},
    ):
        assert climdata._utils._get_data_version() == "main"


def test_drop_string_width_encodings(tmp_path):
    """Test that stale string widths are removed so saving keeps whole names."""
    ds = xr.Dataset(
        {"temperature": ("model", [1.0, 2.0])},
        coords={"model": ["gfdl-esm4", "canesm5"]},
    )
    ds["model"].encoding["dtype"] = np.dtype("<U7")
    ds["temperature"].encoding["dtype"] = np.dtype("float32")
    climdata._utils._drop_string_width_encodings(ds)
    assert "dtype" not in ds["model"].encoding
    assert ds["temperature"].encoding["dtype"] == np.dtype("float32")
    ds.to_netcdf(tmp_path / "out.nc")
    with xr.open_dataset(tmp_path / "out.nc") as reopened:
        assert reopened.model.values.tolist() == ["gfdl-esm4", "canesm5"]
