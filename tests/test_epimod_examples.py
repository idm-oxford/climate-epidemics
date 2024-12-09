"""Unit tests for the _examples.py module of the epimod subpackage."""

from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt

from climepi import epimod


class TestGetExampleModel:
    """Test the get_example_model function."""

    @patch.dict(
        epimod._examples.EXAMPLES,
        {"test": {"temperature_range": [10, 40]}},
    )
    def test_get_example_model_temp_range(self):
        """
        Test with a temperature range supplied.

        Checks an EpiModel object with the supplied temperature range is returned.
        """
        epi_model = epimod.get_example_model("test")
        assert epi_model.temperature_range == [10, 40]

    @patch.dict(
        epimod._examples.EXAMPLES,
        {"test": {"temperature_range": [10, 40], "precipitation_range": [0, 100]}},
    )
    def test_get_example_model_temp_precip_range(self):
        """
        Test with temperature and precipitation ranges supplied.

        Checks an EpiModel object with a compatible suitability table is returned.
        """
        epi_model = epimod.get_example_model("test")
        temperature_vals = np.random.uniform(-5, 55, 1000)
        precipitation_vals = np.random.uniform(-50, 150, 1000)
        ds_clim = xr.Dataset(
            {
                "temperature": ("legspinner", temperature_vals),
                "precipitation": ("legspinner", precipitation_vals),
            }
        )
        suitability_vals = epi_model.run(ds_clim)["suitability"].values
        suitability_vals_expected = (
            (temperature_vals >= 10)
            & (temperature_vals <= 40)
            & (precipitation_vals >= 0)
            & (precipitation_vals <= 100)
        ).astype(int)
        npt.assert_equal(suitability_vals, suitability_vals_expected)

    @patch.dict(
        epimod._examples.EXAMPLES,
        {"test": {"temperature_vals": [10, 20, 30], "suitability_vals": [0, 1, 0]}},
    )
    def test_get_example_model_vals(self):
        """
        Test with temperature and suitability values supplied.

        Checks an EpiModel object with the supplied values is returned.
        """
        epi_model = epimod.get_example_model("test")
        npt.assert_allclose(
            epi_model.suitability_table["temperature"].values, [10, 20, 30]
        )
        npt.assert_allclose(
            epi_model.suitability_table["suitability"].values, [0, 1, 0]
        )

    @patch.dict(
        epimod._examples.EXAMPLES,
        {"test": {"suitability_table_path": "not/a/real/path.nc"}},
    )
    def test_get_example_model_table(self):
        """
        Test with a suitability table path supplied.

        Checks an EpiModel object with the suitability table loaded from the path is
        returned.
        """
        suitability_table = xr.Dataset(
            {"suitability": ("temperature", [0, 1, 0])},
            coords={"temperature": [10, 20, 30]},
        )
        suitability_table["suitability"].attrs = {"long_name": "Suitability"}
        suitability_table["temperature"].attrs = {
            "long_name": "Temperature",
            "units": "°C",
        }
        with patch("xarray.open_dataset", return_value=suitability_table):
            epi_model = epimod.get_example_model("test")
        xrt.assert_identical(epi_model.suitability_table, suitability_table)

    @patch.dict(
        epimod._examples.EXAMPLES,
        {"test": {"precipitation_range": [10, 40]}},
    )
    def test_get_example_model_formatting_error(self):
        """
        Test with a temperature range supplied.

        Checks an EpiModel object with the supplied temperature range is returned.
        """
        with pytest.raises(ValueError, match="does not have a recognised format"):
            epimod.get_example_model("test")


@patch.dict(epimod._examples.EXAMPLES, {"googly": "back of the hand"})
def test_get_example_details():
    """Test that _get_example_details returns the details of an example model."""
    result = epimod._examples._get_example_details("googly")
    assert result == "back of the hand"
    with pytest.raises(ValueError, match="Available examples are"):
        epimod._examples._get_example_details("flipper")
