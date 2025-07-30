"""
Unit tests for model classes in the epimod subpackage.

The EpiModel and SuitabilityModel classes are tested.
"""

import holoviews as hv
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt

import climepi  # noqa
from climepi import epimod
from climepi.testing.fixtures import generate_dataset


class TestEpiModel:
    """Class defining tests for the EpiModel base class."""

    def test_init(self):
        """Test that the EpiModel class can be instantiated."""
        model = epimod.EpiModel()
        assert model is not None

    def test_run(self):
        """
        Test that the run method raises a NotImplementedError.

        (This method should be overridden in subclasses.)
        """
        model = epimod.EpiModel()
        ds = generate_dataset()
        with pytest.raises(NotImplementedError):
            model.run(ds)


class TestSuitabilityModel:
    """Class defining tests for the SuitabilityModel class."""

    def test_init_range(self):
        """Test that the class can be instantiated with a temperature range."""
        model = epimod.SuitabilityModel(temperature_range=[0, 1])
        assert isinstance(model, epimod.EpiModel)
        assert model.temperature_range == [0, 1]
        assert model.suitability_table is None
        assert model._suitability_var_name == "suitability"
        assert model._suitability_var_long_name == "Suitability"

    def test_init_table(self):
        """Test that the class can be instantiated with a suitability table."""
        # Example 1: suitability table without long_name attribute
        suitability_table1 = xr.Dataset(
            {
                "hello": (("there", "general"), [[0.5, 0.6], [0.7, 0.8]]),
            }
        )
        model1 = epimod.SuitabilityModel(suitability_table=suitability_table1)
        xrt.assert_equal(model1.suitability_table, suitability_table1)
        assert model1.temperature_range is None
        assert model1._suitability_var_name == "hello"
        assert model1._suitability_var_long_name == "Hello"
        assert model1.suitability_table["hello"].attrs == {"long_name": "Hello"}
        assert suitability_table1["hello"].attrs == {}
        # Example 2: suitability table with long_name attribute
        suitability_table2 = suitability_table1.copy()
        suitability_table2["hello"].attrs["long_name"] = "Kenobi"
        model2 = epimod.SuitabilityModel(suitability_table=suitability_table2)
        xrt.assert_equal(model2.suitability_table, suitability_table2)
        assert model2.temperature_range is None
        assert model2._suitability_var_name == "hello"
        assert model2._suitability_var_long_name == "Kenobi"
        assert model2.suitability_table["hello"].attrs == {"long_name": "Kenobi"}
        # Example 3: suitability table with multiple independent variables
        suitability_table3 = suitability_table1.copy()
        suitability_table3["you"] = suitability_table3["hello"]
        with pytest.raises(ValueError):
            epimod.SuitabilityModel(suitability_table=suitability_table3)
        # Example 4: try passing both temperature range and suitability table
        with pytest.raises(ValueError):
            epimod.SuitabilityModel(
                temperature_range=[0, 1], suitability_table=suitability_table1
            )

    def test_run_range(self):
        """Test the run method with a temperature range."""
        model = epimod.SuitabilityModel(temperature_range=[0, 0.5])
        ds_clim = generate_dataset(data_var="temperature", frequency="monthly")
        ds_clim.attrs = {"did": "you"}
        ds_suitability = model.run(ds_clim)
        npt.assert_equal(
            ds_suitability["suitability"].values, ds_clim["temperature"].values < 0.5
        )
        assert ds_suitability.attrs == ds_clim.attrs
        assert ds_suitability["suitability"].attrs == {"long_name": "Suitability"}
        xrt.assert_equal(
            ds_suitability[["lon_bnds", "lat_bnds", "time_bnds"]],
            ds_clim[["lon_bnds", "lat_bnds", "time_bnds"]],
        )
        # Check that running return_yearly_portion_suitable=True gives the same result
        # as calculating months suitable from the suitability dataset.
        ds_months_suitable = model.run(ds_clim, return_yearly_portion_suitable=True)
        xrt.assert_equal(
            ds_months_suitable,
            ds_suitability.climepi.yearly_portion_suitable(),
        )

    def test_run_temp_table(self):
        """Test the run method with a temperature-dependent suitability table."""
        suitability_table = xr.Dataset(
            {"hello": ("temperature", [0, 0.5, 1])},
            coords={"temperature": [0, 1, 2]},
        )
        suitability_table["hello"].attrs = {
            "units": "there",
        }
        model = epimod.SuitabilityModel(suitability_table=suitability_table)
        ds_clim = xr.Dataset({"temperature": ("kenobi", [-0.5, 1, 0.51, 1.75, 2.5])})
        ds_suitability = model.run(ds_clim)
        suitability_values_expected = [0, 0.5, 0.5, 1, 1]  # nearest neighbour interp
        npt.assert_equal(
            ds_suitability["hello"].values,
            suitability_values_expected,
        )
        assert ds_suitability["hello"].attrs == {
            "long_name": "Hello",
            "units": "there",
        }

    def test_run_temp_precip_table(self):
        """Test the run method with a temp/precip-dependent suitability table."""
        suitability_table = xr.Dataset(
            {
                "suitability": (
                    ("temperature", "precipitation"),
                    [[0, 0.5], [0.75, 1], [0.25, 0.69]],
                ),
            },
            coords={
                "temperature": [0, 1, 2],
                "precipitation": [0, 1],
            },
        )
        suitability_table["suitability"].attrs = {
            "long_name": "hello",
            "units": "kenobi",
        }
        model = epimod.SuitabilityModel(suitability_table=suitability_table)
        ds_clim = xr.Dataset(
            {
                "temperature": ("general", [-0.3, 0, 1.5, 0.7, 2, 4]),
                "precipitation": ("general", [-0.5, 1, 0.25, 0.75, 0.3, 0.8]),
            }
        )
        ds_suitability = model.run(ds_clim)
        suitability_values_expected = [0, 0.5, 0.25, 1, 0.25, 0.69]  # nearest neighbor
        npt.assert_equal(
            ds_suitability["suitability"].values,
            suitability_values_expected,
        )
        assert ds_suitability["suitability"].attrs == {
            "long_name": "hello",
            "units": "kenobi",
        }
        # Check that running with a suitability table with non-equally spaced
        # temperature or precipitation values raises an error.
        suitability_table1 = suitability_table.assign_coords(temperature=[0, 1, 1.5])
        model1 = epimod.SuitabilityModel(suitability_table=suitability_table1)
        with pytest.raises(ValueError):
            model1.run(ds_clim)

    def test_plot_suitability_range(self):
        """Test the plot_suitability method with a temperature range."""
        model = epimod.SuitabilityModel(temperature_range=[0, 1])
        result = model.plot_suitability(color="red")
        assert isinstance(result, hv.Curve)
        assert result.kdims[0].pprint_label == "Temperature (°C)"
        assert result.vdims[0].pprint_label == "Suitability"
        npt.assert_allclose(
            result.data.suitability.values,
            (result.data.index.values >= 0) & (result.data.index.values <= 1),
        )

    def test_plot_suitability_temp_table(self):
        """Test plot_suitability with a temp-dependent suitability table."""
        suitability_table = xr.Dataset(
            {"suitability": ("temperature", [False, True, False])},
            coords={"temperature": [0, 1, 2]},
        )
        suitability_table["suitability"].attrs = {
            "long_name": "hello there",
            "units": "general kenobi",
        }
        suitability_table["temperature"].attrs = {
            "long_name": "Temperature",
            "units": "°C",
        }
        model = epimod.SuitabilityModel(suitability_table=suitability_table)
        result = model.plot_suitability(color="blue")
        assert isinstance(result, hv.Curve)
        assert result.kdims[0].pprint_label == "Temperature (°C)"
        assert result.vdims[0].pprint_label == "hello there (general kenobi)"

    def test_plot_suitability_temp_precip_table(self):
        """Test plot_suitability with a temp/precip-dependent suitability table."""
        suitability_table = xr.Dataset(
            {
                "suitability": (
                    ("temperature", "precipitation"),
                    [[0, 0.5], [0.75, 1], [0.25, 0.69]],
                ),
            },
            coords={
                "temperature": [0, 1, 2],
                "precipitation": [0, 1],
            },
        )
        suitability_table["temperature"].attrs = {
            "long_name": "Temperature",
            "units": "°C",
        }
        suitability_table["precipitation"].attrs = {
            "long_name": "Precipitation",
            "units": "mm/day",
        }
        model = epimod.SuitabilityModel(suitability_table=suitability_table)
        result = model.plot_suitability()
        assert isinstance(result, hv.QuadMesh)
        assert result.kdims[0].pprint_label == "Temperature (°C)"
        assert result.kdims[1].pprint_label == "Precipitation (mm/day)"
        assert result.vdims[0].pprint_label == "Suitability"

    def test_get_max_suitability_range(self):
        """Test the get_max_suitability method with a temperature range."""
        model = epimod.SuitabilityModel(temperature_range=[0.5, 19])
        result = model.get_max_suitability()
        npt.assert_equal(result, 1)

    def test_get_max_suitability_table(self):
        """Test the get_max_suitability method with a suitability table."""
        suitability_table = xr.Dataset(
            {"hello": ("there", [0, 3.5, 1])},
            coords={"there": [0, 1, 2]},
        )
        model = epimod.SuitabilityModel(suitability_table=suitability_table)
        result = model.get_max_suitability()
        npt.assert_equal(result, 3.5)
