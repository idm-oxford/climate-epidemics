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
        model = epimod.SuitabilityModel(temperature_range=(0, 1))
        assert isinstance(model, epimod.EpiModel)
        assert model.temperature_range == (0, 1)
        assert model.suitability_table is None
        assert model._suitability_var_name == "suitability"
        assert model._suitability_var_long_name == "Suitability"

    def test_init_table(self):
        """Test that the class can be instantiated with a suitability table."""
        # Inferred long_name from variable name
        suitability_table1 = xr.Dataset(
            {
                "hello_there": (("general", "kenobi"), [[0.5, 0.6], [0.7, 0.8]]),
            }
        )
        model1 = epimod.SuitabilityModel(suitability_table=suitability_table1)
        xrt.assert_equal(model1.suitability_table, suitability_table1)
        assert model1.temperature_range is None
        assert model1._suitability_var_name == "hello_there"
        assert model1._suitability_var_long_name == "Hello there"
        assert model1.suitability_table["hello_there"].attrs == {
            "long_name": "Hello there"
        }
        assert suitability_table1["hello_there"].attrs == {}
        # Suitability table with long_name attribute
        suitability_table2 = suitability_table1.copy()
        suitability_table2["hello_there"].attrs["long_name"] = "Kenobi"
        model2 = epimod.SuitabilityModel(suitability_table=suitability_table2)
        xrt.assert_equal(model2.suitability_table, suitability_table2)
        assert model2.temperature_range is None
        assert model2._suitability_var_name == "hello_there"
        assert model2._suitability_var_long_name == "Kenobi"
        assert model2.suitability_table["hello_there"].attrs == {"long_name": "Kenobi"}
        # Pass suitability_var_long_name argument
        model3 = epimod.SuitabilityModel(
            suitability_table=suitability_table2,
            suitability_var_long_name="Bold one",
        )
        assert model3._suitability_var_long_name == "Bold one"
        assert model3.suitability_table["hello_there"].attrs == {
            "long_name": "Bold one"
        }
        # Error passing suitability table with multiple independent variables
        suitability_table3 = suitability_table1.copy()
        suitability_table3["you"] = suitability_table3["hello_there"]
        with pytest.raises(ValueError, match="single data variable"):
            epimod.SuitabilityModel(suitability_table=suitability_table3)
        # Error passing both temperature range and suitability table
        with pytest.raises(
            ValueError, match="temperature_range argument should not be provided"
        ):
            epimod.SuitabilityModel(
                temperature_range=(0, 1),
                suitability_table=suitability_table1,
            )
        # Error passing both suitability table and suitability_var_name
        with pytest.raises(
            ValueError, match="suitability_var_name argument should not be provided"
        ):
            epimod.SuitabilityModel(
                suitability_table=suitability_table1,
                suitability_var_name="there",
            )

    def test_run_range(self):
        """Test the run method with a temperature range."""
        model = epimod.SuitabilityModel(temperature_range=(0, 0.5))
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
            {"hello": (("there", "temperature"), [[0, 0.5, 1], [1, 1, 1]])},
            coords={"temperature": [0, 1, 2], "there": [0, 1]},
        )
        suitability_table["hello"].attrs = {
            "units": "kenobi",
        }
        model = epimod.SuitabilityModel(suitability_table=suitability_table)
        ds_clim = xr.Dataset({"temperature": ("lat", [-0.5, 1, 0.51, 1.75, 2.5])})
        ds_suitability = model.run(ds_clim)
        suitability_values_expected = [  # nearest neighbor interpolation
            [0, 0.5, 0.5, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        npt.assert_equal(
            ds_suitability["hello"].transpose("there", "lat").values,
            suitability_values_expected,
        )
        assert ds_suitability["hello"].attrs == {
            "long_name": "Hello",
            "units": "kenobi",
        }
        # Check that running with a suitability table with non-equally spaced
        # temperature or precipitation values raises an error.
        suitability_table1 = suitability_table.assign_coords(temperature=[0, 1, 1.5])
        model1 = epimod.SuitabilityModel(suitability_table=suitability_table1)
        with pytest.raises(ValueError):
            model1.run(ds_clim)

    def test_run_temp_precip_table(self):
        """Test the run method with a temp/precip-dependent suitability table."""
        suitability_table = xr.Dataset(
            {
                "suitability": (
                    ("general", "temperature", "precipitation"),
                    [[[0, 0.5], [0.75, 1], [0.25, 0.69]], [[0, 0], [0, 0], [0, 0]]],
                ),
            },
            coords={
                "temperature": [0, 1, 2],
                "precipitation": [0, 1],
                "general": ["kenobi", "grievous"],
            },
        )
        suitability_table["suitability"].attrs = {
            "long_name": "hello",
            "units": "there",
        }
        model = epimod.SuitabilityModel(suitability_table=suitability_table)
        ds_clim = xr.Dataset(
            {
                "temperature": ("lat", [-0.3, 0, 1.5, 0.7, 2, 4]),
                "precipitation": ("lat", [-0.5, 1, 0.25, 0.75, 0.3, 0.8]),
            }
        )
        ds_suitability = model.run(ds_clim)
        suitability_values_expected = [  # nearest neighbor interpolation
            [0, 0.5, 0.25, 1, 0.25, 0.69],
            [0, 0, 0, 0, 0, 0],
        ]
        npt.assert_equal(
            ds_suitability["suitability"].transpose("general", "lat").values,
            suitability_values_expected,
        )
        assert ds_suitability["suitability"].attrs == {
            "long_name": "hello",
            "units": "there",
        }
        # Check that running with a suitability table with non-equally spaced
        # temperature or precipitation values raises an error.
        suitability_table1 = suitability_table.assign_coords(temperature=[0, 1, 1.5])
        model1 = epimod.SuitabilityModel(suitability_table=suitability_table1)
        with pytest.raises(ValueError):
            model1.run(ds_clim)

    def test_plot_suitability_range(self):
        """Test the plot_suitability method with a temperature range."""
        model = epimod.SuitabilityModel(temperature_range=(0, 1))
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
        assert isinstance(result, hv.Image)
        assert result.kdims[0].pprint_label == "Temperature (°C)"
        assert result.kdims[1].pprint_label == "Precipitation (mm/day)"
        assert result.vdims[0].pprint_label == "Suitability"

    def test_get_max_suitability_range(self):
        """Test the get_max_suitability method with a temperature range."""
        model = epimod.SuitabilityModel(temperature_range=(0.5, 19))
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


@pytest.mark.parametrize(
    "suitability_vals_in,suitability_threshold, stat, quantile, rescale, "
    "suitability_vals_out_expected, temperature_range_out_expected",
    [
        ([[0, 1, 3], [0, 0, 0]], 0, "mean", None, False, [2 / 3, 0], None),
        ([[0, 2, 3], [0, 0, 0]], None, "median", None, True, [1, 0], None),
        (
            [[False, False, True, True], [False, True, False, False]],
            None,
            "quantile",
            0.5,
            False,
            [True, False],
            None,
        ),
        (
            [list(range(100))],
            None,
            "quantile",
            [0.25, 0.75],
            False,
            [[24.75, 74.25]],
            None,
        ),
        (
            [[0, 2, 4], [0, 1, 0]],
            None,
            None,
            None,
            "mean",
            [[0, 1, 2], [0, 0.5, 0]],
            None,
        ),
        (
            [[False], [True], [True], [False]],
            None,
            "median",
            None,
            False,
            None,
            (0.5, 2.5),
        ),
    ],
)
def test_reduce(
    suitability_vals_in,
    suitability_threshold,
    stat,
    quantile,
    rescale,
    suitability_vals_out_expected,
    temperature_range_out_expected,
):
    """Test the reduce method of SuitabilityModel."""
    suitability_table_in = xr.Dataset(
        {
            "suitability": (("temperature", "sample"), suitability_vals_in),
        },
        coords={"temperature": range(len(suitability_vals_in))},
    )
    model = epimod.SuitabilityModel(suitability_table=suitability_table_in)
    reduced_model = model.reduce(
        suitability_threshold=suitability_threshold,
        stat=stat,
        quantile=quantile,
        rescale=rescale,
    )
    if temperature_range_out_expected is not None:
        npt.assert_equal(
            reduced_model.temperature_range,
            temperature_range_out_expected,
        )
        return
    npt.assert_equal(
        reduced_model.suitability_table["suitability"]
        .transpose("temperature", ...)
        .values,
        suitability_vals_out_expected,
    )


def test_reduce_errors():
    """Test reduce method raises errors for invalid inputs."""
    suitability_table = xr.Dataset(
        {
            "suitability": (("temperature", "sample"), [[0, 1], [0, 0]]),
        },
        coords={"temperature": range(2)},
    )
    # Error using temperature range-based model
    with pytest.raises(
        ValueError, match="only available for suitability table-based models"
    ):
        epimod.SuitabilityModel(temperature_range=(0, 1)).reduce()
    # Error passing invalid stat
    with pytest.raises(ValueError, match="stat must be one of"):
        epimod.SuitabilityModel(suitability_table=suitability_table).reduce(stat="max")
    # Error passing stat="quantile" without quantile
    with pytest.raises(ValueError, match="quantile must be specified"):
        epimod.SuitabilityModel(suitability_table=suitability_table).reduce(
            stat="quantile"
        )
    # Error passing invalid rescale argument
    with pytest.raises(ValueError, match="rescale must be one of"):
        epimod.SuitabilityModel(suitability_table=suitability_table).reduce(
            rescale="max"
        )
