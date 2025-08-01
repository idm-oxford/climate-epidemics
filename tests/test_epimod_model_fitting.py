"""Unit tests for the _model_fitting module of the epimod subpackage."""

from unittest.mock import patch

import holoviews as hv
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytensor.tensor as pt
import pytest
import xarray as xr

from climepi import epimod


@patch("climepi.epimod._model_fitting.pm", autospec=True)
@patch("climepi.epimod._model_fitting._briere", autospec=True)
@pytest.mark.parametrize("specify_std_prior", [True, False])
def test_fit_temperature_response(mock_briere, mock_pm, specify_std_prior):
    """
    Test the fit_temperature_response function.

    For simplicity, we mock pymc methods, and just check that the different methods are
    called as expected.
    """

    def mock_sample(
        *args,
        draws=1000,
        chains=4,
        **kwargs,
    ):
        return xr.Dataset(
            {
                "response_parameter": (
                    ["chain", "draw"],
                    np.zeros((chains, draws)),
                ),
            },
            coords={
                "chain": np.arange(chains),
                "draw": np.arange(draws),
            },
        )

    mock_pm.sample.side_effect = mock_sample

    priors = {
        "scale": lambda: "scale_prior",
        "temperature_min": lambda: "temperature_min_prior",
        "temperature_max": lambda: "temperature_max_prior",
        "noise_precision": lambda: "noise_precision_prior",
    }
    if specify_std_prior:
        priors["noise_std"] = lambda: "noise_std_prior"

    result = epimod.fit_temperature_response(
        temperature_data="temperature_data",
        trait_data="trait_data",
        curve_type="briere",
        probability=True,
        priors=priors,
        step=lambda: "step",
        draws=10000,
        tune=500,
        thin=10,
    )

    # mock_pm.Gamma.assert_called_once()
    # assert mock_pm.Uniform.call_count == 2
    mock_briere.assert_called_once_with(
        "temperature_data",
        scale="scale_prior",
        temperature_min="temperature_min_prior",
        temperature_max="temperature_max_prior",
        probability=True,
        array_lib=pt,
    )
    if specify_std_prior:
        mock_pm.Normal.assert_called_once_with(
            "likelihood",
            mu=mock_briere.return_value,
            observed="trait_data",
            sigma="noise_std_prior",
        )
    else:
        mock_pm.Normal.assert_called_once_with(
            "likelihood",
            mu=mock_briere.return_value,
            observed="trait_data",
            tau="noise_precision_prior",
        )
    mock_pm.sample.assert_called_once_with(
        draws=10000,
        step="step",
        tune=500,
    )
    assert result.response_parameter.shape == (4, 1000)


def test_get_posterior_temperature_response():
    """Test the get_posterior_temperature_response function."""
    scale = np.array([[1, 2, 3], [4, 5, 6]])
    temperature_min = np.array([[0, 0.25, 0.5], [0.5, 0.7, 0.5]])
    temperature_max = np.array([[1, 1, 1], [1, 1, 1]])
    idata = xr.Dataset(
        {
            "scale": (["chain", "draw"], scale),
            "temperature_min": (["chain", "draw"], temperature_min),
            "temperature_max": (["chain", "draw"], temperature_max),
        },
    )
    result = epimod.get_posterior_temperature_response(
        idata,
        num_samples=6,
        curve_type="quadratic",
        trait_name="googly",
        trait_attrs={"back of": "the hand"},
    )

    temperature_vals_expected = np.linspace(-1, 2, 500)
    npt.assert_equal(result.temperature.values, temperature_vals_expected)

    temperature_vals_expanded = temperature_vals_expected[:, np.newaxis]
    scale_flat_expanded = scale.flatten()[np.newaxis]
    temperature_min_flat_expanded = temperature_min.flatten()[np.newaxis]
    temperature_max_flat_expanded = temperature_max.flatten()[np.newaxis]
    response_vals_expected = (
        scale_flat_expanded
        * (temperature_vals_expanded - temperature_min_flat_expanded)
        * (temperature_max_flat_expanded - temperature_vals_expanded)
        * (temperature_vals_expanded >= temperature_min_flat_expanded)
        * (temperature_vals_expanded <= temperature_max_flat_expanded)
    )
    npt.assert_allclose(
        np.sort(result.values, axis=1),
        np.sort(response_vals_expected, axis=1),
    )

    assert result.name == "googly"
    assert result.attrs == {"back of": "the hand"}
    assert result.temperature.attrs == {
        "long_name": "Temperature",
        "units": "°C",
    }


@patch(
    "climepi.epimod._model_fitting.get_posterior_temperature_response", autospec=True
)
def test_plot_fitted_temperature_response(mock_get_posterior):
    """Test the plot_fitted_temperature_response function."""
    da = xr.DataArray(
        np.random.rand(10, 3),
        dims=["temperature", "sample"],
        coords={
            "temperature": np.linspace(0, 1, 10),
            "sample": np.arange(3),
        },
        name="some_trait",
    )

    def _mock_get_posterior(*args, **kwargs):
        return da

    mock_get_posterior.side_effect = _mock_get_posterior

    p = epimod.plot_fitted_temperature_response(
        "idata",
        temperature_vals="temperature_vals",
        temperature_data=np.array([0, 0.5, 1]),
        trait_data=np.array([0, 1, 0]),
        curve_type="briere",
        probability=True,
    )
    assert isinstance(p, hv.Overlay)
    npt.assert_equal(
        p.Curve.Median_response.dframe().temperature.values,
        da.temperature.values,
    )
    npt.assert_equal(
        p.Curve.Median_response.data.some_trait.values,
        da.median(dim="sample").values,
    )
    npt.assert_equal(
        p.Area.A_95_percent_credible_interval.dframe().temperature.values,
        da.temperature.values,
    )
    npt.assert_equal(
        p.Area.A_95_percent_credible_interval.data.lower.values,
        da.quantile(0.025, dim="sample").values,
    )
    npt.assert_equal(
        p.Area.A_95_percent_credible_interval.data.upper.values,
        da.quantile(0.975, dim="sample").values,
    )
    npt.assert_equal(
        p.Scatter.I.dframe().temperature.values,
        np.array([0, 0.5, 1]),
    )
    npt.assert_equal(
        p.Scatter.I.data.trait.values,
        np.array([0, 1, 0]),
    )

    mock_get_posterior.assert_called_once_with(
        idata="idata",
        temperature_vals="temperature_vals",
        curve_type="briere",
        probability=True,
        trait_name=None,
        trait_attrs=None,
    )


class TestParameterizedSuitabilityModel:
    """Test the ParameterizedSuitabilityModel class."""

    def test_init(self):
        """
        Test the initialization of the ParameterizedSuitabilityModel.

        Includes testing of the _extract_data method, which is called during
        initialization.
        """
        model = epimod.ParameterizedSuitabilityModel(
            parameters={
                "general": {"curve_type": "hello"},
                "kenobi": {"curve_type": "there"},
            },
            data=pd.DataFrame(
                {
                    "trait_name": ["general", "general", "kenobi"],
                    "temperature": [0, 1, 0],
                    "trait_value": [1, 2, 3],
                }
            ),
            suitability_function="some_function",
        )
        assert model.suitability_table is None
        assert model._parameters["general"]["curve_type"] == "hello"
        npt.assert_equal(
            model._parameters["general"]["temperature_data"],
            [0, 1],
        )
        npt.assert_equal(
            model._parameters["general"]["trait_data"],
            [1, 2],
        )
        assert model._parameters["kenobi"]["curve_type"] == "there"
        npt.assert_equal(
            model._parameters["kenobi"]["temperature_data"],
            [0],
        )
        npt.assert_equal(
            model._parameters["kenobi"]["trait_data"],
            [3],
        )
        assert model._suitability_function == "some_function"
        assert model._suitability_var_name == "suitability"
        assert model._suitability_var_long_name == "Suitability"

    def test_overridden_methods(self):
        """
        Test methods that override the parent SuitabilityModel methods.

        Tests that the _check_fitting and _check_suitability_table methods raise errors
        when the model is not fitted or the suitability table is not constructed, and
        that the parent methods are called when the checks pass.
        """
        model = epimod.ParameterizedSuitabilityModel()
        for method, parent_method_name, kwargs in zip(
            [model.run, model.plot_suitability, model.get_max_suitability],
            [
                "climepi.epimod._model_fitting.SuitabilityModel." + x
                for x in ["run", "plot_suitability", "get_max_suitability"]
            ],
            [
                {"ds_clim": "some_climate_data"},
                {},
                {},
            ],
            strict=True,
        ):
            with patch(parent_method_name, autospec=True) as mock_parent_method:
                model._parameters = {
                    "hello": "there",
                    "general": {"curve_type": "kenobi"},
                }
                model.suitability_table = None
                with pytest.raises(ValueError, match="Need to fit the model"):
                    method(**kwargs)
                model._parameters["general"]["idata"] = "some_idata"
                with pytest.raises(
                    ValueError, match="Need to construct the suitability table"
                ):
                    method(**kwargs)
                model.suitability_table = "some_table"
                method(**kwargs)
                mock_parent_method.assert_called_once_with(model, **kwargs)

    @patch(
        "climepi.epimod._model_fitting.fit_temperature_response",
        autospec=True,
    )
    def test_fit_temperature_responses(self, mock_fit_temperature_response):
        """Test the fit_temperature_responses method."""

        def _mock_fit(*args, trait_data=None, **kwargs):
            return ["i" + trait_data]

        mock_fit_temperature_response.side_effect = _mock_fit

        model = epimod.ParameterizedSuitabilityModel(
            parameters={
                "hello": "there",
                "general": {
                    "curve_type": "kenobi",
                    "temperature_data": "temperature_data_general",
                    "trait_data": "data_general",
                },
                "bold": {
                    "curve_type": "one",
                    "temperature_data": "temperature_data_bold",
                    "trait_data": "data_bold",
                    "probability": True,
                    "priors": "priors_bold",
                },
            }
        )

        idata_dict = model.fit_temperature_responses(step="step", draws=10)

        # Assert that the mock was called with the expected arguments
        assert mock_fit_temperature_response.call_count == 2
        mock_fit_temperature_response.assert_any_call(
            temperature_data="temperature_data_general",
            trait_data="data_general",
            curve_type="kenobi",
            probability=False,
            priors=None,
            step="step",
            thin=1,
            draws=10,
        )
        mock_fit_temperature_response.assert_any_call(
            temperature_data="temperature_data_bold",
            trait_data="data_bold",
            curve_type="one",
            probability=True,
            priors="priors_bold",
            step="step",
            thin=1,
            draws=10,
        )
        assert idata_dict == {
            "general": ["idata_general"],
            "bold": ["idata_bold"],
        }
        assert model._parameters["general"]["idata"] == ["idata_general"]
        assert model._parameters["bold"]["idata"] == ["idata_bold"]

    @patch(
        "climepi.epimod._model_fitting.plot_fitted_temperature_response",
        autospec=True,
    )
    @patch("climepi.epimod._model_fitting.hv.Layout", autospec=True)
    def test_plot_fitted_temperature_responses(self, mock_layout, mock_plot_response):
        """Test the plot_fitted_temperature_responses method."""

        def _mock_plot_response(*args, idata=None, **kwargs):
            return idata

        mock_plot_response.side_effect = _mock_plot_response

        model = epimod.ParameterizedSuitabilityModel(
            parameters={
                "hello": "there",
                "general": {  # not fitted so shouldn't be plotted
                    "curve_type": "kenobi"
                },
                "bold": {
                    "curve_type": "one",
                    "idata": "idata_bold",
                    "temperature_data": "temperature_data_bold",
                    "trait_data": "data_bold",
                    "attrs": {"fine": "addition"},
                },
            }
        )

        for parameter_names in [None, "bold", ["bold"]]:
            p = model.plot_fitted_temperature_responses(
                parameter_names=parameter_names, temperature_vals="temperature_vals"
            )

            assert p == mock_layout.return_value.opts.return_value
            mock_plot_response.assert_called_once_with(
                idata="idata_bold",
                temperature_vals="temperature_vals",
                temperature_data="temperature_data_bold",
                trait_data="data_bold",
                curve_type="one",
                probability=False,
                trait_name="bold",
                trait_attrs={"fine": "addition"},
            )
            mock_layout.assert_called_once_with(["idata_bold"])
            mock_layout.reset_mock()
            mock_plot_response.reset_mock()

    @patch(
        "climepi.epimod._model_fitting.get_posterior_temperature_response",
        autospec=True,
    )
    def test_construct_suitability_table(self, mock_get_posterior_response):
        """Test the construct_suitability_table method."""
        temperature_vals = np.array([0, 1])
        precipitation_vals = np.array([0.1, 0.2])

        def _mock_get_posterior_response(*args, idata=None, **kwargs):
            if idata == "general_idata":
                da_posterior = xr.DataArray(
                    [[1, 2], [3, 4]],
                    coords={
                        "temperature": temperature_vals,
                        "sample": [0, 1],
                    },
                    name="general",
                )
                da_posterior["temperature"] = da_posterior["temperature"].assign_attrs(
                    long_name="Temperature",
                    units="°C",
                )
                return da_posterior
            return None

        mock_get_posterior_response.side_effect = _mock_get_posterior_response

        model = epimod.ParameterizedSuitabilityModel(
            parameters={
                "hello": 1,
                "general": {
                    "curve_type": "kenobi",
                    "idata": "general_idata",
                },
                "bold": lambda temperature=None, precipitation=None: precipitation,
            },
            suitability_function=lambda hello, general, bold: hello + general + bold,
            suitability_var_long_name="Grievous",
        )
        suitability_table = model.construct_suitability_table(
            temperature_vals=temperature_vals,
            precipitation_vals=precipitation_vals,
        )
        npt.assert_equal(
            suitability_table.suitability.transpose(
                "temperature", "precipitation", "sample"
            ).values,
            1
            + np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]]])
            + np.array([[[0.1, 0.1], [0.2, 0.2]], [[0.1, 0.1], [0.2, 0.2]]]),
        )
        assert suitability_table.suitability.attrs["long_name"] == "Grievous"
        assert suitability_table.temperature.attrs["long_name"] == "Temperature"
        assert suitability_table.precipitation.attrs["long_name"] == "Precipitation"


def test_get_posterior_min_optimal_max_temperature():
    """Test the get_posterior_min_optimal_max_temperature method."""
    suitability_model = epimod.ParameterizedSuitabilityModel(parameters={})
    suitability_model.suitability_table = xr.Dataset(
        {
            "suitability": (
                ("sample", "temperature"),
                np.array([[0, 0.1, 0.2, 0.1, 0], [0, 1, 0.8, 0.5, 0]]),
            ),
        },
        coords={
            "temperature": [0, 1, 2, 3, 4],
            "sample": [0, 1],
        },
    )

    result1 = suitability_model.get_posterior_min_optimal_max_temperature()
    npt.assert_equal(result1.temperature_min.values, [0.5, 0.5])
    npt.assert_equal(result1.temperature_optimal.values, [2, 1])
    npt.assert_equal(result1.temperature_max.values, [3.5, 3.5])

    result2 = suitability_model.get_posterior_min_optimal_max_temperature(
        suitability_threshold=0.15
    )
    npt.assert_equal(result2.temperature_min.values, [1.5, 0.5])
    npt.assert_equal(result2.temperature_optimal.values, [2, 1])
    npt.assert_equal(result2.temperature_max.values, [2.5, 3.5])

    with pytest.raises(
        ValueError, match="Minimum and/or maximum suitable temperatures do not exist."
    ):
        suitability_model.get_posterior_min_optimal_max_temperature(
            suitability_threshold=0.5
        )

    with pytest.raises(
        ValueError,
        match="This method only works for models that depend on temperature only.",
    ):
        suitability_model.suitability_table["suitability"] = (
            suitability_model.suitability_table["suitability"].expand_dims(
                precipitation=1
            )
        )
        suitability_model.get_posterior_min_optimal_max_temperature()
