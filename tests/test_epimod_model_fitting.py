"""Unit tests for the _model_fitting module of the epimod subpackage."""

from unittest.mock import patch

import holoviews as hv
import numpy as np
import numpy.testing as npt
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
