"""Unit tests for the _model_fitting module of the epimod subpackage."""

from unittest.mock import patch

import numpy as np
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
