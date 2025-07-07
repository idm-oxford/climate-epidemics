"""Subpackage for epidemiological models."""

from climepi.epimod._examples import (  # noqa
    EXAMPLES,
    EXAMPLE_NAMES,
    get_example_model,
    get_example_temperature_response_data,
)
from climepi.epimod._model_classes import (  # noqa
    EpiModel,
    SuitabilityModel,
)
from climepi.epimod._model_fitting import (  # noqa
    fit_temperature_response,
    get_posterior_temperature_response,
    plot_fitted_temperature_response,
    ParameterizedSuitabilityModel,
)
