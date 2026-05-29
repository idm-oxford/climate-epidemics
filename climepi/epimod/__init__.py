"""Subpackage for epidemiological models."""

from climepi.epimod._examples import (  # noqa
    EXAMPLES,
    EXAMPLE_NAMES,
    get_example_model,
    get_example_temperature_response_data,
)
from climepi.epimod._base_classes import (  # noqa
    EpiModel,
    SuitabilityModel,
)

# Attributes provided by _model_fitting are loaded lazily so that `import climepi` does
# not pay the cost of pymc / pytensor / arviz_base unless the model-fitting API is
# actually needed.
_LAZY_MODEL_FITTING_ATTRS = frozenset(
    {
        "ParameterizedSuitabilityModel",
        "fit_temperature_response",
        "get_posterior_temperature_response",
        "plot_fitted_temperature_response",
    }
)


def __getattr__(name):
    if name in _LAZY_MODEL_FITTING_ATTRS:
        from climepi.epimod import _model_fitting

        return getattr(_model_fitting, name)
    raise AttributeError(f"module 'climepi.epimod' has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_LAZY_MODEL_FITTING_ATTRS})
