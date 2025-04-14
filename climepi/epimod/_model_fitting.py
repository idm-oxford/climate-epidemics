import numpy as np
import pymc as pm

from climepi.epimod._model_classes import UncertainSuitabilityModel


class ParameterizedSuitabilityModel(UncertainSuitabilityModel):
    """
    Class for parameterized suitability models.

    Represents models in which a suitability metric (e.g., the basic reproduction
    number) is defined as a function of parameters, which in turn may depend on
    climate variables. Provides methods for inferring the dependence of parameters on
    temperature from laboratory data.

    Subclass of UncertainSuitabilityModel
    """

    def __init__(
        self,
        parameters=None,
        data=None,
        suitability_function=None,
        suitability_var_name=None,
        suitability_var_long_name=None,
    ):
        self.suitability_table = None
        self._parameters = parameters
        self._data = data
        self._suitability_function = suitability_function
        self._suitability_var_name = suitability_var_name
        self._suitability_var_long_name = suitability_var_long_name
        self._parameter_functions = None

    def fit_temperature_responses(self, data=None):
        """
        Fit the model to data.

        Parameters
        ----------
        data : dict, optional
            The data to fit the model to. If not provided, the data argument passed
            during initialization will be used (assuming it is provided). If provided,
            this will override the data passed during initialization.

        Returns
        -------
        None
        """
        if data is not None:
            self._data = data
        data = self._data
        raise NotImplementedError()

    def run(self, *args, **kwargs):
        """
        Run the epidemiological model on a given climate dataset.

        See the documentation for SuitabilityModel.run() for details.
        """
        if self._parameter_functions is None:
            raise ValueError(
                "Dependence of model parameters on climate variables has"
                "not been inferred. Use the fit_temperature_responses() method to infer"
                "parameter dependence before running this method."
            )
        return super().run(*args, **kwargs)

    def plot_suitability_region(self, **kwargs):
        """
        Plot suitability against temperature and (if relevant) precipitation.

        See the documentation for SuitabilityModel.plot_suitability_region() for
        details.
        """
        if self._parameter_functions is None:
            raise ValueError(
                "Dependence of model parameters on climate variables has"
                "not been inferred. Use the fit_temperature_responses() method to infer"
                "parameter dependence before running this method."
            )
        return super().plot_suitability_region(**kwargs)

    def get_max_suitability(self):
        """
        Get the maximum suitability value.

        See the documentation for SuitabilityModel.get_max_suitability() for
        details.
        """
        if self._parameter_functions is None:
            raise ValueError(
                "Dependence of model parameters on climate variables has"
                "not been inferred. Use the fit_temperature_responses() method to infer"
                "parameter dependence before running this method."
            )
        return super().get_max_suitability()


def fit_temperature_response(
    temperature_data=None,
    trait_data=None,
    curve_type=None,
    priors=None,
    **kwargs_sample,
):
    """
    Fit the dependence of a parameter on temperature.

    Parameters
    ----------
    temperature_data : array-like
        Vector of temperature values for which response data are available.
    trait_data : array-like
        Vector of values of the trait variable for the corresponding temperature values.
    curve_type : str
        The type of curve to fit. Options are 'quadratic' and 'briere'.
    priors : dict, optional
        Dictionary of priors for the parameters of the model. The keys should be the
        parameter names and the values should be the corresponding prior distributions.
    **kwargs_sample : dict
        Keyword arguments to pass to pymc.sample().

    Returns
    -------
    dict
        A dictionary containing the fitted parameters.
    """
    priors = priors or {}
    with pm.Model():
        if priors["steepness"] is not None:
            steepness = priors["steepness"]()
        else:
            steepness = pm.Gamma("steepness", alpha=1, beta=1)
        if priors["temperature_min"] is not None:
            temperature_min = priors["temperature_min"]()
        else:
            temperature_min = pm.Uniform("temperature_min", lower=0, upper=24)
        if priors["temperature_max"] is not None:
            temperature_max = priors["temperature_max"]()
        else:
            temperature_max = pm.Uniform("temperature_max", lower=25, upper=50)
        if priors["noise_variance"] is not None:
            noise_variance = priors["noise_variance"]()
        else:
            noise_variance = pm.Uniform("noise_variance", lower=0, upper=100)
        if curve_type == "quadratic":
            # mu = (
            #     steepness
            #     * (temperature_data - temperature_min)
            #     * (temperature_max - temperature_data)
            # )
            mu = np.maximum(
                steepness
                * (temperature_data - temperature_min)
                * (temperature_max - temperature_data),
                0,
            )
        elif curve_type == "briere":
            # mu = (
            #     steepness
            #     * temperature_data
            #     * (temperature_data - temperature_min)
            #     * (temperature_max - temperature_data) ** 0.5
            # )
            mu = (
                steepness
                * temperature_data
                * (temperature_data - temperature_min)
                * np.abs(temperature_max - temperature_data) ** 0.5
                * (temperature_data >= temperature_min)
                * (temperature_data <= temperature_max)
            )
        else:
            raise ValueError(
                f"Invalid curve_type: {curve_type}. Must be 'quadratic' or 'briere'."
            )
        likelihood = pm.Censored(  # noqa
            "likelihood",
            pm.Normal.dist(mu=mu, sigma=noise_variance**0.5),
            lower=0,
            observed=trait_data,
        )

        # Sample from the posterior distribution
        idata = pm.sample(**kwargs_sample)
    return idata
