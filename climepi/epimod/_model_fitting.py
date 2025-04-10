from pymc import Gamma, Model, TruncatedNormal, Uniform, sample

from climepi.epimod._model_classes import UncertainSuitabilityModel


class ParameterizedSuitabilityModel(UncertainSuitabilityModel):
    """
    Class for parameterized suitability models.

    Represents models in which a suitability metric (e.g., the basic reproduction
    number) is defined as a function of parameters, which in turn may depend on
    climate variables. Provides methods for inferring the dependence of parameters on
    climate variables from laboratory data.

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

    def construct_niche(self, data=None):
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
                "not been inferred. Use the fit() method to infer parameter"
                "dependence before running this method."
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
                "not been inferred. Use the fit() method to infer parameter"
                "dependence before running this method."
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
                "not been inferred. Use the fit() method to infer parameter"
                "dependence before running this method."
            )
        return super().get_max_suitability()


def fit_parameter_dependence(
    climate_var_data=None,
    response_var_data=None,
    curve_type=None,
    priors=None,
    samples=None,
):
    """
    Fit the dependence of a parameter on a climate variable.

    Parameters
    ----------
    climate_var_data : array-like
        Vector of values of the climate variable (e.g., temperature) for which response
        variable data are available.
    response_var_data : array-like
        Vector of values of the response variable (e.g., suitability) corresponding to
        the climate variable data.
    curve_type : str
        The type of curve to fit. Options are 'quadratic' and 'briere'.
    priors : dict, optional
        Dictionary of priors for the parameters of the model. The keys should be the
        parameter names and the values should be the corresponding prior distributions.
    samples : int, optional
        Number of samples to draw from the posterior distribution. If not provided, the
        default for the pymc.sample function will be used.

    Returns
    -------
    dict
        A dictionary containing the fitted parameters.
    """
    priors = {
        **{
            "steepness": Gamma("steepness", alpha=1, beta=1),
            "climate_var_min": Uniform("climate_var_min", lower=0, upper=24),
            "climate_var_max": Uniform("climate_var_max", lower=25, upper=50),
            "noise_variance": Uniform("noise_variance", lower=0, upper=50),
        },
        **(priors if priors is not None else {}),
    }
    with Model() as model:
        steepness = priors[steepness]
        climate_var_min = priors[climate_var_min]
        climate_var_max = priors[climate_var_max]
        noise_variance = priors[noise_variance]

        mu = (
            steepness
            * (climate_var_data - climate_var_min)
            * (climate_var_data - climate_var_max)
        )

        if curve_type == "quadratic":
            mu = (
                steepness
                * (climate_var_data - climate_var_min)
                * (climate_var_max - climate_var_data)
            )
        elif curve_type == "briere":
            mu = (
                steepness
                * climate_var_data
                * (climate_var_data - climate_var_min)
                * (climate_var_max - climate_var_data) ** 0.5
            )
        likelihood = TruncatedNormal(
            "likelihood",
            mu=mu,
            sigma=noise_variance**0.5,
            lower=0,
            observed=response_var_data,
        )

        # Sample from the posterior distribution
        idata = sample()
