import copy
import numbers

import arviz as az
import holoviews
import hvplot.xarray  # noqa
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from climepi.epimod._model_classes import SuitabilityModel


class ParameterizedSuitabilityModel(SuitabilityModel):
    """
    Class for parameterized suitability models.

    Represents models in which a suitability metric (e.g., the basic reproduction
    number) is defined as a function of parameters, which in turn may depend on
    climate variables. Provides methods for inferring the dependence of parameters on
    temperature from laboratory data.

    Subclass of SuitabilityModel
    """

    def __init__(
        self,
        parameters=None,
        data=None,
        suitability_function=None,
        suitability_var_name="suitability",
        suitability_var_long_name="Suitability",
    ):
        self.suitability_table = None
        self._parameters = parameters
        self._suitability_function = suitability_function
        self._suitability_var_name = suitability_var_name
        self._suitability_var_long_name = suitability_var_long_name
        if data is not None:
            self._extract_data(data)

    def fit_temperature_responses(self, step=None, thin=1, **kwargs_sample):
        """
        Fit the model to data.

        Parameters
        ----------
        step : callable, optional
            A callable that returns a pymc step method for sampling. If None, the
            default step method (DEMetropolisZ) is used.
        thin : int, optional
            Only keep one in every `thin` samples. Default is 1 (no thinning).
        **kwargs_sample : dict, optional
            Keyword arguments to pass to pymc.sample().
        """
        parameters = copy.deepcopy(self._parameters)
        idata_dict = {}
        for parameter_name, parameter_dict in parameters.items():
            if not isinstance(parameter_dict, dict):
                continue
            print(f"Fitting temperature response for parameter: {parameter_name}")
            idata = fit_temperature_response(
                temperature_data=parameter_dict["temperature_data"],
                trait_data=parameter_dict["trait_data"],
                curve_type=parameter_dict["curve_type"],
                probability=parameter_dict.get("probability", False),
                priors=parameter_dict.get("priors", None),
                step=step,
                thin=thin,
                **kwargs_sample,
            )
            parameter_dict["idata"] = idata
            idata_dict[parameter_name] = idata
        self._parameters = parameters
        return idata_dict

    def plot_fitted_temperature_responses(
        self, parameter_names=None, temperature_vals=None
    ):
        """
        Plot the fitted temperature responses.

        Parameters
        ----------
        parameters : str or list of str, optional
            The name of the parameter(s) to plot. If None, all fitted parameters
            will be plotted.
        temperature_vals : array-like, optional
            Vector of temperature values for which each response is to be plotted.
            If not provided, a default range is generated on a per-parameter basis
            based on the minimum and maximum temperature values in the posterior
            distribution.
        """
        if parameter_names is None:
            parameter_names = [
                parameter_name
                for parameter_name, parameter_dict in self._parameters.items()
                if isinstance(parameter_dict, dict) and "idata" in parameter_dict
            ]
        elif isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        plots = []
        for parameter_name in parameter_names:
            parameter_dict = self._parameters[parameter_name]
            if "idata" not in parameter_dict:
                continue
            plots.append(
                plot_fitted_temperature_response(
                    idata=parameter_dict["idata"],
                    temperature_vals=temperature_vals,
                    temperature_data=parameter_dict["temperature_data"],
                    trait_data=parameter_dict["trait_data"],
                    curve_type=parameter_dict["curve_type"],
                    probability=parameter_dict.get("probability", False),
                    trait_name=parameter_name,
                    trait_attrs=parameter_dict.get("attrs", None),
                )
            )
        return holoviews.Layout(plots).opts(shared_axes=False)

    def construct_suitability_table(
        self,
        temperature_vals=None,
        precipitation_vals=None,
        num_samples=None,
        rescale=False,
    ):
        """
        Construct a suitability table based on the fitted parameters.

        Parameters
        ----------
        temperature_vals : array-like
            Vector of temperature values for which the suitability is to be computed.
            Must be provided.
        precipitation_vals : array-like, optional
            Vector of precipitation values for which the suitability is to be computed.
            Only needed if the suitability function depends on precipitation.
        num_samples : int, optional
            Number of samples to draw from the posterior distribution of the fitted
            parameters. If None, all samples are used.
        """
        parameter_vals = {}
        for parameter_name, parameter_entry in self._parameters.items():
            if isinstance(parameter_entry, dict):
                idata = parameter_entry["idata"]
                da_response = get_posterior_temperature_response(
                    idata=idata,
                    num_samples=num_samples,
                    temperature_vals=temperature_vals,
                    curve_type=parameter_entry["curve_type"],
                    probability=parameter_entry.get("probability", False),
                    trait_name=parameter_name,
                    trait_attrs=parameter_entry.get("attrs", None),
                )
                parameter_vals[parameter_name] = da_response
            elif isinstance(parameter_entry, numbers.Number):
                parameter_vals[parameter_name] = parameter_entry
            elif callable(parameter_entry):
                ds_coords = xr.Dataset(coords={"temperature": temperature_vals})
                ds_coords["temperature"].attrs = {
                    "long_name": "Temperature",
                    "units": "°C",
                }
                if precipitation_vals is not None:
                    ds_coords = ds_coords.assign_coords(
                        precipitation=("precipitation", precipitation_vals)
                    )
                    ds_coords["precipitation"].attrs = {
                        "long_name": "Precipitation",
                        "units": "mm/day",
                    }
                parameter_vals[parameter_name] = parameter_entry(**ds_coords.coords)
            else:
                raise ValueError(
                    f"Invalid parameter entry for '{parameter_name}': {parameter_entry}"
                )
        suitability_table = (
            self._suitability_function(**parameter_vals)
            .assign_attrs(long_name=self._suitability_var_long_name)
            .to_dataset(name=self._suitability_var_name)
        )
        if rescale:
            suitability_table = (
                suitability_table / suitability_table.mean(dim="sample").max()
            )
        self.suitability_table = suitability_table
        return suitability_table.copy()

    def run(self, *args, **kwargs):
        """
        Run the epidemiological model on a given climate dataset.

        See the documentation for SuitabilityModel.run() for details.
        """
        self._check_fitting_suitability_table()
        return super().run(*args, **kwargs)

    def plot_suitability_region(self, **kwargs):
        """
        Plot suitability against temperature and (if relevant) precipitation.

        See the documentation for SuitabilityModel.plot_suitability_region() for
        details.
        """
        self._check_fitting_suitability_table()
        return super().plot_suitability_region(**kwargs)

    def get_max_suitability(self):
        """
        Get the maximum suitability value.

        See the documentation for SuitabilityModel.get_max_suitability() for
        details.
        """
        self._check_fitting_suitability_table()
        return super().get_max_suitability()

    def _extract_data(self, data):
        # Extracts temperature and trait data from the provided dataset.
        parameters = copy.deepcopy(self._parameters)
        for parameter_name, parameter_dict in parameters.items():
            if not isinstance(parameter_dict, dict):
                continue
            data_subset = data[data["trait_name"] == parameter_name]
            parameter_dict["temperature_data"] = data_subset["temperature"].values
            parameter_dict["trait_data"] = data_subset["trait_value"].values
        self._parameters = parameters

    def _check_fitting_suitability_table(self):
        # Checks if the suitability table has been constructed.
        if self.suitability_table is None:
            raise ValueError(
                "Need to fit the model with fit_temperature_responses() and/or "
                "construct the suitability table with construct_suitability_table() "
                "before running this method."
            )


def fit_temperature_response(
    temperature_data=None,
    trait_data=None,
    curve_type=None,
    probability=False,
    priors=None,
    step=None,
    thin=1,
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
    probability : bool, optional
        If True, the response is constrained to be between 0 and 1. Default is False.
    priors : dict, optional
        Dictionary of priors for the parameters of the model. The keys should be the
        parameter names and the values should callable functions that return pymc
        distributions. If None, default priors are used based on the curve type.
    step : callable, optional
        A callable that returns a pymc step method for sampling. If None, the default
        step method (DEMetropolisZ) is used.
    thin: int, optional
        Only keep one in every `thin` samples. Default is 1 (no thinning).
    **kwargs_sample : dict
        Keyword arguments to pass to pymc.sample().

    Returns
    -------
    dict
        A dictionary containing the fitted parameters.
    """
    curve_func = _get_curve_func(curve_type)
    priors = priors or {}
    priors = {
        "steepness": (lambda: pm.Gamma("steepness", alpha=1, beta=1))
        if curve_type == "quadratic"
        else (lambda: pm.Gamma("steepness", alpha=1, beta=10))
        if curve_type == "briere"
        else (lambda: None),
        "temperature_min": lambda: pm.Uniform("temperature_min", lower=0, upper=24),
        "temperature_max": lambda: pm.Uniform("temperature_max", lower=25, upper=50),
        "noise_precision": lambda: pm.Gamma(
            "noise_precision", alpha=0.0001, beta=0.0001
        ),
        **priors,
    }
    with pm.Model():
        steepness = priors["steepness"]()
        temperature_min = priors["temperature_min"]()
        temperature_max = priors["temperature_max"]()
        mu = curve_func(
            temperature_data,
            steepness=steepness,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
            # probability=False,  # clipping mean at 1 can cause issues with sampling
            probability=probability,
            array_lib=pt,
        )
        if "noise_std" in priors:
            noise_std = priors["noise_std"]()
            kwargs_normal = {"sigma": noise_std}
        else:
            noise_precision = priors["noise_precision"]()
            kwargs_normal = {"tau": noise_precision}
        # kwargs_likelihood = {"lower": 0, "observed": trait_data}
        # if probability:
        #     kwargs_likelihood["upper"] = 1
        # likelihood = pm.Censored(  # noqa
        #     "likelihood",
        #     pm.Normal.dist(mu=mu, **kwargs_normal),
        #     **kwargs_likelihood,
        # )
        likelihood = pm.Normal(
            "likelihood",
            mu=mu,
            observed=trait_data,
            **kwargs_normal,
        )
        # Sample from the posterior distribution using the DEMetropolisZ algorithm
        # (NUTS not suitable given non-differentiable likelihood?)
        kwargs_sample = {
            "step": pm.DEMetropolisZ() if step is None else step(),
            **kwargs_sample,
        }
        idata = pm.sample(**kwargs_sample)
    idata = idata.sel(draw=slice(None, None, thin))
    return idata


def get_posterior_temperature_response(
    idata,
    num_samples=None,
    temperature_vals=None,
    curve_type=None,
    probability=None,
    trait_name=None,
    trait_attrs=None,
):
    """
    Get the posterior distribution of the fitted temperature response.

    Parameters
    ----------
    idata : pymc.backends.base.MultiTrace
        The posterior distribution of the fitted parameters (as returned by
        fit_temperature_response()).
    num_samples : int, optional
        Number of samples to draw from the posterior distribution. If None, all samples
        are used.
    temperature_vals : array-like, optional
        Vector of temperature values for which the response is to be computed. If not
        provided, a default range is generated based on the minimum and maximum
        temperature values in the posterior distribution.
    curve_type : str
        The type of curve fitted. Options are 'quadratic' and 'briere'.
    probability : bool, optional
        If True, the response is constrained to be between 0 and 1. Default is False.
    trait_name : str, optional
        The name of the trait variable. If None, the response variable is named
        "response".
    trait_attrs : dict, optional
        Additional attributes to assign to the trait variable in the returned dataset.
        If None, no additional attributes are assigned.

    Returns
    -------
    xarray.DataArray
        The posterior distribution of the fitted temperature response, with dimensions
        "temperature" and "sample".
    """
    curve_func = _get_curve_func(curve_type)
    ds_posterior = az.extract(
        data=idata,
        var_names=["steepness", "temperature_min", "temperature_max"],
        num_samples=num_samples,
    )
    ds_posterior = ds_posterior.drop_vars(["chain", "draw"]).assign_coords(
        sample=np.arange(ds_posterior.sample.size)
    )
    if temperature_vals is None:
        temperature_vals = np.linspace(
            ds_posterior.temperature_min.min().item(),
            ds_posterior.temperature_max.max().item(),
            500,
        )
    ds_posterior_expanded = ds_posterior.assign_coords(temperature=temperature_vals)
    da_posterior_response = curve_func(
        temperature=ds_posterior_expanded.temperature,
        steepness=ds_posterior_expanded.steepness,
        temperature_min=ds_posterior_expanded.temperature_min,
        temperature_max=ds_posterior_expanded.temperature_max,
        probability=probability,
        array_lib=xr,
    )
    da_posterior_response.name = trait_name if trait_name is not None else "response"
    if trait_attrs is not None:
        da_posterior_response.attrs.update(trait_attrs)
    da_posterior_response.temperature.attrs["long_name"] = "Temperature"
    da_posterior_response.temperature.attrs["units"] = "°C"
    return da_posterior_response


def plot_fitted_temperature_response(
    idata,
    temperature_vals=None,
    temperature_data=None,
    trait_data=None,
    curve_type=None,
    probability=False,
    trait_name=None,
    trait_attrs=None,
):
    """
    Plot the posterior distribution of the fitted temperature response.

    Parameters
    ----------
    idata : pymc.backends.base.MultiTrace
        The posterior distribution of the fitted parameters.
    temperature_vals : array-like, optional
        Vector of temperature values for which the response is to be plotted. If not
        provided, a default range is generated based on the minimum and maximum
        temperature values in the posterior distribution.
    temperature_data : array-like, optional
        Vector of temperature values for which response data are available.
    trait_data : array-like, optional
        Vector of values of the trait variable for the corresponding temperature values.
    curve_type : str
        The type of curve fitted. Options are 'quadratic' and 'briere'.
    probability : bool, optional
        If True, the response is constrained to be between 0 and 1. Default is False.
    trait_name : str, optional
        The name of the trait variable.
    trait_attrs : dict, optional
        Additional attributes to assign to the trait variable in the plotted dataset.
    """
    da_posterior_response = get_posterior_temperature_response(
        idata=idata,
        temperature_vals=temperature_vals,
        curve_type=curve_type,
        probability=probability,
        trait_name=trait_name,
        trait_attrs=trait_attrs,
    )
    da_response_quantiles = da_posterior_response.quantile(
        [0.025, 0.5, 0.975], dim="sample", keep_attrs=True
    ).assign_coords(quantile=["lower", "median", "upper"])
    return (
        da_response_quantiles.sel(quantile="median", drop=True).hvplot.line(
            label="Median response"
        )
        * da_response_quantiles.to_dataset(dim="quantile").hvplot.area(
            y="lower",
            y2="upper",
            alpha=0.2,
            label="95% credible interval",
        )
        * xr.Dataset(
            {"trait": ("temperature", trait_data)},
            coords={"temperature": temperature_data},
        ).hvplot.scatter()
    )


def _bounded_quadratic(
    temperature,
    steepness=None,
    temperature_min=None,
    temperature_max=None,
    probability=None,
    array_lib=np,
):
    response = (
        steepness * (temperature - temperature_min) * (temperature_max - temperature)
    )
    response = array_lib.where(temperature >= temperature_min, response, 0)
    response = array_lib.where(temperature <= temperature_max, response, 0)
    if probability:
        response = response.clip(0, 1)
    return response


def _briere(
    temperature,
    steepness=None,
    temperature_min=None,
    temperature_max=None,
    probability=None,
    array_lib=np,
):
    response = (
        steepness
        * temperature
        * (temperature - temperature_min)
        * np.abs(temperature_max - temperature) ** 0.5
    )
    response = array_lib.where(temperature >= temperature_min, response, 0)
    response = array_lib.where(temperature <= temperature_max, response, 0)
    if probability:
        response = response.clip(0, 1)
    return response


def _get_curve_func(curve_type):
    # Returns the appropriate curve function based on the curve type.
    if curve_type == "briere":
        return _briere
    elif curve_type == "quadratic":
        return _bounded_quadratic
    else:
        raise ValueError(
            f"Invalid curve_type: {curve_type}. Must be 'briere' or 'quadratic'."
        )
