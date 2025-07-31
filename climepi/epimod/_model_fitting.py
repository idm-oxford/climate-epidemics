import copy
import numbers

import arviz as az
import holoviews as hv
import hvplot.xarray  # noqa
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from climepi.epimod._base_classes import SuitabilityModel


class ParameterizedSuitabilityModel(SuitabilityModel):
    """
    Class for parameterized suitability models.

    Represents models in which a suitability metric (e.g., the basic reproduction
    number) is defined as a function of parameters, which in turn may depend on
    climate variables. Provides methods for inferring the dependence of parameters on
    temperature from laboratory data.

    Subclass of SuitabilityModel.

    Parameters
    ----------
    parameters : dict, optional
        Dictionary of model parameters. Each key is a parameter name, and the value
        is either a number (constant parameter), a callable (function which takes
        keyword arguments `temperature` and, if the model is dependent on
        precipitation, `precipitation`, which should be able to handle xarray DataArrays
        as inputs), or, for temperature-dependent parameters that
        are to be fitted, a dictionary with the following keys:
            curve_type : str
                The type of curve to fit. Options are 'quadratic' (response =
                a*(T-T_min)*(T-T_max) for T_min < T < T_max, where T is temperature, and
                zero otherwise) and 'briere' (response = a*T*(T-T_min)*sqrt(T_max-T) for
                T_min < T < T_max, where T is temperature, and zero otherwise). In both
                cases, a is the scale parameter, T_min is the minimum temperature,
                and T_max is the maximum temperature, and normally distributed noise is
                assumed on the response.
            probability : bool, optional
                If True, the fitted curve is constrained to be between 0 and 1. Default
                is False.
            priors : dict, optional
                Dictionary of priors for the parameters of the model. The keys should be
                the parameter names ('scale', 'temperature_min', 'temperature_max',
                and either 'noise_std' or 'noise_precision' - see description for
                'curve_type' above) and the values should callable functions that return
                pymc distributions. Where not specified, default priors are used based
                on the curve type (as used by Mordecai et al., PLoS Negl Trop Dis 2017).
            attrs : dict, optional
                Additional attributes to assign to the trait variable in the posterior
                response DataArray (in particular, the 'long_name' and 'units'
                attributes are used by hvplot to automatically label axes in the
                plot_fitted_temperature_responses() method).
        Additionally, the following can either be provided as part of the parameter
        dictionary, or will be automatically extracted from the `data` argument if
        provided:
            temperature_data: array-like
                Vector of temperature values for which response data are available.
            trait_data: array-like
                Vector of trait values corresponding to the temperature data.
        data : pandas.DataFrame, optional
            A DataFrame containing the temperature and trait data for the parameters to
            be fitted. The DataFrame should have columns "trait_name", "temperature",
            and "trait_value".
    suitability_function : callable
        A callable that takes the model parameters as keyword arguments and returns a
        suitability metric (e.g., the basic reproduction number). The callable should
        be able to handle xarray DataArrays as inputs.

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
        parameters = self._parameters
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

        Note that this method requires that the model has been fitted to data
        using fit_temperature_responses() before it can be called.

        Parameters
        ----------
        parameter_names : str or list of str, optional
            The name of the parameter(s) to plot. If None, all fitted parameters
            will be plotted.
        temperature_vals : array-like, optional
            Vector of temperature values for which each response is to be plotted.
            If not provided, a default range is generated on a per-parameter basis
            based on the minimum and maximum temperature values in the posterior
            distribution.

        Returns
        -------
        holoviews.Layout
            A holoviews Layout object containing the plots of the fitted temperature
            responses for the specified parameters.
        """
        self._check_fitting()
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
        return hv.Layout(plots).opts(shared_axes=False)

    def construct_suitability_table(
        self,
        temperature_vals=None,
        precipitation_vals=None,
        num_samples=None,
        rescale=False,
    ):
        """
        Construct a suitability table based on the fitted parameters.

        Note that this method requires that the model has been fitted to data
        using fit_temperature_responses() before it can be called.

        Parameters
        ----------
        temperature_vals : array-like
            Vector of temperature values for which the suitability is to be computed.
            Must be provided.
        precipitation_vals : array-like
            Vector of precipitation values for which the suitability is to be computed.
            Only needed for models that depend on precipitation.
        num_samples : int, optional
            Number of samples to draw from the posterior distribution of the fitted
            parameters. If None, all samples are used.
        """
        self._check_fitting()
        parameter_vals = {}
        for parameter_name, parameter_entry in self._parameters.items():
            if isinstance(parameter_entry, dict):
                try:
                    idata = parameter_entry.get("idata")
                except KeyError as e:
                    raise ValueError(
                        f"Parameter '{parameter_name}' does not have fitted data. "
                        "Please fit the model using fit_temperature_responses() first."
                    ) from e
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

    def get_posterior_min_peak_max_temperature(self, suitability_threshold=0):
        """
        Get posterior distributions of minimum, peak, and maximum temperatures.

        Calculates the posterior distributions of the minimum/maximum temperatures that
        are considered suitable, as well as the temperature at which the suitability is
        at its peak.

        Note that this method requires that the model has been fitted to data using
        fit_temperature_responses() and that a suitability table has been constructed
        using construct_suitability_table() before it can be called. Note also that
        this method will only work if suitability is a function of temperature only.

        Parameters
        ----------
        suitability_threshold : float, optional
            The threshold above which to consider the temperature values suitable.
            Default is 0.

        Returns
        -------
        xarray.Dataset
            A dataset containing the posterior distributions of the minimum,
            peak, and maximum temperature values.
        """
        self._check_fitting()
        self._check_suitability_table()
        da_suitability_table = self.suitability_table[self._suitability_var_name]
        da_temperature = da_suitability_table.temperature
        if "precipitation" in da_suitability_table.dims:
            raise ValueError(
                "This method only works for models that depend on temperature only."
            )
        da_posterior_peak = (
            da_temperature.isel(
                temperature=da_suitability_table.argmax(dim="temperature")
            )
            .reset_coords(drop=True)
            .assign_attrs(long_name="Temperature of peak suitability", units="°C")
        )
        da_suitable = da_suitability_table > suitability_threshold
        da_posterior_min = (
            da_temperature.isel(temperature=da_suitable.argmax(dim="temperature"))
            .reset_coords(drop=True)
            .assign_attrs(long_name="Minimum suitable temperature", units="°C")
        )
        da_posterior_max = (
            da_temperature.isel(temperature=slice(None, None, -1))
            .isel(
                temperature=da_suitable.isel(temperature=slice(None, None, -1)).argmax(
                    dim="temperature"
                )
            )
            .reset_coords(drop=True)
            .assign_attrs(long_name="Maximum suitable temperature", units="°C")
        )
        ds_posterior_min_peak_max = xr.Dataset(
            {
                "temperature_min": da_posterior_min,
                "temperature_peak": da_posterior_peak,
                "temperature_max": da_posterior_max,
            }
        )
        return ds_posterior_min_peak_max

    def run(self, *args, **kwargs):
        """
        Run the epidemiological model on a given climate dataset.

        See the documentation for SuitabilityModel.run() for details.
        """
        self._check_fitting()
        self._check_suitability_table()
        return super().run(*args, **kwargs)

    def plot_suitability(self, **kwargs):
        """
        Plot suitability against temperature and (if relevant) precipitation.

        See the documentation for SuitabilityModel.plot_suitability() for details.
        """
        self._check_fitting()
        self._check_suitability_table()
        return super().plot_suitability(**kwargs)

    def get_max_suitability(self):
        """
        Get the maximum suitability value.

        See the documentation for SuitabilityModel.get_max_suitability() for
        details.
        """
        self._check_fitting()
        self._check_suitability_table()
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

    def _check_fitting(self):
        # Checks if the model has been fitted to data.
        if any(
            (isinstance(parameter_entry, dict)) and ("idata" not in parameter_entry)
            for parameter_entry in self._parameters.values()
        ):
            raise ValueError(
                "Need to fit the model with fit_temperature_responses() before running "
                "this method."
            )

    def _check_suitability_table(self):
        # Checks if the suitability table has been constructed.
        if self.suitability_table is None:
            raise ValueError(
                "Need to construct the suitability table with "
                "construct_suitability_table() before running this method."
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
        The type of curve to fit. Options are 'quadratic' (response =
        a*(T-T_min)*(T-T_max) for T_min < T < T_max, where T is temperature, and zero
        otherwise) and 'briere' (response = a*T*(T-T_min)*sqrt(T_max-T) for
        T_min < T < T_max, where T is temperature, and zero otherwise). In both cases,
        a is the scale parameter, T_min is the minimum temperature, and T_max is
        the maximum temperature, and normally distributed noise is assumed on the
        response.
    probability : bool, optional
        If True, the fitted curve is constrained to be between 0 and 1. Default is
        False.
    priors : dict, optional
        Dictionary of priors for the parameters of the model. The keys should be the
        parameter names ('scale', 'temperature_min', 'temperature_max', and either
        'noise_std' or 'noise_precision' - see description for 'curve_type' above) and
        the values should callable functions that return pymc distributions. Where not
        specified, default priors are used based on the curve type (as used by Mordecai
        et al., PLoS Negl Trop Dis 2017).
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
        "scale": (lambda: pm.Gamma("scale", alpha=1, beta=1))
        if curve_type == "quadratic"
        else (lambda: pm.Gamma("scale", alpha=1, beta=10))
        if curve_type == "briere"
        else (lambda: None),
        "temperature_min": lambda: pm.Uniform("temperature_min", lower=0, upper=24),
        "temperature_max": lambda: pm.Uniform("temperature_max", lower=25, upper=50),
        "noise_precision": lambda: pm.Gamma(
            "noise_precision", alpha=0.0001, beta=0.0001
        ),
        **priors,
    }
    # temperature_min_obs = np.min(temperature_data[trait_data > 0])
    # temperature_max_obs = np.max(temperature_data[trait_data > 0])
    with pm.Model():
        scale = priors["scale"]()
        temperature_min = priors["temperature_min"]()
        temperature_max = priors["temperature_max"]()
        mu = curve_func(
            temperature_data,
            scale=scale,
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
        # constraint = pm.math.and_(
        #     pm.math.ge(temperature_min_obs, temperature_min),
        #     pm.math.le(temperature_max_obs, temperature_max),
        # )
        # pm.Potential(
        #     "temp_bounds_constraint", pm.math.log(pm.math.switch(constraint, 1, 0))
        # )
        # kwargs_likelihood = {"lower": 0, "observed": trait_data}
        # # if probability:
        # #     kwargs_likelihood["upper"] = 1
        # likelihood = pm.Truncated(  # noqa
        #     "likelihood",
        #     pm.Normal.dist(mu=mu, **kwargs_normal),
        #     **kwargs_likelihood,
        # )
        likelihood = pm.Normal(  # noqa
            "likelihood",
            mu=mu,
            observed=trait_data,
            **kwargs_normal,
        )
        # Sample from the posterior distribution using the DEMetropolisZ algorithm
        # (NUTS not suitable given non-differentiable likelihood?)
        kwargs_sample = {
            "step": pm.DEMetropolisZ() if step is None else step(),
            # "initvals": {
            #     "temperature_min": temperature_min_obs - 1,
            #     "temperature_max": temperature_max_obs + 1,
            # },
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
        var_names=["scale", "temperature_min", "temperature_max"],
        num_samples=num_samples,
    )
    ds_posterior = ds_posterior.drop_vars(["sample", "chain", "draw"]).assign_coords(
        sample=np.arange(ds_posterior.sample.size)
    )
    if temperature_vals is None:
        temperature_vals = np.linspace(
            ds_posterior.temperature_min.min().item() - 1,
            ds_posterior.temperature_max.max().item() + 1,
            500,
        )
    ds_posterior_expanded = ds_posterior.assign_coords(temperature=temperature_vals)
    da_posterior_response = curve_func(
        temperature=ds_posterior_expanded.temperature,
        scale=ds_posterior_expanded.scale,
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
    scale=None,
    temperature_min=None,
    temperature_max=None,
    probability=None,
    array_lib=np,
):
    response = scale * (temperature - temperature_min) * (temperature_max - temperature)
    response = array_lib.where(temperature >= temperature_min, response, 0)
    response = array_lib.where(temperature <= temperature_max, response, 0)
    if probability:
        response = response.clip(0, 1)
    return response


def _briere(
    temperature,
    scale=None,
    temperature_min=None,
    temperature_max=None,
    probability=None,
    array_lib=np,
):
    response = (
        scale
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
