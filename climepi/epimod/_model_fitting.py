import copy
import numbers
from typing import Any, Callable, Literal, Union, cast

import arviz as az
import holoviews as hv
import hvplot.xarray  # noqa
import numpy as np
import pandas as pd
import param
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from numpy.typing import ArrayLike, NDArray

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
    parameters : dict
        Dictionary of model parameters. Each key is a parameter name, and the value
        is either a number (constant parameter), a callable (function which takes
        keyword arguments 'temperature' and, if the model is dependent on precipitation,
        'precipitation', which should be able to handle xarray DataArrays as inputs),
        or, for temperature-dependent parameters that are to be fitted, a dictionary
        with the following keys:

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

    suitability_function : callable
        A callable that takes the model parameters as keyword arguments and returns a
        suitability metric (e.g., the basic reproduction number). The callable should
        be able to handle xarray DataArrays as inputs.
    data : pandas.DataFrame, optional
        A DataFrame containing the temperature and trait data for the parameters to be
        fitted. The DataFrame should have columns "trait_name", "temperature", and
        "trait_value".
    suitability_var_name : str, optional
        The name of the suitability variable. Default is "suitability".
    suitability_var_long_name : str, optional
        The long name of the suitability variable. Default is "Suitability".
    """

    def __init__(
        self,
        *,
        parameters: dict[str, Any],
        suitability_function: Callable,
        data: pd.DataFrame | None = None,
        suitability_var_name: str = "suitability",
        suitability_var_long_name: str = "Suitability",
    ):
        super().__init__(
            suitability_var_name=suitability_var_name,
            suitability_var_long_name=suitability_var_long_name,
        )
        self._parameters = parameters
        self._suitability_function = suitability_function
        if data is not None:
            self._extract_data(data)

    def fit_temperature_responses(
        self, step: Callable | None = None, thin: int = 1, **kwargs_sample: Any
    ) -> dict[str, az.InferenceData]:
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

        Returns
        -------
        dict
            A dictionary with fitted trait names as keys, and arviz.InferenceData
            objects giving posterior distributions of response curve parameters for
            that trait as corresponding values.
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
            idata_dict[parameter_name] = idata.copy()
        self._parameters = parameters
        return idata_dict

    def plot_fitted_temperature_responses(
        self,
        parameter_names: str | list[str] | None = None,
        temperature_vals: ArrayLike | None = None,
        kwargs_scatter: dict[str, Any] | None = None,
        kwargs_area: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> hv.Layout:
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
        kwargs_scatter : dict, optional
            Keyword arguments to pass to hvplot.scatter() when plotting the response
            data.
        kwargs_area : dict, optional
            Keyword arguments to pass to hvplot.area() when plotting the credible
            intervals.
        **kwargs : dict, optional
            Additional keyword arguments to pass to hvplot.line().

        Returns
        -------
        holoviews.Layout
            A holoviews Layout object containing the plots of the fitted temperature
            responses for the specified parameters.
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
                    kwargs_scatter=kwargs_scatter,
                    kwargs_area=kwargs_area,
                    **kwargs,
                )
            )
        return hv.Layout(plots).opts(shared_axes=False)

    def construct_suitability_table(
        self,
        *,
        temperature_vals: ArrayLike,
        precipitation_vals: ArrayLike | None = None,
        num_samples: int | None = None,
    ) -> xr.Dataset:
        """
        Construct a suitability table based on the fitted parameters.

        Note that this method requires that the model has been fitted to data
        using fit_temperature_responses() before it can be called. The suitability
        table is retained as an attribute of the ParameterizedSuitabilityModel instance
        (which can be accessed via the `suitability_table` attribute), and a copy is
        also returned.

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

        Returns
        -------
        xarray.Dataset
            A dataset containing the suitability values for the specified temperature
            and precipitation values, and for each posterior sample.
        """
        self._check_fitting()
        parameter_vals: dict[str, xr.DataArray | numbers.Number] = {}
        for parameter_name, parameter_entry in self._parameters.items():
            if isinstance(parameter_entry, dict):
                idata = parameter_entry.get("idata")
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
        self.suitability_table = suitability_table
        return suitability_table.copy()

    def get_posterior_min_optimal_max_temperature(
        self, suitability_threshold: float = 0
    ) -> xr.Dataset:
        """
        Get posterior distributions of minimum, optimal, and maximum temperatures.

        Calculates the posterior distributions of the minimum/maximum temperatures that
        are considered suitable, as well as the optimal temperature at which the
        suitability is at its peak.

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
            optimal, and maximum temperature values.
        """
        self._check_fitting()
        self._check_suitability_table()
        assert self.suitability_table is not None
        da_suitability_table = self.suitability_table[self._suitability_var_name]
        da_temperature = da_suitability_table.temperature
        if "precipitation" in da_suitability_table.dims:
            raise ValueError(
                "This method only works for models that depend on temperature only."
            )
        da_posterior_optimal = (
            da_temperature.isel(
                temperature=da_suitability_table.argmax(dim="temperature")
            )
            .reset_coords(drop=True)
            .assign_attrs(long_name="Optimal temperature for suitability", units="°C")
        )
        da_suitable = da_suitability_table > suitability_threshold
        first_suitable_idx = cast(xr.DataArray, da_suitable.argmax(dim="temperature"))
        last_suitable_idx = (
            da_suitable.sizes["temperature"]
            - 1
            - cast(
                xr.DataArray,
                da_suitable.isel(temperature=slice(None, None, -1)).argmax(
                    dim="temperature"
                ),
            )
        )
        if np.any(first_suitable_idx == 0) or np.any(
            last_suitable_idx == da_suitable.sizes["temperature"] - 1
        ):
            raise ValueError(
                "Minimum and/or maximum suitable temperatures do not exist."
            )
        da_posterior_min = (
            0.5
            * (
                da_temperature.isel(temperature=first_suitable_idx - 1).reset_coords(
                    drop=True
                )
                + da_temperature.isel(temperature=first_suitable_idx).reset_coords(
                    drop=True
                )
            )
        ).assign_attrs(long_name="Minimum suitable temperature", units="°C")
        da_posterior_max = (
            0.5
            * (
                da_temperature.isel(temperature=last_suitable_idx).reset_coords(
                    drop=True
                )
                + da_temperature.isel(temperature=last_suitable_idx + 1).reset_coords(
                    drop=True
                )
            )
        ).assign_attrs(long_name="Maximum suitable temperature", units="°C")
        ds_posterior_min_optimal_max = xr.Dataset(
            {
                "temperature_min": da_posterior_min,
                "temperature_optimal": da_posterior_optimal,
                "temperature_max": da_posterior_max,
            }
        )
        return ds_posterior_min_optimal_max

    def run(self, *args: Any, **kwargs: Any) -> xr.Dataset:
        """
        Run the epidemiological model on a given climate dataset.

        See the documentation for SuitabilityModel.run() for details.
        """
        self._check_fitting()
        self._check_suitability_table()
        return super().run(*args, **kwargs)

    def plot_suitability(self, **kwargs: Any) -> param.Parameterized:
        """
        Plot suitability against temperature and (if relevant) precipitation.

        See the documentation for SuitabilityModel.plot_suitability() for details.
        """
        self._check_fitting()
        self._check_suitability_table()
        return super().plot_suitability(**kwargs)

    def get_max_suitability(self) -> float:
        """
        Get the maximum suitability value.

        See the documentation for SuitabilityModel.get_max_suitability() for
        details.
        """
        self._check_fitting()
        self._check_suitability_table()
        return super().get_max_suitability()

    def _extract_data(self, data: pd.DataFrame):
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
    *,
    temperature_data: ArrayLike,
    trait_data: ArrayLike,
    curve_type: str,
    probability: bool = False,
    priors: dict[str, Callable] | None = None,
    step: Callable | None = None,
    thin: int = 1,
    **kwargs_sample: Any,
) -> az.InferenceData:
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
    arviz.InferenceData
        The posterior distribution of the fitted parameters.
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
            backend="pytensor",
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
    *,
    curve_type: str,
    num_samples: int | None = None,
    temperature_vals: ArrayLike | None = None,
    probability: bool = False,
    trait_name: str | None = None,
    trait_attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """
    Get the posterior distribution of the fitted temperature response.

    Parameters
    ----------
    idata : arviz.InferenceData
        The posterior distribution of the fitted parameters (as returned by
        fit_temperature_response()).
    curve_type : str
        The type of curve fitted. Options are 'quadratic' and 'briere'.
    num_samples : int, optional
        Number of samples to draw from the posterior distribution. If None, all samples
        are used.
    temperature_vals : array-like, optional
        Vector of temperature values for which the response is to be computed. If not
        provided, a default range is generated based on the minimum and maximum
        temperature values in the posterior distribution.
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
    da_temperature = xr.DataArray(
        temperature_vals,
        dims=["temperature"],
        coords={"temperature": temperature_vals},
    )
    da_posterior_response = curve_func(
        temperature=da_temperature,
        scale=ds_posterior.scale,
        temperature_min=ds_posterior.temperature_min,
        temperature_max=ds_posterior.temperature_max,
        probability=probability,
        backend="xarray",
    )
    da_posterior_response.name = trait_name if trait_name is not None else "response"
    if trait_attrs is not None:
        da_posterior_response.attrs.update(trait_attrs)
    da_posterior_response.temperature.attrs["long_name"] = "Temperature"
    da_posterior_response.temperature.attrs["units"] = "°C"
    return da_posterior_response


def plot_fitted_temperature_response(
    idata,
    *,
    curve_type: str,
    temperature_vals: ArrayLike | None = None,
    temperature_data: ArrayLike | None = None,
    trait_data: ArrayLike | None = None,
    probability: bool = False,
    trait_name: str | None = None,
    trait_attrs: dict[str, Any] | None = None,
    kwargs_scatter: dict[str, Any] | None = None,
    kwargs_area: dict[str, Any] | None = None,
    **kwargs: Any,
) -> hv.Overlay:
    """
    Plot a fitted temperature response curve.

    The median response is plotted along with the 95% credible interval.

    Parameters
    ----------
    idata : arviz.InferenceData
        The posterior distribution of the fitted parameters.
    curve_type : str
        The type of curve fitted. Options are 'quadratic' and 'briere'.
    temperature_vals : array-like, optional
        Vector of temperature values for which the response is to be plotted. If not
        provided, a default range is generated based on the minimum and maximum
        temperature values in the posterior distribution.
    temperature_data : array-like, optional
        Vector of temperature values for which response data are available.
    trait_data : array-like, optional
        Vector of values of the trait variable for the corresponding temperature values.
    probability : bool, optional
        If True, the response is constrained to be between 0 and 1. Default is False.
    trait_name : str, optional
        The name of the trait variable.
    trait_attrs : dict, optional
        Additional attributes to assign to the trait variable in the plotted dataset.
    kwargs_scatter : dict, optional
        Keyword arguments to pass to hvplot.scatter() when plotting the response data.
    kwargs_area : dict, optional
        Keyword arguments to pass to hvplot.area() when plotting the credible interval.
    **kwargs : dict, optional
        Additional keyword arguments to pass to hvplot.line().

    Returns
    -------
    holoviews.Overlay
        The plot object containing the fitted temperature response curve.
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
    p = da_response_quantiles.sel(quantile="median", drop=True).hvplot.line(
        label="Median response", **kwargs
    ) * da_response_quantiles.to_dataset(dim="quantile").hvplot.area(
        **{
            "y": "lower",
            "y2": "upper",
            "alpha": 0.2,
            "label": "95% credible interval",
            **(kwargs_area or {}),
        },
    )
    if trait_data is not None or temperature_data is not None:
        if trait_data is None or temperature_data is None:
            raise ValueError(
                "Either both, or neither, trait data and corresponding temperature data "
                "should be provided."
            )
        p = p * xr.Dataset(
            {"trait": ("temperature", trait_data)},
            coords={"temperature": temperature_data},
        ).hvplot.scatter(**(kwargs_scatter or {}))
    return p


_SupportedArrayType = Union[
    NDArray[np.floating], xr.DataArray, xr.Dataset, pt.TensorVariable
]


def _bounded_quadratic(
    temperature: _SupportedArrayType,
    *,
    scale: float,
    temperature_min: float,
    temperature_max: float,
    probability: bool = False,
    backend: Literal["numpy", "xarray", "pytensor"] = "numpy",
) -> _SupportedArrayType:
    response = scale * (temperature - temperature_min) * (temperature_max - temperature)
    response = _where(temperature >= temperature_min, response, 0, backend=backend)
    response = _where(temperature <= temperature_max, response, 0, backend=backend)
    if probability:
        response = response.clip(0, 1)
    return response


def _briere(
    temperature: _SupportedArrayType,
    *,
    scale: float,
    temperature_min: float,
    temperature_max: float,
    probability: bool = False,
    backend: Literal["numpy", "xarray", "pytensor"] = "numpy",
) -> _SupportedArrayType:
    response = (
        scale
        * temperature
        * (temperature - temperature_min)
        * np.abs(temperature_max - temperature) ** 0.5
    )
    response = _where(temperature >= temperature_min, response, 0, backend=backend)
    response = _where(temperature <= temperature_max, response, 0, backend=backend)
    if probability:
        response = response.clip(0, 1)
    return response


def _where(
    condition: Any,
    x: Any,
    y: Any,
    backend: Literal["numpy", "xarray", "pytensor"] = "numpy",
) -> _SupportedArrayType:
    if backend == "numpy":
        result = np.where(condition, x, y)
    elif backend == "xarray":
        result = xr.where(condition, x, y)
    elif backend == "pytensor":
        result = pt.where(condition, x, y)
    else:
        raise ValueError(f"Invalid backend: {backend}")
    return cast(_SupportedArrayType, result)


def _get_curve_func(curve_type: str) -> Callable:
    # Returns the appropriate curve function based on the curve type.
    if curve_type == "briere":
        return _briere
    if curve_type == "quadratic":
        return _bounded_quadratic
    raise ValueError(
        f"Invalid curve_type: '{curve_type}'. Must be 'briere' or 'quadratic'."
    )
