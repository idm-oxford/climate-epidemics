"""
Base module of the epimod subpackage.

Provides a base epidemiological model class and a subclass for temperature- and/or
rainfall-dependent suitability models.
"""

from typing import Any, Literal

import numpy as np
import param
import xarray as xr
from numpy.typing import ArrayLike

from climepi.utils import add_bnds_from_other


class EpiModel:
    """
    Base class for epidemiological models. Intended to be subclassed.

    Subclasses must implement the ``run()`` method to run the main logic of the
    epidemiological model on a given climate dataset.
    """

    def __init__(self):
        pass

    def run(self, ds_clim: xr.Dataset) -> xr.Dataset:
        """
        Run the epidemiological model on a given climate dataset.

        Should be implemented by subclasses.

        Parameters
        ----------
        ds_clim : xarray.Dataset
            The input climate dataset.

        Returns
        -------
        xarray.Dataset
            The output epidemiological dataset.
        """
        raise NotImplementedError(
            "The run method should be implemented by subclasses of the base EpiModel "
            "class."
        )


class SuitabilityModel(EpiModel):
    """
    Generic class for suitability models.

    Attributes
    ----------
    temperature_range : list or tuple of two floats, optional
        A list or tuple of two floats defining the temperature range of suitability
        (in degrees Celsius). Only defined if the parameter ``temperature_range`` is
        provided.
    suitability_table : xarray.Dataset
        A dataset containing suitability values defined for different temperature
        values or temperature/precipitation combinations. Only defined if the parameter
        ``suitability_table`` is provided.

    Parameters
    ----------
    temperature_range : list or tuple of two floats, optional
        A list or tuple of two floats defining the temperature range of suitability
        (in degrees Celsius), where suitability is assumed to be 1 for temperatures
        within the range and 0 otherwise. Default is ``None``. Only one of
        ``temperature_range`` and ``suitability_table`` should be provided.
    suitability_table : xarray.Dataset, optional
        A dataset containing suitability values defined for different temperature
        values or temperature/precipitation combinations. The dataset should have a
        single data variable (with any desired name) with dimension(s) 'temperature'
        and, optionally, 'precipitation'. Temperatures should be in degrees Celsius and
        precipitation values in mm/day. Equi-spaced temperature and (where applicable)
        precipitation values should be provided, and nearest neighbour interpolation is
        used to calculate suitability values away from grid points (this is for
        performance reasons). Suitability values can be either binary (0 or 1) or
        continuous. Suitability is assumed to take the nearest endpoint value for
        temperature and/or precipitation values outside the provided range(s). May also
        have an additional dimension named 'sample' indexing equally likely possible
        suitability tables, or a dimension 'suitability_quantile' indexing quantiles of
        the suitability values. Default is ``None``. Only one of ``temperature_range``
        and ``suitability_table`` should be provided.
    suitability_var_name : str, optional
        The name of the suitability variable. Should not be provided if
        ``suitability_table`` is provided (in this case, the name of the single
        data variable in the suitability table will be used instead). If
        ``suitability_table`` is not provided, the name will default to 'suitability'.
    suitability_var_long_name : str, optional
        The long name of the suitability variable, used to label axes in plots. If not
        provided and if ``suitability_table`` is provided, the long name will be taken
        from the suitability table (either via the 'long_name' attribute of the
        suitability variable, if it exists, or otherwise by capitalizing the suitability
        variable name). If ``suitability_table`` is not provided, the long name will
        default to 'Suitability'.
    """

    def __init__(
        self,
        temperature_range: tuple[float, float] | None = None,
        suitability_table: xr.Dataset | None = None,
        suitability_var_name: str | None = None,
        suitability_var_long_name: str | None = None,
    ) -> None:
        super().__init__()
        if suitability_table is not None:
            if temperature_range is not None:
                raise ValueError(
                    "The temperature_range argument should not be provided if the "
                    "suitability_table argument is provided."
                )
            if len(suitability_table.data_vars) != 1:
                raise ValueError(
                    "The suitability table should only have a single data variable."
                )
            if suitability_var_name is not None:
                raise ValueError(
                    "The suitability_var_name argument should not be provided if the "
                    "suitability_table argument is provided (the name of the single "
                    "data variable in the suitability table will be used instead)."
                )
            suitability_var_name = list(suitability_table.data_vars)[0]
            suitability_var_long_name = suitability_var_long_name or suitability_table[
                suitability_var_name
            ].attrs.get(
                "long_name", suitability_var_name.capitalize().replace("_", " ")
            )
            suitability_table = suitability_table.assign(
                {
                    suitability_var_name: suitability_table[
                        suitability_var_name
                    ].assign_attrs(long_name=suitability_var_long_name)
                }
            )
        self.temperature_range = temperature_range
        self.suitability_table = suitability_table
        self._suitability_var_name = suitability_var_name or "suitability"
        self._suitability_var_long_name = suitability_var_long_name or "Suitability"

    def run(
        self,
        ds_clim: xr.Dataset,
        return_yearly_portion_suitable: bool = False,
        suitability_threshold: float = 0,
    ) -> xr.Dataset:
        """
        Run the epidemiological model on a given climate dataset.

        Extends the parent method to include the option to return the number of
        days/months suitable each year (depending on the resolution of the climate
        data), rather than the full suitability dataset.

        Parameters
        ----------
        ds_clim : xarray.Dataset
            The input climate dataset.
        return_yearly_portion_suitable : bool, optional
            Whether to return the number of days/months suitable each year (depending on
            the resolution of the climate data), rather than the full suitability
            dataset. Default is ``False``.
        suitability_threshold : float, optional
            The minimum suitability threshold for a day/month to be considered suitable.
            Only used if ``return_yearly_portion_suitable`` is ``True``. Default is 0.

        Returns
        -------
        xarray.Dataset
            The output epidemiological dataset.
        """
        suitability_var_name = self._suitability_var_name
        if self.suitability_table is None:
            da_suitability = self._run_main_temp_range(ds_clim)
        elif "precipitation" not in self.suitability_table.dims:
            da_suitability = self._run_main_temp_table(ds_clim)
        else:
            da_suitability = self._run_main_temp_precip_table(ds_clim)
        ds_epi = xr.Dataset(attrs=ds_clim.attrs)
        ds_epi[suitability_var_name] = da_suitability
        if self.suitability_table is not None:
            ds_epi[suitability_var_name].attrs = self.suitability_table[
                suitability_var_name
            ].attrs
        if "long_name" not in ds_epi[suitability_var_name].attrs:
            ds_epi[suitability_var_name].attrs["long_name"] = (
                self._suitability_var_long_name
            )
        ds_epi = add_bnds_from_other(ds_epi, ds_clim)
        if return_yearly_portion_suitable:
            ds_epi = ds_epi.climepi.yearly_portion_suitable(
                suitability_threshold=suitability_threshold
            )
        return ds_epi

    def plot_suitability(self, **kwargs: Any) -> param.Parameterized:
        """
        Plot suitability against temperature and (if relevant) precipitation.

        Parameters
        ----------
        **kwargs: dict, optional
            Additional keyword arguments to pass to the plotting function
            (:meth:`hvplot.hvPlot.line()` for temperature-only suitability, or
            :meth:`hvplot.hvPlot.image()` for temperature- and
            precipitation-dependent suitability).

        Returns
        -------
        :mod:`holoviews` object
            The suitability plot.
        """
        suitability_table = self.suitability_table
        suitability_var_name = self._suitability_var_name
        if suitability_table is None:
            temperature_range = self.temperature_range
            assert temperature_range is not None
            temperature_vals = np.linspace(0, 1.25 * temperature_range[1], 1000)
            suitability_vals = (
                (temperature_vals >= temperature_range[0])
                & (temperature_vals <= temperature_range[1])
            ).astype(int)
            suitability_table = xr.Dataset(
                {
                    "temperature": temperature_vals,
                    suitability_var_name: (["temperature"], suitability_vals),
                },
            )
            suitability_table[suitability_var_name].attrs = {
                "long_name": self._suitability_var_long_name
            }
            suitability_table["temperature"].attrs = {
                "long_name": "Temperature",
                "units": "°C",
            }
        if suitability_table[suitability_var_name].dtype == bool:
            suitability_table = suitability_table.astype(int)
        if "precipitation" not in suitability_table.dims:
            kwargs_hvplot = {"x": "temperature", **kwargs}
            return suitability_table[suitability_var_name].hvplot.line(**kwargs_hvplot)
        kwargs_hvplot = {
            "x": "temperature",
            "y": "precipitation",
            **kwargs,
        }
        return suitability_table[suitability_var_name].hvplot.image(**kwargs_hvplot)

    def get_max_suitability(self) -> float:
        """
        Return the maximum suitability value.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The maximum suitability value.
        """
        if self.suitability_table is None:
            return 1
        return self.suitability_table[self._suitability_var_name].max().item()

    def _run_main_temp_range(self, ds_clim: xr.Dataset) -> xr.DataArray:
        # Run the main logic of a suitability model defined by a temperature range.
        temperature = ds_clim["temperature"]
        temperature_range = self.temperature_range
        assert temperature_range is not None
        da_suitability = (temperature >= temperature_range[0]) & (
            temperature <= temperature_range[1]
        )
        return da_suitability

    def _run_main_temp_table(self, ds_clim: xr.Dataset) -> xr.DataArray:
        # Run the main logic of a suitability model defined by a temperature suitability
        # table.
        temperature = ds_clim["temperature"]
        suitability_var_name = self._suitability_var_name
        assert self.suitability_table is not None
        da_suitability_table = self.suitability_table[suitability_var_name].transpose(
            "temperature", ...
        )
        table_suitability_vals = da_suitability_table.values
        table_temp_vals = da_suitability_table["temperature"].values
        table_temp_deltas = np.diff(table_temp_vals)
        if not np.all(
            np.isclose(table_temp_deltas, table_temp_deltas[0], rtol=1e-3, atol=0)
        ):
            raise ValueError(
                "The suitability table must be defined on a regular grid of "
                "temperature values.",
            )

        table_temp_delta = table_temp_deltas[0]

        temp_inds = (temperature - table_temp_vals[0]) / table_temp_delta
        temp_inds = temp_inds.round(0).astype(int).clip(0, len(table_temp_vals) - 1)

        def suitability_func(temp_inds_curr: np.ndarray) -> np.ndarray:
            suitability_curr = table_suitability_vals[temp_inds_curr, ...]
            return suitability_curr

        output_core_dims = [list(da_suitability_table.dims)[1:]]
        output_sizes = {
            dim: size
            for dim, size in da_suitability_table.sizes.items()
            if dim != "temperature"
        }

        da_suitability = xr.apply_ufunc(
            suitability_func,
            temp_inds,
            dask="parallelized",
            output_core_dims=output_core_dims,
            dask_gufunc_kwargs={
                "output_sizes": output_sizes,
            },
        )

        if output_sizes:
            da_suitability = da_suitability.assign_coords(
                {dim: da_suitability_table[dim] for dim in output_sizes.keys()}
            )

        return da_suitability

    def _run_main_temp_precip_table(self, ds_clim: xr.Dataset) -> xr.DataArray:
        # Run the main logic of a suitability model defined by a temperature and
        # precipitation suitability table.
        temperature = ds_clim["temperature"]
        precipitation = ds_clim["precipitation"]
        suitability_var_name = self._suitability_var_name
        assert self.suitability_table is not None
        da_suitability_table = self.suitability_table[suitability_var_name].transpose(
            "temperature", "precipitation", ...
        )
        table_suitability_vals = da_suitability_table.values
        table_temp_vals = da_suitability_table["temperature"].values
        table_temp_deltas = np.diff(table_temp_vals)
        table_precip_vals = da_suitability_table["precipitation"].values
        table_precip_deltas = np.diff(table_precip_vals)
        if not np.all(
            np.isclose(table_temp_deltas, table_temp_deltas[0], rtol=1e-3, atol=0)
        ) or not np.all(
            np.isclose(table_precip_deltas, table_precip_deltas[0], rtol=1e-3, atol=0)
        ):
            raise ValueError(
                "The suitability table must be defined on a regular grid of ",
                "temperature and precipitation values.",
            )
        table_temp_delta = table_temp_deltas[0]
        table_precip_delta = table_precip_deltas[0]

        temp_inds = (temperature - table_temp_vals[0]) / table_temp_delta
        temp_inds = temp_inds.round(0).astype(int).clip(0, len(table_temp_vals) - 1)
        precip_inds = (precipitation - table_precip_vals[0]) / table_precip_delta
        precip_inds = (
            precip_inds.round(0).astype(int).clip(0, len(table_precip_vals) - 1)
        )

        def suitability_func(
            temp_inds_curr: np.ndarray, precip_inds_curr: np.ndarray
        ) -> np.ndarray:
            suitability_curr = table_suitability_vals[
                temp_inds_curr, precip_inds_curr, ...
            ]
            return suitability_curr

        output_core_dims = [list(da_suitability_table.dims)[2:]]
        output_sizes = {
            dim: size
            for dim, size in da_suitability_table.sizes.items()
            if dim not in ["temperature", "precipitation"]
        }

        da_suitability = xr.apply_ufunc(
            suitability_func,
            temp_inds,
            precip_inds,
            dask="parallelized",
            output_core_dims=output_core_dims,
            dask_gufunc_kwargs={
                "output_sizes": output_sizes,
            },
        )

        if output_sizes:
            da_suitability = da_suitability.assign_coords(
                {dim: da_suitability_table[dim] for dim in output_sizes.keys()}
            )

        return da_suitability

    def reduce(
        self,
        suitability_threshold: float | None = None,
        stat: Literal["mean", "median", "quantile"] | None = None,
        quantile: ArrayLike | None = None,
        rescale: bool | Literal["mean", "median"] | None = False,
    ) -> "SuitabilityModel":
        """
        Get a summary suitability model.

        Applies a summary statistic over the equally likely suitability tables and/or
        calculates a binary suitability model based on a threshold value. If both a
        suitability threshold and a summary statistic are provided, the suitability
        threshold is applied first, and then the statistic.

        Parameters
        ----------
        suitability_threshold : float, optional
            The threshold value (strictly) above which climate conditions are considered
            suitable in a binary suitability model. If ``None``, a binary suitability
            a binary suitability model is not computed. Default is ``None``.
        stat : str, optional
            The summary statistic to compute. Can be 'mean', 'median', or 'quantile'.
            If ``None``, no summary statistic is computed. Default is ``None``.
        quantile : array-like of floats, optional
            The quantile(s) to compute if ``stat`` is 'quantile'. Default is ``None``.
        rescale : bool or str, optional
            If ``True``, the suitability values (after applying any summary statistics)
            are rescaled so that the maximum value is one. Can also be set to 'mean'
            or 'median' such that the mean/median suitability table has max value 1.
            Has no effect if ``suitability_threshold`` is specified. Default is
            ``False``.

        Returns
        -------
        SuitabilityModel
            The summary suitability model.
        """
        if self.suitability_table is None:
            raise ValueError(
                "The reduce() method is only available for suitability table-based "
                "models."
            )
        suitability_var_name = self._suitability_var_name
        da_suitability = self.suitability_table[suitability_var_name]
        if suitability_threshold is not None:
            da_suitability = da_suitability > suitability_threshold
        if da_suitability.dtype == bool and stat in ["median", "quantile"]:
            binary_suitability = True
            da_suitability = da_suitability.astype(int)
        else:
            binary_suitability = False
        suitability_stat_dict: dict[str, xr.DataArray | None] = {
            "mean": None,
            "median": None,
            "quantile": None,
        }
        if stat == "mean" or rescale == "mean":
            suitability_stat_dict["mean"] = da_suitability.mean(dim="sample")
        if stat == "median" or rescale == "median":
            suitability_stat_dict["median"] = da_suitability.median(dim="sample")
        if stat == "quantile":
            if quantile is None:
                raise ValueError("quantile must be specified if stat is 'quantile'")
            suitability_stat_dict["quantile"] = da_suitability.quantile(
                q=quantile, dim="sample"
            ).rename({"quantile": "suitability_quantile"})
        if stat is not None:
            if stat not in ["mean", "median", "quantile"]:
                raise ValueError(
                    "stat must be one of 'mean', 'median', or 'quantile' if specified."
                )
            da_stat = suitability_stat_dict.get(stat, None)
            assert da_stat is not None
            da_suitability = da_stat
        if binary_suitability:
            da_suitability = da_suitability.astype(bool)
        if rescale:
            if isinstance(rescale, str):
                if rescale not in ["mean", "median"]:
                    raise ValueError(
                        "rescale must be one of 'mean' or 'median' if a string is "
                        "provided."
                    )
                da_scale_by = suitability_stat_dict.get(rescale, None)
                assert da_scale_by is not None
                da_suitability = da_suitability / da_scale_by.max()
            else:
                da_suitability = da_suitability / da_suitability.max()
        suitability_table_new = da_suitability.to_dataset(name=suitability_var_name)
        if (
            da_suitability.dtype == bool
            and len(suitability_table_new.dims) == 1
            and "temperature" in suitability_table_new.dims
        ):
            temperature = suitability_table_new.temperature
            suitable_temperatures = temperature.where(
                suitability_table_new[suitability_var_name], drop=True
            )
            first_suitable_table_temp = suitable_temperatures.item(0)
            last_suitable_table_temp = suitable_temperatures.item(-1)
            first_suitable_loc = temperature.get_index("temperature").get_loc(
                first_suitable_table_temp
            )
            last_suitable_loc = temperature.get_index("temperature").get_loc(
                last_suitable_table_temp
            )
            if (
                first_suitable_loc
                and first_suitable_loc > 0
                and last_suitable_loc < len(temperature) - 1
                and len(suitable_temperatures)
                == last_suitable_loc - first_suitable_loc + 1
            ):
                min_temp = (  # interpolates between first suitable temperature and the
                    # temperature just below it, consistent with nearest neighbour
                    # interpolation used in _run_main_temp_table
                    0.5
                    * (
                        temperature.item(first_suitable_loc - 1)
                        + first_suitable_table_temp
                    )
                )
                max_temp = 0.5 * (
                    last_suitable_table_temp + temperature.item(last_suitable_loc + 1)
                )
                return SuitabilityModel(
                    temperature_range=(min_temp, max_temp),
                )
        return SuitabilityModel(suitability_table=suitability_table_new)
