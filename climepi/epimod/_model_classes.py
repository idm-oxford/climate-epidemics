"""Base module of the epimod subpackage providing a base epidemiological model
class, a subclass for temperature- and/or rainfall-dependent suitability models, and an
xarray accessor class.
"""

import numpy as np
import xarray as xr

from climepi.utils import add_bnds_from_other


class EpiModel:
    """
    Base class for epidemiological models. Intended to be subclassed.

    Subclasses must implement the _run_main method to run the main logic of the
    epidemiological model on a given climate dataset.
    """

    def __init__(self):
        pass

    def run(self, ds_clim):
        """
        Runs the epidemiological model on a given climate dataset. Should be implemented
        by subclasses.

        Parameters:
        -----------
        ds_clim : xarray.Dataset
            The input climate dataset.

        Returns:
        --------
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

    Attributes:
    -----------
    temperature_range : list or tuple or array-like of two floats, optional
        A list or tuple of two floats defining the temperature range of suitability
        (in degrees Celsius), where suitability is assumed to be 1 for temperatures
        within the range and 0 otherwise. Default is None. Not used if
        `suitability_table` is not None.
    suitability_table : xarray.Dataset
        A dataset containing suitability values defined for different temperature
        values or temperature/precipitation combinations. The dataset should have a
        single data variable called "suitability" with dimension(s) "temperature" and,
        optionally, "precipitation". Temperatures should be in degrees Celsius and
        precipitation values in mm/day. If suitability only depends on temperature,
        linear interpolation is used to calculate suitability values away from grid
        points. If suitability only depends on both tempperature and precipitation,
        equi-spaced temperature and precipitation values should be provided (this is not
        required if suitability only depends on temperature), and nearest neighbour
        interpolation is used to calculate suitability values away from grid points
        (this is for performance reasons). Suitability values can be either binary (0
        or 1) or continuous. Suitability is assumed to be 0 outside of the grid.
        Default is None.

    Parameters:
    -----------
    temperature_range : list or tuple or array-like of two floats, optional
        Value for the `temperature_range` attribute. Default is None. Not used if
        `suitability_table` is provided.
    suitability_table : xarray.Dataset, optional
        Value for the `suitability_table` attribute. Default is None.
    """

    def __init__(self, temperature_range=None, suitability_table=None):
        super().__init__()
        self.temperature_range = temperature_range
        self.suitability_table = suitability_table
        if suitability_table is None:
            self._suitability_var_name = "suitability"
            self._suitability_var_long_name = "Suitability"
        else:
            assert (
                len(suitability_table.data_vars) == 1
            ), "The suitability table should only have a single data variable."
            self._suitability_var_name = list(suitability_table.data_vars)[0]
            self._suitability_var_long_name = suitability_table[
                self._suitability_var_name
            ].attrs.get("long_name", self._suitability_var_name.capitalize())
            self.suitability_table[self._suitability_var_name].attrs.update(
                {"long_name": self._suitability_var_long_name}
            )

    def run(self, ds_clim, return_months_suitable=False, suitability_threshold=0):
        """
        Runs the epidemiological model on a given climate dataset. Extends the parent
        method to include the option to return the number of months suitable each year,
        rather than the full suitability dataset.

        Parameters:
        -----------
        ds_clim : xarray.Dataset
            The input climate dataset.
        return_months_suitable : bool, optional
            Whether to return the number of months suitable each year, rather than the
            full suitability dataset. Default is False.
        suitability_threshold : float, optional
            The minimum suitability threshold for a month to be considered suitable.
            Only used if `return_months_suitable` is True. Default is 0.

        Returns:
        --------
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
        if return_months_suitable:
            ds_epi = ds_epi.epimod.months_suitable(
                suitability_threshold=suitability_threshold
            )
        return ds_epi

    def plot_suitability_region(self, **kwargs):
        """
        Plot suitability against temperature and (if relevant) precipitation.

        Parameters:
        -----------
        **kwargs: dict, optional
            Additional keyword arguments to pass to the plotting function (hvplot.line
            for temperature-only suitability, or hvplot.image for temperature-
            precipitation suitability).

        Returns:
        --------
        hvplot object
            A holoviews object representing the ecological niche.
        """
        suitability_table = self.suitability_table
        suitability_var_name = self._suitability_var_name
        if suitability_table is None:
            temperature_range = self.temperature_range
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
        if "precipitation" not in suitability_table.dims:
            kwargs_hvplot = {"x": "temperature", **kwargs}
            return suitability_table[suitability_var_name].hvplot.line(**kwargs_hvplot)
        kwargs_hvplot = {"x": "temperature", "y": "precipitation", **kwargs}
        return suitability_table[suitability_var_name].hvplot.image(**kwargs_hvplot)

    def get_max_suitability(self):
        """
        Returns the maximum suitability value.

        Parameters:
        -----------
        None

        Returns:
        --------
        float
            The maximum suitability value.
        """
        if self.suitability_table is None:
            return 1
        return self.suitability_table[self._suitability_var_name].max().item()

    def _run_main_temp_range(self, ds_clim):
        # Run the main logic of a suitability model defined by a temperature range.
        temperature = ds_clim["temperature"]
        temperature_range = self.temperature_range
        da_suitability = (temperature > temperature_range[0]) & (
            temperature < temperature_range[1]
        )
        return da_suitability

    def _run_main_temp_table(self, ds_clim):
        # Run the main logic of a suitability model defined by a temperature suitability
        # table.
        temperature = ds_clim["temperature"]
        suitability_table = self.suitability_table
        table_temp_vals = suitability_table["temperature"].values
        suitability_var_name = self._suitability_var_name
        table_suitability_vals = suitability_table[suitability_var_name].values

        def suitability_func(temperature_curr):
            suitability_curr = np.interp(
                temperature_curr,
                table_temp_vals,
                table_suitability_vals,
                left=0,
                right=0,
            )
            return suitability_curr

        da_suitability = xr.apply_ufunc(
            suitability_func, temperature, dask="parallelized"
        )
        da_suitability.attrs = suitability_table[suitability_var_name].attrs
        return da_suitability

    def _run_main_temp_precip_table(self, ds_clim):
        # Run the main logic of a suitability model defined by a temperature and
        # precipitation suitability table.
        temperature = ds_clim["temperature"]
        precipitation = ds_clim["precipitation"]
        suitability_table = self.suitability_table
        suitability_var_name = self._suitability_var_name
        table_values = (
            suitability_table[suitability_var_name]
            .transpose("temperature", "precipitation")
            .values
        )
        table_temp_vals = suitability_table["temperature"].values
        table_temp_deltas = np.diff(table_temp_vals)
        table_precip_vals = suitability_table["precipitation"].values
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

        def suitability_func(temp_inds_curr, precip_inds_curr):
            suitability_curr = table_values[temp_inds_curr, precip_inds_curr]
            return suitability_curr

        da_suitability = xr.apply_ufunc(
            suitability_func, temp_inds, precip_inds, dask="parallelized"
        )
        return da_suitability
