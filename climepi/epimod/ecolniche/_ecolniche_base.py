"""
Base module defining an ecological niche model class and functionality for
importing the model of Kaye et al.
"""

import pathlib

import numpy as np
import xarray as xr

from climepi.epimod import EpiModel


class EcolNicheModel(EpiModel):
    """
    Class for ecological niche models.

    Parameters:
    -----------
    suitability_table_in : xarray.Dataset
        A dataset containing Aedes aegypti suitability values for different
        temperature and precipitation combinations.
    """

    def __init__(self, suitability_table_in=None):
        super().__init__()
        self._suitability_table = suitability_table_in

    @property
    def suitability_table(self):
        """The suitability table (xarray.Dataset)."""
        if self._suitability_table is None:
            raise ValueError("Suitability table not set.")
        return self._suitability_table

    @suitability_table.setter
    def suitability_table(self, suitability_table_in):
        self._suitability_table = suitability_table_in

    def run_main1(self, ds_clim):
        """
        Runs the main logic of the ecological niche model on a given climate
        dataset.

        Parameters:
        -----------
        ds_clim : xarray.Dataset
            The input climate dataset.

        Returns:
        --------
        xarray.DataArray
            Boolean suitability values.
        """
        da_suitability = xr.map_blocks(self._get_suitability, ds_clim)
        return da_suitability

    def _get_suitability(self, ds_clim):
        # Function to apply to each block of the input dataset, ds_clim
        # (xarray.Dataset), using the suitability table to determine
        # suitability values.
        ds_suitability = self.suitability_table.sel(
            temperature=ds_clim["temperature"],
            precipitation=ds_clim["precipitation"],
            method="nearest",
        )
        da_suitability = ds_suitability["suitability"].reset_coords(
            names=["temperature", "precipitation"], drop=True
        )
        return da_suitability

    def run_main(self, ds_clim):
        temperature = ds_clim["temperature"]
        precipitation = ds_clim["precipitation"]
        suitability_table = self.suitability_table
        table_values = (
            suitability_table["suitability"]
            .transpose("temperature", "precipitation")
            .values
        )
        table_temp_vals = suitability_table["temperature"].values
        table_temp_delta = table_temp_vals[1] - table_temp_vals[0]
        table_precip_vals = suitability_table["precipitation"].values
        table_precip_delta = table_precip_vals[1] - table_precip_vals[0]

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
        da_suitability = da_suitability.rename("suitability")
        da_suitability.attrs = suitability_table["suitability"].attrs
        return da_suitability

    def run_main2(self, ds_clim):
        temperature = ds_clim["temperature"]
        precipitation = ds_clim["precipitation"]
        suitability_table = self.suitability_table
        table_values = (
            suitability_table["suitability"]
            .transpose("temperature", "precipitation")
            .values
        )
        table_temp_vals = suitability_table["temperature"].values
        table_precip_vals = suitability_table["precipitation"].values

        import scipy.interpolate

        interp_func = scipy.interpolate.RegularGridInterpolator(
            (table_temp_vals, table_precip_vals),
            table_values,
            method="nearest",
            bounds_error=False,
            fill_value=0,
        )

        def suitability_func(temp, precip):
            suitability = interp_func((temp, precip))
            return suitability

        da_suitability = xr.apply_ufunc(
            suitability_func, temperature, precipitation, dask="parallelized"
        )
        da_suitability = da_suitability.rename("suitability")
        da_suitability.attrs = suitability_table["suitability"].attrs
        return da_suitability


def import_kaye_model():
    """
    Imports the suitability model for the model of Kaye et al. from a netCDF
    file and returns an instance of the EcolNicheModel class.

    Returns:
    --------
    epi_model : EcolNicheModel
        An instance of the EcolNicheModel class initialized with the
        suitability table from the Kaye model.
    """
    data_path = str(pathlib.Path(__file__).parent) + "/data/kaye_ecol_niche.nc"
    suitability_table = xr.open_dataset(data_path)
    epi_model = EcolNicheModel(suitability_table)
    return epi_model
