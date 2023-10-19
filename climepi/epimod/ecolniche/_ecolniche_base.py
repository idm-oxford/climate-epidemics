"""Base module defining an ecological niche model class and functionality for importing the model of
Kaye et al."""

import pathlib
import xarray as xr
from climepi.epimod import EpiModel


class EcolNicheModel(EpiModel):
    """
    Class for ecological niche models.

    Parameters:
    -----------
    suitability_table_in : xarray.Dataset
        A dataset containing Aedes aegypti suitability values for different temperature and
        precipitation combinations.
    """

    def __init__(self, suitability_table_in=None):
        super().__init__()
        self._suitability_table = suitability_table_in

    @property
    def suitability_table(self):
        """The suitability table (xarray.Dataset)."""
        if self._suitability_table is None:
            raise ValueError('Suitability table not set.')
        return self._suitability_table

    @suitability_table.setter
    def suitability_table(self, suitability_table_in):
        self._suitability_table = suitability_table_in

    def run_main(self, ds_clim):
        """
        Runs the main logic of the ecological niche model on a given climate dataset.

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
        # Function to apply to each block of the input dataset, ds_clim (xarray.Dataset), using the
        # suitability table to determine suitability values.
        ds_suitability = self.suitability_table.sel(
            temperature=ds_clim['temperature'], precipitation=ds_clim['precipitation'],
            method='nearest')
        da_suitability = ds_suitability['suitability'].reset_coords(
            names=['temperature', 'precipitation'], drop=True)
        return da_suitability


def import_kaye_model():
    """
    Imports the suitability model for the model of Kaye et al. from a netCDF file and returns an
    instance of the EcolNicheModel class.
    
    Returns:
    --------
    epi_model : EcolNicheModel
        An instance of the EcolNicheModel class initialized with the suitability table from the Kaye
        model.
    """
    data_path = str(pathlib.Path(__file__).parent)+'/data/kaye_ecol_niche.nc'
    suitability_table = xr.open_dataset(data_path)
    epi_model = EcolNicheModel(suitability_table)
    return epi_model
