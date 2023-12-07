"""
Base module defining an ecological niche model class and functionality for
importing the model of Kaye et al.
"""

import pathlib

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import shapely

    epi_model = import_kaye_model()
    suitability_table = epi_model.suitability_table
    temp_grid, precip_grid = xr.broadcast(
        suitability_table["temperature"].rename("temp_grid"),
        suitability_table["precipitation"].rename("precip_grid"),
    )
    suitable = suitability_table["suitability"] > 0.5
    temp_suitable_coords = temp_grid.where(suitable).values.flatten()
    temp_suitable_coords = temp_suitable_coords[~np.isnan(temp_suitable_coords)]
    precip_suitable_coords = precip_grid.where(suitable).values.flatten()
    precip_suitable_coords = precip_suitable_coords[~np.isnan(precip_suitable_coords)]
    pts = shapely.points(temp_suitable_coords, precip_suitable_coords)
    mp = shapely.MultiPoint(pts)
    bdry = mp.buffer(distance=0.1)

    suitability_table["suitability"].plot()
    plt.plot(*bdry.exterior.xy)
    plt.show()
