import pathlib
import xarray as xr
from climepi.epimod import EpiModel

class EcolNicheModel(EpiModel):
    def __init__(self,suitability_table=None):
        EpiModel.__init__(self)
        self.suitability_table = suitability_table
    @property
    def suitability_table(self):
        if self._suitability_table is None:
            raise ValueError('Suitability table not set.')
        return self._suitability_table
    @suitability_table.setter
    def suitability_table(self, ds):
        self._suitability_table = ds
    def _run_main(self, clim_ds):
        suitability_ds = self.suitability_table.sel(temperature=clim_ds['temperature'], precipitation=clim_ds['precipitation'], method='nearest')
        suitability_da = suitability_ds['suitability'].reset_coords(names=['temperature','precipitation'],drop=True)
        return suitability_da
    
def import_kaye_model():
    data_path = str(pathlib.Path(__file__).parent)+'/data/kaye_ecol_niche.nc'
    suitability_table = xr.open_dataset(data_path)
    epi_model = EcolNicheModel(suitability_table)
    return epi_model
