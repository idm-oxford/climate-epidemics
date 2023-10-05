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
    def _run_main(self, ds_clim):
        da_suitability = xr.map_blocks(self._get_suitability, ds_clim)
        return da_suitability
    def _get_suitability(self, ds_clim):
        ds_suitability = self.suitability_table.sel(temperature=ds_clim['temperature'], precipitation=ds_clim['precipitation'], method='nearest')
        da_suitability = ds_suitability['suitability'].reset_coords(names=['temperature','precipitation'],drop=True)
        return da_suitability
    
def import_kaye_model():
    data_path = str(pathlib.Path(__file__).parent)+'/data/kaye_ecol_niche.nc'
    suitability_table = xr.open_dataset(data_path)
    epi_model = EcolNicheModel(suitability_table)
    return epi_model

if __name__=='__main__':
    import climepi.climdata.cesm as cesm
    from climepi.epimod import EpiModDatasetAccessor
    ds_clim = cesm.import_data()
    epi_model = import_kaye_model()
    ds_clim.epimod.model = epi_model
    ds_epi = ds_clim.epimod.run()
    ds_epi