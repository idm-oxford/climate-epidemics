import xarray as xr
import climepi

class EpiModel(object):
    def __init__(self):
        pass
    def run(self, clim_ds):
        epi_da = self._run_main(clim_ds)
        epi_ds = xr.Dataset(attrs=clim_ds.attrs)
        epi_ds[epi_da.name] = epi_da
        epi_ds.climepi._copy_bnds(clim_ds)
        return epi_ds
    def _run_main(self, clim_ds):
        raise NotImplementedError

@xr.register_dataset_accessor("epimod")
class EpiModDatasetAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._model = None
    @property
    def model(self):
        if self._model is None:
            raise ValueError('Model not set.')
        return self._model
    @model.setter
    def model(self, model):
        if not isinstance(model, EpiModel):
            raise ValueError('Model must be an instance of EpiModel.')
        self._model = model
    def run(self):
        return self.model.run(self._obj)