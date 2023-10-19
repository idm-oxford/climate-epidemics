"""Base module of the epimod subpackage providing a base epidemiological model class and an xarray
accessor.
"""

import xarray as xr
# import climepi

class EpiModel:
    def __init__(self):
        pass
    def run(self, ds_clim):
        da_epi = self._run_main(ds_clim)
        ds_epi = xr.Dataset(attrs=ds_clim.attrs)
        ds_epi[da_epi.name] = da_epi
        ds_epi.climepi.copy_bnds_from(ds_clim)
        return ds_epi
    def _run_main(self, ds_clim):
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