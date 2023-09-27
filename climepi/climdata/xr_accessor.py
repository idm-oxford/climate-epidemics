import xarray as xr

@xr.register_dataarray_accessor("climdata")
class ClimDataAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj