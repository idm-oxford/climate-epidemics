import xarray as xr
import cf_xarray.units
import pint_xarray
import hvplot.xarray
import cartopy.crs
import warnings

class _SharedClimEpiAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._ensemble_type = None
        self._geo_scope = None
        self._crs = None
    
    # @property
    # def ensemble_type(self):
    #     #Possible values: 'single' and 'multiple' (may add other options later)
    #     if self._ensemble_type is None:
    #         if 'realization' in self._obj.dims:
    #             self._ensemble_type = 'multiple'
    #         else:
    #             self._ensemble_type = 'single'
    #     return self._ensemble_type
    
    # @property
    # def geo_scope(self):
    #     #Possible values: 'single' and 'multiple' (may add other options later)
    #     if any(x in self._obj.sizes.keys() for x in ['lat','lon']):
    #         self._geo_scope = 'multiple'
    #     else:
    #         self._geo_scope = 'single'
    #     return self._geo_scope

    # @property
    # def temporal_scope(self):
    #     #Possible values: 'single' and 'multiple' (may add other options later)
    #     if 'time' in self._obj.sizes.keys():
    #         self._temporal_scope = 'multiple'
    #     else:
    #         self._temporal_scope = 'single'
    #     return self._temporal_scope

    @property
    def crs(self):
        if self._crs is None:
            if 'crs' in self._obj.attrs:
                self._crs = cartopy.crs.CRS.from_cf(self._obj.attrs['crs'])
            else:
                self._crs = cartopy.crs.PlateCarree()
                warnings.warn('PlateCarree CRS assumed.') #TODO: Add setter method to set crs and mention it here.
        return self._crs


@xr.register_dataset_accessor("climepi")
class ClimEpiDatasetAccessor(_SharedClimEpiAccessor):
    pass


@xr.register_dataarray_accessor("climepi")
class ClimEpiDataArrayAccessor(_SharedClimEpiAccessor):

    def plot_time_series(self, **kwargs):
        if 'time' not in self._obj.sizes:
            raise ValueError('Time series plot only defined for time series.')
        elif any(x != 'time' for x in self._obj.sizes):
            raise ValueError('Time series plot only defined for single time series.')
        kwargs_hvplot = {'x':'time', 'frame_width':600}
        kwargs_hvplot.update(kwargs)
        return self._obj.hvplot.line(**kwargs_hvplot)
    
    def plot_map(self, **kwargs):
        if any(x not in self._obj.sizes for x in ('lat','lon')):
            raise ValueError('Map plot only defined for spatial data.')
        elif any(x not in ['time','lat','lon'] for x in self._obj.sizes):
            raise ValueError('Input variable has unsupported dimensions.')
        kwargs_hvplot = {'x':'lon','y':'lat','groupby':'time','cmap':'viridis', 'project':True, 'geo':True, 'rasterize':True, 'coastline':True, 'frame_width':600, 'dynamic':False}
        kwargs_hvplot.update(kwargs)
        if 'crs' not in kwargs_hvplot:
            kwargs_hvplot['crs'] = self.crs
        return self._obj.hvplot.quadmesh(**kwargs_hvplot)
    
if __name__ == "__main__":
    import climdata.cesm as cesm
    ds = cesm.import_data()
    # ds = xr.tutorial.open_dataset('air_temperature').rename_vars({'air':'temperature'}).isel(time=range(10)).expand_dims(dim={'realization':[0]},axis=3)
    plot = ds['temperature'].sel(lon=0, lat=0, realization=0, method='nearest').climepi.plot_time_series()
    # plot = ds['temperature'].isel(realization=0).climepi.plot_map()
    hvplot.show(plot)
