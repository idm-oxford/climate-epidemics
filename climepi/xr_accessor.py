import xarray as xr
import hvplot.xarray
import cartopy.crs as ccrs

@xr.register_dataarray_accessor("climepi")
class ClimEpiAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.crs = ccrs.PlateCarree()
    def plot_time_series(self, var, lon, lat, **kwargs):
        return self._obj.sel(lon=lon, lat=lat, method='nearest').hvplot.line('time', var, **kwargs)
    def plot_map(self, var, **kwargs):
        kwargs_out = {'cmap':'viridis', 'project':True, 'geo':True, 'rasterize':True, 'coastline':True, 'frame_width':600, 'dynamic':False}
        kwargs_out.update(kwargs)
        return self._obj[var].hvplot.quadmesh('lon', 'lat', var, crs=self.crs, **kwargs_out)