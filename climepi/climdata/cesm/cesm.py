import xarray as xr
import dask
import netCDF4
import os

def import_data(**kwargs):
    path_str = os.path.dirname(__file__)+'/data/sim*.nc'
    return xr.open_mfdataset(path_str, parallel=True, **kwargs)