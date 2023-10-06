import numpy as np
import xarray as xr
import cf_xarray.units
import pint_xarray
import hvplot.xarray
import geoviews.feature as gf
import cartopy.crs
import xclim
import xcdat
import warnings

@xr.register_dataset_accessor("climepi")
class ClimEpiDatasetAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        # self._ensemble_type = None
        # self._geo_scope = None
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
    
    def annual_mean(self, data_var=None):
        data_var = self._auto_select_data_var(data_var)
        if 'time' not in self._obj.sizes:
            raise ValueError('Annual mean only defined for time series.')
        if  np.issubdtype(self._obj[data_var].dtype, np.integer) or np.issubdtype(self._obj[data_var].dtype, 'bool'):
            # Workaround for bug in xcdat group-average using integer or boolean data types
            ds = self._obj.copy()
            ds[data_var] = ds[data_var].astype('float64')
            return ds.climepi.annual_mean(data_var)
        ds_m = self._obj.temporal.group_average(data_var, freq='year')
        return ds_m
    
    def ensemble_mean(self, data_var=None):
        data_var = self._auto_select_data_var(data_var)
        if 'realization' not in self._obj.sizes:
            raise ValueError('Ensemble mean only defined for ensemble data.')
        ds_m = xr.Dataset(attrs=self._obj.attrs)
        ds_m[data_var] = self._obj[data_var].mean(dim='realization')
        ds_m[data_var].attrs = self._obj[data_var].attrs
        ds_m[data_var].attrs['ensemble_mode'] = 'statistics'
        ds_m.climepi._copy_bnds(self._obj)
        return ds_m
    
    def ensemble_percentiles(self, data_var=None, values=[5,50,95], **kwargs):
        data_var = self._auto_select_data_var(data_var)
        if 'realization' not in self._obj.sizes:
            raise ValueError('Ensemble percentiles only defined for ensemble data.')
        ds_p = xr.Dataset(attrs=self._obj.attrs)
        ds_p[data_var] = xclim.ensembles.ensemble_percentiles(self._obj[data_var], values, split=False, **kwargs)
        ds_p[data_var].attrs = self._obj[data_var].attrs
        ds_p[data_var].attrs['ensemble_mode'] = 'statistics'
        ds_p.climepi._copy_bnds(self._obj)
        return ds_p
    
    def ensemble_mean_std_max_min(self, data_var=None, **kwargs):
        data_var = self._auto_select_data_var(data_var)
        if 'realization' not in self._obj.sizes:
            raise ValueError('Ensemble statistics only defined for ensemble data.')
        ds_stat_xclim = xclim.ensembles.ensemble_mean_std_max_min(self._obj[data_var].to_dataset(), **kwargs)
        stat_list = ['mean','std','max','min']
        stat_list_xclim = [data_var+'_'+stat_list[i] for i in range(len(stat_list))]
        stat_list_xclim[1]+='ev'
        ds_stat = xr.Dataset(attrs=self._obj.attrs)
        da_stat_xclim_list = [ds_stat_xclim[stat_list_xclim[i]].rename(data_var).expand_dims(dim={'statistic':[stat_list[i]]}) for i in range(len(stat_list))]
        ds_stat[data_var] = xr.concat(da_stat_xclim_list,dim='statistic')
        ds_stat[data_var].attrs = self._obj[data_var].attrs
        ds_stat[data_var].attrs['ensemble_mode'] = 'statistics'
        ds_stat.climepi._copy_bnds(self._obj)
        return ds_stat

    def ensemble_stats(self, data_var=None, conf_level = 90, **kwargs):
        data_var = self._auto_select_data_var(data_var)
        if 'realization' not in self._obj.sizes:
            raise ValueError('Ensemble statistics only defined for ensemble data.')
        ds_msmm = self._obj.climepi.ensemble_mean_std_max_min(data_var, **kwargs)
        ds_mci = self._obj.climepi.ensemble_percentiles(data_var, [50-conf_level/2, 50, 50+conf_level/2], **kwargs)
        ds_mci = ds_mci.rename({'percentiles':'statistic'}).assign_coords(statistic=['lower','median','upper'])
        ds_stat = xr.concat([ds_msmm,ds_mci],dim='statistic')
        return ds_stat
    
    # def ensemble_mean_conf(self, data_var, conf_level = 90, **kwargs):
    #     if 'realization' not in self._obj.sizes:
    #         raise ValueError('Ensemble statistics only defined for ensemble data.')
    #     ds_m = self.ensemble_mean(data_var)
    #     ds_p = self.ensemble_percentiles(data_var, [50-conf_level/2, 50+conf_level/2], **kwargs)
    #     da_m = ds_m[data_var].expand_dims(dim={'statistic':['mean']})
    #     da_p = ds_p[data_var].rename({'percentiles':'statistic'}).assign_coords(statistic=['lower','upper'])
    #     ds_mp = xr.Dataset(attrs=self._obj.attrs)
    #     ds_mp[data_var] = xr.concat([da_m,da_p], dim='statistic')
    #     ds_mp.assign_coords(statistic=['mean','lower','upper'])
    #     ds_mp[data_var].attrs['ensemble_mode'] = 'mean and CI'
    #     ds_mp.climepi._copy_bnds(self._obj)
    #     return ds_mp

    def plot_time_series(self, data_var=None, **kwargs):
        data_var = self._auto_select_data_var(data_var)
        da_plot = self._obj[data_var]
        if 'time' not in da_plot.sizes:
            raise ValueError('Time series plot only defined for time series.')
        kwargs_hvplot = {'x':'time', 'frame_width':600}
        kwargs_hvplot.update(kwargs)
        return da_plot.hvplot.line(**kwargs_hvplot)
    
    def plot_map(self, data_var=None, include_ocean=False, **kwargs):
        data_var = self._auto_select_data_var(data_var)
        da_plot = self._obj[data_var]
        if any(x not in da_plot.sizes for x in ('lat','lon')):
            raise ValueError('Map plot only defined for spatial data.')
        elif any(x not in ['time','lat','lon'] for x in da_plot.sizes):
            raise ValueError('Input variable has unsupported dimensions.')
        kwargs_hvplot = {'x':'lon','y':'lat','cmap':'viridis', 'project':True, 'geo':True, 'rasterize':True, 'coastline':True, 'frame_width':600, 'dynamic':False}
        if 'time' in da_plot.sizes:
            kwargs_hvplot['groupby'] = 'time'
        kwargs_hvplot.update(kwargs)
        if 'crs' not in kwargs_hvplot:
            kwargs_hvplot['crs'] = self.crs
        p_main = da_plot.hvplot.quadmesh(**kwargs_hvplot)
        if include_ocean:
            return p_main
        else:
            p_ocean = gf.ocean.options(fill_color='white')
            return p_main*p_ocean
    
    def plot_ensemble_ci(self, data_var=None, central='mean', conf_level=None, **kwargs):
        data_var = self._auto_select_data_var(data_var)
        if 'realization' in self._obj.sizes:
            return self.ensemble_stats(data_var, conf_level).climepi.plot_ensemble_ci(data_var, **kwargs)
        elif 'ensemble_mode' not in self._obj[data_var].attrs:
            raise ValueError('Invalid ensemble input type or formatting.')
        ds_ci = xr.Dataset(attrs=self._obj.attrs)
        ds_ci['lower'] = self._obj[data_var].sel(statistic='lower')
        ds_ci['upper'] = self._obj[data_var].sel(statistic='upper')
        kwargs_hv_ci = {'x':'time','y':'lower','y2':'upper','frame_width':600, 'alpha':0.2}
        kwargs_hv_ci.update(kwargs)
        p_ci = ds_ci.hvplot.area(**kwargs_hv_ci)
        if central is None:
            return p_ci
        else:
            da_central = self._obj[data_var].sel(statistic=central)
            kwargs_hv_central = {'x':'time','frame_width':600}
            kwargs_hv_central.update(kwargs)
            p_central = da_central.hvplot.line(**kwargs_hv_central)
            return p_central*p_ci
    
    def _auto_select_data_var(self, data_var): #could implement this as a decorator
        if data_var is None:
            data_vars = list(self._obj.data_vars)
            bnd_vars = ['lat_bnds','lon_bnds','time_bnds']
            data_vars_not_bnds = [data_vars[i] for i in range(len(data_vars)) if data_vars[i] not in bnd_vars]
            if len(data_vars_not_bnds) == 1:
                data_var = data_vars_not_bnds[0]
            else:
                raise ValueError('Data variable must be specified.')
        return data_var

    def _copy_bnds(self, ds_from):
        for var in ['lat_bnds','lon_bnds','time_bnds']:
            if var in ds_from.data_vars:
                self._obj[var] = ds_from[var]
                self._obj[var].attrs = ds_from[var].attrs
    
if __name__ == "__main__":
    from climepi.climdata.cesm import import_data
    ds = import_data()
    # ds = xr.tutorial.open_dataset('air_temperature').rename_vars({'air':'temperature'}).isel(time=range(10)).expand_dims(dim={'realization':[0]},axis=3)
    ds_tmm = ds.climepi.annual_mean('temperature')
    ds_tmm_em = ds_tmm.climepi.ensemble_mean()
    ds_tmm_es = ds_tmm.climepi.ensemble_stats()
    ds_tmm_es.climepi.plot_ensemble_ci()
    # ds_tmm_perc_ex = ds_tmm.sel(lon=0, lat=0, method='nearest').climepi.ensemble_percentiles('temperature', [5,50,95])
    # ds_tmm_stats_ex = ds_tmm.sel(lon=0, lat=0, method='nearest').climepi.ensemble_mean_conf('temperature', conf_level = 90)