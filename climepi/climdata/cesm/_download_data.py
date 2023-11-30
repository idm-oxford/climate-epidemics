import pathlib

import intake
import numpy as np
import pooch
import xarray as xr
import xcdat

DEFAULT_SAVE_DIR = pooch.os_cache("climepi")


def get_cesm_data(
    years,
    realizations=None,
    lon_range=None,
    lat_range=None,
    loc_str=None,
    download=False,
    save_dir=DEFAULT_SAVE_DIR,
):
    data_getter = CESMDataGetter(
        years,
        realizations=realizations,
        lon_range=lon_range,
        lat_range=lat_range,
        loc_str=loc_str,
        save_dir=save_dir,
    )
    ds = data_getter.get_data(download=download)
    return ds


class CESMDataGetter:
    def __init__(
        self,
        years,
        realizations=None,
        lon_range=None,
        lat_range=None,
        loc_str=None,
        save_dir=DEFAULT_SAVE_DIR,
    ):
        self._years = years
        self._realizations = realizations
        self._lon_range = lon_range
        self._lat_range = lat_range
        self._loc_str = loc_str
        self._catalog = None
        self._ds_in = None
        self._ds_dict = None
        self._ds = None
        self._save_dir = save_dir
        self._file_name_dict = None
        self._file_names = None

    @property
    def catalog(self):
        if self._catalog is None:
            print("Loading data catalog...")
            self._catalog = intake.open_esm_datastore(
                "https://raw.githubusercontent.com/NCAR/cesm2-le-aws"
                + "/main/intake-catalogs/aws-cesm2-le.json"
            )
            print("Data catalog loaded.")
        return self._catalog

    @property
    def file_name_dict(self):
        if self._file_name_dict is None:
            base_name_str_list = ["cesm"]
            if all(np.diff(self._years) == 1):
                base_name_str_list.extend(
                    [f"{self._years[0]}", "to", f"{self._years[-1]}"]
                )
            else:
                base_name_str_list.extend([f"{year}" for year in self._years])
            if self._loc_str is not None:
                base_name_str_list.append(self._loc_str.replace(" ", "_"))
            else:
                if self._lon_range is not None:
                    base_name_str_list.extend(
                        ["lon", f"{self._lon_range[0]}", "to", f"{self._lon_range[1]}"]
                    )
                if self._lat_range is not None:
                    base_name_str_list.extend(
                        ["lat", f"{self._lat_range[0]}", "to", f"{self._lat_range[1]}"]
                    )
            file_name_dict = {}
            for realization in self._realizations:
                name_str_list = base_name_str_list.copy()
                name_str_list.extend(["realization", f"{realization}.nc"])
                name_str = "_".join(name_str_list)
                file_name_dict[realization] = name_str
            self._file_name_dict = file_name_dict
        return self._file_name_dict

    @property
    def file_names(self):
        if self._file_names is None:
            self._file_names = list(self.file_name_dict.values())
        return self._file_names

    def get_data(self, download=False):
        try:
            self._open_local_data()
            return self._ds
        except FileNotFoundError:
            pass
        self._open_remote_data()
        self._process_remote_data()
        if download:
            self._download_data()
            self._open_local_data()
        return self._ds

    def _open_local_data(self):
        save_dir = pathlib.Path(self._save_dir)
        file_names = self.file_names
        ds = xcdat.open_mfdataset([save_dir / file_name for file_name in file_names])
        self._ds = ds

    def _open_remote_data(self):
        print("Opening remote data...")
        catalog = self.catalog
        catalog_subset = catalog.search(
            variable=["TREFHT", "PRECC", "PRECL"], frequency="monthly"
        )
        ds_dict_in = catalog_subset.to_dataset_dict(storage_options={"anon": True})
        ds_cmip6_in = xr.concat(
            [
                ds_dict_in["atm.historical.monthly.cmip6"],
                ds_dict_in["atm.ssp370.monthly.cmip6"],
            ],
            dim="time",
        )
        ds_smbb_in = xr.concat(
            [
                ds_dict_in["atm.historical.monthly.smbb"],
                ds_dict_in["atm.ssp370.monthly.smbb"],
            ],
            dim="time",
        )
        ds_in = xr.concat([ds_cmip6_in, ds_smbb_in], dim="member_id")
        self._ds_in = ds_in
        print("\n\nRemote data opened.")

    def _process_remote_data(self):
        years = self._years
        realizations = self._realizations
        lon_range = self._lon_range
        lat_range = self._lat_range
        loc_str = self._loc_str
        ds = self._ds_in.copy()
        ds = ds.swap_dims({"member_id": "realization"})
        ds["realization"] = range(100)
        ds = xcdat.swap_lon_axis(ds, to=(-180, 180))
        ds = ds.rename_dims({"nbnd": "bnds"})
        ds = ds.bounds.add_missing_bounds(axes=["X", "Y"])
        ds = ds.isel(time=np.isin(ds.time.dt.year, years))
        if loc_str is not None:
            ds.climepi.modes = {"spatial": "global"}
            ds = ds.climepi.sel_geopy(loc_str)
        else:
            if lon_range is not None:
                ds = ds.sel(lon=slice(*lon_range))
            if lat_range is not None:
                ds = ds.sel(lat=slice(*lat_range))
        ds_dict = {}
        for realization in realizations:
            ds_curr = ds.sel(realization=[realization])
            ds_curr["temperature"] = ds_curr["TREFHT"] - 273.15
            ds_curr["temperature"].attrs.update(long_name="Temperature")
            ds_curr["temperature"].attrs.update(units="°C")
            ds_curr["precipitation"] = (ds_curr["PRECC"] + ds_curr["PRECL"]) * (
                1000 * 60 * 60 * 24
            )
            ds_curr["precipitation"].attrs.update(long_name="Precipitation")
            ds_curr["precipitation"].attrs.update(units="mm/day")
            ds_curr = ds_curr.drop(["TREFHT", "PRECC", "PRECL"])
            ds_dict[realization] = ds_curr
        ds = xr.concat(list(ds_dict.values()), dim="realization")
        self._ds_dict = ds_dict
        self._ds = ds

    def _download_data(self):
        save_dir = pathlib.Path(self._save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name_dict = self._file_name_dict
        ds_dict = self._ds_dict
        print(f"Downloading data to {save_dir}...")
        for download_no, (realization, ds_curr) in enumerate(ds_dict.items()):
            save_path = save_dir / file_name_dict[realization]
            if save_path.exists():
                print("Data already downloaded for current model realization.")
            else:
                ds_curr.to_netcdf(save_path)
            print(
                "Data downloaded for",
                download_no + 1,
                "out of",
                len(ds_dict),
                "model realizations.",
            )
