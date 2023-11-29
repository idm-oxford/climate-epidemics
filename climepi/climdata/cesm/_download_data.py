import os
import pathlib
import re

import aiohttp
import numpy as np
import pandas as pd
import parfive
import pooch
import xarray as xr
import xcdat
from siphon.catalog import TDSCatalog

CESM_DATA_VARS = ["TS", "PRECT"]
VAR_NAME_MAPPING = {"TS": "temperature", "PRECT": "precipitation"}

URL_SAVE_DIR = pooch.os_cache("climepi/cesm_urls")
TEMP_FILE_DIR = pooch.os_cache("climepi/cesm_temp")


class ParfiveDownloaderPostprocess(parfive.Downloader):
    max_tries = 3
    total_timeout = 60

    def __init__(self, postprocess=None):
        config = parfive.SessionConfig(
            timeouts=aiohttp.ClientTimeout(total=self.total_timeout)
        )
        super().__init__(overwrite=True, config=config)
        self._postprocess = postprocess
        self._raw_files_to_delete = []

    def download(self):
        results = super().download()
        self._delete_raw_files(max_fails=0)
        return results

    def download_with_retries(self):
        results = self.download()
        no_errors = len(results.errors)
        if no_errors == 0:
            return results
        tries = 1
        while no_errors > 0 and tries < self.max_tries:
            print(
                f"Retrying download of {no_errors} files "
                f"({tries+1}/{self.max_tries})..."
            )
            results = super().retry(results)
            self._delete_raw_files(max_fails=0)
            no_errors = len(results.errors)
            tries += 1
        if no_errors > 0:
            raise RuntimeError(
                f"Unable to download {no_errors} files after "
                f"{self.max_tries} tries."
            )
        return results

    async def _get_http(self, *args, **kwargs):
        filepath_raw = await super()._get_http(*args, **kwargs)
        postprocess = self._postprocess
        if postprocess is None:
            return filepath_raw
        filepath_processed = self._postprocess(filepath_raw)
        self._raw_files_to_delete.append(filepath_raw)
        self._delete_raw_files()
        return filepath_processed

    def _delete_raw_files(self, max_fails=1):
        files_to_delete = self._raw_files_to_delete
        files_couldnt_delete = []
        for filepath in files_to_delete:
            try:
                pathlib.Path(filepath).unlink()
            except WindowsError as exc:
                files_couldnt_delete.append(filepath)
                if len(files_couldnt_delete) > max_fails:
                    raise RuntimeError("Temporary files could not be deleted.") from exc
        self._raw_files_to_delete = files_couldnt_delete


class CESMDataFinder:
    def __init__(self, years=None, realizations=None):
        self._years = years
        self._realizations = realizations
        self._urls_all = None
        self._urls_filtered = None
        self._ensemble_ids_all = None
        self._ensemble_ids_filtered = None
        self._id_realization_mapping = None
        os.makedirs(URL_SAVE_DIR, exist_ok=True)
        self._save_path = pathlib.Path(URL_SAVE_DIR).joinpath("urls.csv")

    def get_urls_all(self):
        if self._urls_all is None:
            try:
                self._load_urls_all()
            except FileNotFoundError:
                self._find_urls()
        return self._urls_all

    def get_ensemble_ids_all(self):
        if self._ensemble_ids_all is None:
            urls_all = self.get_urls_all()
            regex = re.compile(r"LE2-(?P<id>.{8})\.cam")
            ensemble_ids_all = []
            for i in range(len(urls_all)):  # pylint: disable=consider-using-enumerate
                url = urls_all[i]
                match = regex.search(url)
                if not match:
                    raise ValueError("URL does not match expected pattern")
                match_dict = match.groupdict()
                ensemble_ids_all.append(match_dict["id"])
            self._ensemble_ids_all = ensemble_ids_all
        return self._ensemble_ids_all

    def get_urls_filtered(self):
        if self._urls_filtered is None:
            self._filter()
        return self._urls_filtered

    def get_ensemble_ids_filtered(self):
        if self._ensemble_ids_filtered is None:
            self._filter()
        return self._ensemble_ids_filtered

    def get_id_realization_mapping(self):
        if self._id_realization_mapping is None:
            ensemble_ids_all = self.get_ensemble_ids_all()
            unique_ensemble_ids = sorted(
                list(set(ensemble_ids_all)),
                key=lambda x: (int(x.split(".")[0]), int(x.split(".")[1])),
            )
            assert (
                len(unique_ensemble_ids) == 100
            ), "Expected to find 100 ensemble members."
            id_realization_mapping = {
                id: realization for realization, id in enumerate(unique_ensemble_ids)
            }
            self._id_realization_mapping = id_realization_mapping
        return self._id_realization_mapping

    def save_urls_all(self):
        urls_all = self.get_urls_all()
        save_path = self._save_path
        df = pd.DataFrame(urls_all, columns=["url"])
        df.to_csv(save_path)

    def _load_urls_all(self):
        save_path = self._save_path
        df = pd.read_csv(save_path)
        urls_all = df["url"].values.tolist()
        self._urls_all = urls_all

    def _find_urls(self):
        siphon_datasets = []
        print("Searching for CESM data files...")
        for var in CESM_DATA_VARS:
            if var == "TS":
                catalog_url = (
                    "https://tds.ucar.edu/thredds/catalog/"
                    + "esgcet/459/ucar.cgd.cesm2le.atm.proc.monthly_ave."
                    + var
                    + ".v1.xml"
                )
            elif var == "PRECT":
                catalog_url = (
                    "https://tds.ucar.edu/thredds/catalog/"
                    + "esgcet/457/ucar.cgd.cesm2le.atm.proc.monthly_ave."
                    + var
                    + ".v1.xml"
                )
            else:
                raise ValueError("Variable name not recognized")
            siphon_catalog = TDSCatalog(catalog_url)
            siphon_datasets += [
                siphon_catalog.datasets[i] for i in range(len(siphon_catalog.datasets))
            ]
        print("Data found.")
        urls_all = [
            "https://tds.ucar.edu/thredds/fileServer"
            + siphon_dataset.access_urls["GRIDFTPatNCAR"]
            for siphon_dataset in siphon_datasets
        ]
        self._urls_all = urls_all

    def _filter(self):
        years = self._years
        realizations = self._realizations
        urls_all = self.get_urls_all()
        ensemble_ids_all = self.get_ensemble_ids_all()
        if realizations is None:
            urls_filtered_realization = urls_all
            ensemble_ids_filtered_realization = ensemble_ids_all
        else:
            id_realization_mapping = self.get_id_realization_mapping()
            urls_filtered_realization = []
            ensemble_ids_filtered_realization = []
            for url, ensemble_id in zip(urls_all, ensemble_ids_all):
                realization_data = id_realization_mapping[ensemble_id]
                if realization_data in realizations:
                    urls_filtered_realization.append(url)
                    ensemble_ids_filtered_realization.append(ensemble_id)
        if years is None:
            urls_filtered = urls_filtered_realization
            ensemble_ids_filtered = ensemble_ids_filtered_realization
        else:
            regex_time = re.compile(
                r"(?P<startyear>\d{4})(?P<startmonth>\d{2})"
                + "-"
                + r"(?P<endyear>\d{4})(?P<endmonth>\d{2})"
            )
            urls_filtered = []
            ensemble_ids_filtered = []
            for url, ensemble_id in zip(
                urls_filtered_realization, ensemble_ids_filtered_realization
            ):
                match_time = regex_time.search(url)
                if not match_time:
                    raise ValueError("URL does not match expected pattern")
                match_time_dict = match_time.groupdict()
                years_data = np.arange(
                    int(match_time_dict["startyear"]),
                    int(match_time_dict["endyear"]) + 1,
                )
                if any(np.isin(years_data, years)):
                    urls_filtered.append(url)
                    ensemble_ids_filtered.append(ensemble_id)
        self._urls_filtered = urls_filtered
        self._ensemble_ids_filtered = ensemble_ids_filtered


class CESMDataDownloader:
    """
    Class for downloading CESM LENS2 data.
    """

    def __init__(
        self,
        data_dir,
        years,
        realizations=None,
        lon_range=None,
        lat_range=None,
        loc_str=None,
        temp_file_dir=TEMP_FILE_DIR,
    ):
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._years = years
        self._realizations = realizations
        self._lon_range = lon_range
        self._lat_range = lat_range
        self._loc_str = loc_str
        self._urls = None
        self._temp_file_paths = None
        self._temp_file_ensemble_ids = None
        self._id_realization_mapping = None
        self._file_paths = None
        self._temp_file_dir = temp_file_dir

    def download(self):
        self._find_data()
        self._download_data()
        self._organize_data()
        self._delete_temp_files()

    def _find_data(self):
        years = self._years
        realizations = self._realizations

        data_finder = CESMDataFinder(years=years, realizations=realizations)
        self._urls = data_finder.get_urls_filtered()
        self._temp_file_ensemble_ids = data_finder.get_ensemble_ids_filtered()
        self._id_realization_mapping = data_finder.get_id_realization_mapping()

    def _download_data(self):
        def _postprocess(path_in):
            path_out = path_in.replace(".nc", "_processed.nc")
            self._subset_format_dataset(path_in, path_out)
            return path_out

        downloader = ParfiveDownloaderPostprocess(postprocess=_postprocess)
        temp_file_dir = self._temp_file_dir
        urls = self._urls
        no_files = len(urls)
        temp_file_paths = []

        for url in urls:
            downloader.enqueue_file(url, path=temp_file_dir)

        if any(pathlib.Path(temp_file_dir).iterdir()):
            raise RuntimeError(
                f"Directory for temporary files ({temp_file_dir}) must be empty."
                " Either clear the directory or pass an empty directory to the"
                " 'temp_file_dir' argument of the CESMDataDownloader constructor."
            )

        print(f"Downloading {no_files} files...")
        temp_file_paths = [*downloader.download_with_retries()]
        print("Download completed.")

        self._temp_file_paths = temp_file_paths

    def _subset_format_dataset(self, path_in, path_out):
        years = self._years
        lon_range = self._lon_range
        lat_range = self._lat_range
        loc_str = self._loc_str

        with xr.open_dataset(path_in) as ds:
            ds = xcdat.center_times(ds)
            ds = xcdat.swap_lon_axis(ds, to=(-180, 180))
            ds = ds.rename_dims({"nbnd": "bnds"})
            ds = ds.bounds.add_missing_bounds(axes=["X", "Y"])

            keep_dims = ["lon", "lat", "time", "bnds"]
            keep_vars = list(CESM_DATA_VARS) + ["lon_bnds", "lat_bnds", "time_bnds"]
            ds = ds.drop_dims([dim for dim in ds.dims if dim not in keep_dims])
            ds = ds.drop_vars([var for var in ds.data_vars if var not in keep_vars])

            ds = ds.isel(time=np.isin(ds.time.dt.year, years))
            if loc_str is not None:
                ds.climepi.modes = {"spatial": "global"}
                ds = ds.climepi.sel_geopy(loc_str)
            elif lon_range is not None:
                ds = ds.sel(
                    lon=slice(*lon_range),
                    lat=slice(*lat_range),
                )

            for key, value in VAR_NAME_MAPPING.items():
                if key not in ds:
                    continue
                ds = ds.rename_vars({key: value})
                if value == "temperature":
                    ds[value] = ds[value] - 273.15
                    ds[value].attrs.update(long_name="Temperature")
                    ds[value].attrs.update(units="°C")
                elif value == "precipitation":
                    ds[value] = ds[value] * (1000 * 60 * 60 * 24)
                    ds[value].attrs.update(long_name="Precipitation")
                    ds[value].attrs.update(units="mm/day")

            ds.to_netcdf(path_out)

    def _organize_data(self):
        temp_file_paths = self._temp_file_paths
        temp_file_ensemble_ids = self._temp_file_ensemble_ids
        id_realization_mapping = self._id_realization_mapping

        for id_curr in set(temp_file_ensemble_ids):
            realization_curr = id_realization_mapping[id_curr]
            temp_file_paths_curr = [
                temp_file_paths[i]
                for i, id in enumerate(temp_file_ensemble_ids)
                if id == id_curr
            ]
            with xr.open_mfdataset(temp_file_paths_curr) as ds_curr:
                ds_curr["lon_bnds"] = ds_curr["lon_bnds"].isel(time=0)
                ds_curr["lat_bnds"] = ds_curr["lat_bnds"].isel(time=0)
                for var in VAR_NAME_MAPPING.values():
                    ds_curr[var] = ds_curr[var].expand_dims(
                        {"realization": [realization_curr]}
                    )
                file_path_curr = pathlib.Path(self._data_dir).joinpath(
                    "realization_" + str(realization_curr) + ".nc"
                )
                ds_curr.to_netcdf(file_path_curr)

    def _delete_temp_files(self):
        temp_file_paths = self._temp_file_paths
        for path in temp_file_paths:
            pathlib.Path(path).unlink()


if __name__ == "__main__":
    data_finder = CESMDataFinder()
    data_finder.get_urls_all()
    data_finder.save_urls_all()
