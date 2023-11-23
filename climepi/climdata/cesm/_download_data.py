import os
import pathlib
import re
import shutil
import urllib.request
from urllib.parse import urlparse

import numpy as np
import xcdat
from siphon.catalog import TDSCatalog


class CESMDataDownloader:
    def __init__(self, years):
        self._years = years
        self._var_name_mapping = {"TS": "temperature", "PRECT": "precipitation"}
        file_dir = str(pathlib.Path(__file__).parent)
        self._temp_data_dir = file_dir + "/data/temp/"
        self._data_dir = file_dir + "/data/downloaded/"
        os.makedirs(self._temp_data_dir, exist_ok=True)
        os.makedirs(self._data_dir, exist_ok=True)
        self._urls = None
        self._temp_file_paths = None
        self._temp_file_ensemble_ids = None
        self._id_realization_mapping = None
        self._file_paths = None

    def download(self):
        self._find_data()
        self._download_data()
        self._format_data()

    def _find_data(self):
        siphon_datasets = []
        for var_name in self._var_name_mapping:
            # Delicate handling to avoid siphon datasets being converted to ID strings
            # when a collection is iterated over
            curr_datasets = TDSCatalog(_get_catalog_name(var_name)).datasets
            siphon_datasets += [curr_datasets[i] for i in range(len(curr_datasets))]
        siphon_datasets = _filter_time_range(
            siphon_datasets, np.min(self._years), np.max(self._years)
        )

        urls = _get_urls(siphon_datasets)
        temp_file_paths = [
            self._temp_data_dir + os.path.basename(urlparse(url).path) for url in urls
        ]
        temp_file_ensemble_ids = _get_ensemble_ids(siphon_datasets)

        ind = [
            x
            for x, _ in sorted(
                enumerate(temp_file_ensemble_ids),
                key=lambda y: (int(y[1].split(".")[0]), int(y[1].split(".")[1])),
            )
        ]
        urls = [urls[i] for i in ind]
        temp_file_paths = [temp_file_paths[i] for i in ind]
        temp_file_ensemble_ids = [temp_file_ensemble_ids[i] for i in ind]

        unique_ensemble_ids = sorted(
            list(set(temp_file_ensemble_ids)),
            key=lambda x: (int(x.split(".")[0]), int(x.split(".")[1])),
        )
        id_realization_mapping = {
            id: realization for realization, id in enumerate(unique_ensemble_ids)
        }

        no_sims_include = 2
        unique_ensemble_ids_include = unique_ensemble_ids[:no_sims_include]
        ind = [
            i
            for i, id in enumerate(temp_file_ensemble_ids)
            if id in unique_ensemble_ids_include
        ]
        urls = [urls[i] for i in ind]
        temp_file_paths = [temp_file_paths[i] for i in ind]
        temp_file_ensemble_ids = [temp_file_ensemble_ids[i] for i in ind]
        id_realization_mapping = {
            id: id_realization_mapping[id] for id in unique_ensemble_ids_include
        }

        self._urls = urls
        self._temp_file_paths = temp_file_paths
        self._temp_file_ensemble_ids = temp_file_ensemble_ids
        self._id_realization_mapping = id_realization_mapping

    def _download_data(self):
        urls = self._urls
        temp_file_paths = self._temp_file_paths

        no_files = len(urls)
        opener = urllib.request.build_opener()
        for file_no, url, temp_file_path in zip(range(no_files), urls, temp_file_paths):
            _download_from_url(opener, url, temp_file_path)
            print(f"Downloaded {file_no+1} of {no_files} files")

    def _format_data(self):
        var_name_mapping = self._var_name_mapping
        temp_file_paths = self._temp_file_paths
        temp_file_ensemble_ids = self._temp_file_ensemble_ids
        id_realization_mapping = self._id_realization_mapping

        for id_curr, realization_curr in id_realization_mapping.items():
            temp_file_paths_curr = [
                temp_file_paths[i]
                for i, id in enumerate(temp_file_ensemble_ids)
                if id == id_curr
            ]
            ds_curr = xcdat.open_mfdataset(temp_file_paths_curr, center_times=True)
            ds_curr = ds_curr[
                ["lon", "lon_bnds", "lat", "lat_bnds", "time", "time_bnds"]
                + list(var_name_mapping.keys())
            ]
            ds_curr = xcdat.swap_lon_axis(ds_curr, to=(-180, 180))
            ds_curr = ds_curr.rename_vars(var_name_mapping)

            ds_curr["temperature"] = ds_curr["temperature"] - 273.15
            ds_curr.temperature.attrs.update(long_name="Temperature")
            ds_curr.temperature.attrs.update(units="°C")
            ds_curr["precipitation"] = ds_curr["precipitation"] * (1000 * 60 * 60 * 24)
            ds_curr.precipitation.attrs.update(long_name="Precipitation")
            ds_curr.precipitation.attrs.update(units="mm/day")

            ds_curr = ds_curr.assign_coords(realization=realization_curr)

            file_path_curr = (
                self._data_dir + "realization_" + str(realization_curr) + ".nc"
            )
            ds_curr.to_netcdf(file_path_curr)


def _get_catalog_name(var_name):
    if var_name == "TS":
        return (
            "https://tds.ucar.edu/thredds/catalog/"
            + "esgcet/459/ucar.cgd.cesm2le.atm.proc.monthly_ave."
            + var_name
            + ".v1.xml"
        )
    if var_name == "PRECT":
        return (
            "https://tds.ucar.edu/thredds/catalog/"
            + "esgcet/457/ucar.cgd.cesm2le.atm.proc.monthly_ave."
            + var_name
            + ".v1.xml"
        )
    raise ValueError("Variable name not recognized")


def _filter_time_range(siphon_datasets_in, start_year, end_year):
    regex = re.compile(
        r"(?P<startyear>\d{4})(?P<startmonth>\d{2})"
        + "-"
        + r"(?P<endyear>\d{4})(?P<endmonth>\d{2})"
    )
    siphon_datasets_out = []
    for i in range(len(siphon_datasets_in)):  # pylint: disable=consider-using-enumerate
        dataset = siphon_datasets_in[i]
        match = regex.search(dataset.id)
        if not match:
            raise ValueError("Dataset name does not match expected pattern")
        match_dict = match.groupdict()
        start_year_data = int(match_dict["startyear"])
        end_year_data = int(match_dict["endyear"])
        if start_year_data <= end_year and end_year_data >= start_year:
            siphon_datasets_out.append(dataset)
    return siphon_datasets_out


def _get_urls(siphon_datasets):
    return [
        "https://tds.ucar.edu/thredds/fileServer"
        + siphon_dataset.access_urls["GRIDFTPatNCAR"]
        for siphon_dataset in siphon_datasets
    ]


def _get_ensemble_ids(siphon_datasets):
    regex = re.compile(r"LE2-(?P<id>.{8})\.cam")
    ids = []
    for i in range(len(siphon_datasets)):  # pylint: disable=consider-using-enumerate
        dataset = siphon_datasets[i]
        match = regex.search(dataset.id)
        if not match:
            raise ValueError("Dataset name does not match expected pattern")
        match_dict = match.groupdict()
        ids.append(match_dict["id"])
    return ids


def _download_from_url(opener, url, out_file_path):
    try:
        with opener.open(url) as response, open(out_file_path, "ab") as out_file:
            shutil.copyfileobj(response, out_file)
    except urllib.error.HTTPError as e:
        # Return code error (e.g. 404, 501, ...)
        print(f"HTTPError: {e.code}")
    except urllib.error.URLError as e:
        # Not an HTTP-specific error (e.g. connection refused)
        print(f"URLError: {e.reason}")


if __name__ == "__main__":
    years = np.arange(1850, 1851)
    downloader = CESMDataDownloader(years)
    downloader.download()

    file_dir_path = str(pathlib.Path(__file__).parent)
    paths = [
        file_dir_path + "/data/downloaded/realization_" + str(i) + ".nc"
        for i in range(1, 101)
    ]
    ds = xcdat.open_mfdataset(paths)
    ds
