import os
import pathlib
import re
from urllib.parse import urlparse

import numpy as np
import pooch
import xcdat
from siphon.catalog import TDSCatalog


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
    ):
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._years = years
        self._realizations = realizations
        self._lon_range = lon_range
        self._lat_range = lat_range
        self._loc_str = loc_str
        self._var_name_mapping = {"TS": "temperature", "PRECT": "precipitation"}
        self._urls = None
        self._temp_file_paths = None
        self._temp_file_ensemble_ids = None
        self._id_realization_mapping = None
        self._file_paths = None

    def download(self):
        self._find_data()
        self._download_data()
        self._organize_data()
        self._delete_temp_files()

    def _find_data(self):
        realizations = self._realizations

        print("Searching for remote data files...")
        siphon_datasets = []
        for var_name in self._var_name_mapping:
            # Delicate handling to avoid siphon datasets being converted to ID strings
            # when a collection is iterated over
            curr_datasets = TDSCatalog(_get_catalog_name(var_name)).datasets
            siphon_datasets += [curr_datasets[i] for i in range(len(curr_datasets))]
        print("Data found.")

        siphon_datasets = _filter_time_range(siphon_datasets, self._years)

        urls = _get_urls(siphon_datasets)
        temp_file_ensemble_ids = _get_ensemble_ids(siphon_datasets)

        ind = [
            x
            for x, _ in sorted(
                enumerate(temp_file_ensemble_ids),
                key=lambda y: (int(y[1].split(".")[0]), int(y[1].split(".")[1])),
            )
        ]
        urls = [urls[i] for i in ind]
        temp_file_ensemble_ids = [temp_file_ensemble_ids[i] for i in ind]

        unique_ensemble_ids = sorted(
            list(set(temp_file_ensemble_ids)),
            key=lambda x: (int(x.split(".")[0]), int(x.split(".")[1])),
        )
        assert len(unique_ensemble_ids) == 100, "Expected to find 100 ensemble members."
        id_realization_mapping = {
            id: realization for realization, id in enumerate(unique_ensemble_ids)
        }

        if realizations is not None:
            unique_ensemble_ids_include = [unique_ensemble_ids[i] for i in realizations]
            ind = [
                i
                for i, id in enumerate(temp_file_ensemble_ids)
                if id in unique_ensemble_ids_include
            ]
            urls = [urls[i] for i in ind]
            temp_file_ensemble_ids = [temp_file_ensemble_ids[i] for i in ind]
            id_realization_mapping = {
                id: id_realization_mapping[id] for id in unique_ensemble_ids_include
            }

        self._urls = urls
        self._temp_file_ensemble_ids = temp_file_ensemble_ids
        self._id_realization_mapping = id_realization_mapping

    def _download_data(self):
        def _pooch_postprocess(path_in, _action, _pup):
            path_out = path_in.replace(".nc", "_formatted.nc")
            self._subset_format_dataset(path_in, path_out)
            pathlib.Path(path_in).unlink()
            return path_out

        urls = self._urls
        file_names_in = [os.path.basename(urlparse(url).path) for url in urls]
        pup = pooch.create(
            base_url="",
            path=pooch.os_cache("climepi/CESM/temp"),
            registry={file_name: None for file_name in file_names_in},
            urls=dict(zip(file_names_in, urls)),
        )

        temp_file_paths = []

        print(f"Downloading {len(file_names_in)} files...")
        for i, file_name in enumerate(file_names_in):
            path = pup.fetch(file_name, processor=_pooch_postprocess, progressbar=True)
            temp_file_paths.append(path)
            print(f"Downloaded {i+1}/{len(file_names_in)} files.")

        self._temp_file_paths = temp_file_paths

    def _subset_format_dataset(self, path_in, path_out):
        years = self._years
        lon_range = self._lon_range
        lat_range = self._lat_range
        loc_str = self._loc_str
        var_name_mapping = self._var_name_mapping

        with xcdat.open_mfdataset(path_in, center_times=True) as ds:
            keep_vars = [
                "lon",
                "lon_bnds",
                "lat",
                "lat_bnds",
                "time",
                "time_bnds",
            ] + list(var_name_mapping.keys())
            ds = ds.drop_vars([var for var in ds.data_vars if var not in keep_vars])
            ds = ds.drop_dims([dim for dim in ds.dims if dim not in keep_vars])
            ds = xcdat.swap_lon_axis(ds, to=(-180, 180))

            ds = ds.isel(time=(np.isin(ds.time.dt.year, years)))
            if loc_str is not None:
                ds.climepi.modes = {"spatial": "global"}
                ds = ds.climepi.sel_geopy(loc_str)
            elif lon_range is not None:
                ds = ds.sel(
                    lon=slice(*lon_range),
                    lat=slice(*lat_range),
                )

            for key, value in var_name_mapping.items():
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
            ds_curr = xcdat.open_mfdataset(temp_file_paths_curr)

            for var in var_name_mapping.values():
                ds_curr[var] = ds_curr[var].expand_dims(
                    {"realization": [realization_curr]}
                )

            # file_path_curr = os.path.join(
            #     self._data_dir, "realization_" + str(realization_curr) + ".nc"
            # )
            file_path_curr = pathlib.Path(self._data_dir).joinpath(
                "realization_" + str(realization_curr) + ".nc"
            )
            ds_curr.to_netcdf(file_path_curr)

    def _delete_temp_files(self):
        temp_file_paths = self._temp_file_paths
        for path in temp_file_paths:
            pathlib.Path(path).unlink()


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


def _filter_time_range(siphon_datasets_in, years):
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
        years_data = np.arange(
            int(match_dict["startyear"]), int(match_dict["endyear"]) + 1
        )
        if any(np.isin(years_data, years)):
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
