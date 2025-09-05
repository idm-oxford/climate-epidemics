"""Module for accessing and downloading CESM LENS2 data."""

import functools
import time
from copy import deepcopy

import dask.diagnostics
import intake
import numpy as np
import pooch
import requests
import siphon.catalog
import xarray as xr

from climepi._core import ClimEpiDatasetAccessor  # noqa
from climepi._xcdat import BoundsAccessor, center_times  # noqa
from climepi.climdata._data_getter_class import ClimateDataGetter
from climepi.climdata._utils import _get_data_version


class CESMDataGetter(ClimateDataGetter):
    """
    Generic class for accessing and downloading CESM data from AWS servers.

    Subclasses should be created for data from specific CESM experiments (e.g. LENS2).

    See the base class (`climepi.climdata._data_getter_class.ClimateDataGetter`) for
    further details.
    """

    lon_res = 1.25
    lat_res = 180 / 191

    def _find_remote_data(self):
        raise NotImplementedError(
            "Method _find_remote_data must be implemented in a sub(sub)class."
        )

    def _subset_remote_data(self):
        # Subset the remotely opened dataset to the requested years and location(s), and
        # store the subsetted dataset in the _ds attribute.
        years = self._subset["years"]
        locations = self._subset["locations"]
        lon = self._subset["lon"]
        lat = self._subset["lat"]
        lon_range = self._subset["lon_range"]
        lat_range = self._subset["lat_range"]
        ds_subset = self._ds.copy()
        ds_subset = ds_subset.isel(time=np.isin(ds_subset.time.dt.year, years))
        if locations is not None:
            # Use the climepi package to find the nearest grid points to the provided
            # locations, and subset the data accordingly (ensure locations is a list
            # so "location" is made a dimension coordinate).
            location_list = np.atleast_1d(locations).tolist()
            lon_list = np.atleast_1d(lon).tolist() if lon is not None else None
            lat_list = np.atleast_1d(lat).tolist() if lat is not None else None
            ds_subset = ds_subset.climepi.sel_geo(
                location_list, lon=lon_list, lat=lat_list
            )
        else:
            if lon_range is not None:
                # Note the remote data are stored with longitudes in the range 0 to 360.
                if lon_range[0] < 0 <= lon_range[1]:
                    ds_subset = xr.concat(
                        [
                            ds_subset.sel(lon=slice(0, lon_range[1] % 360)),
                            ds_subset.sel(lon=slice(lon_range[0] % 360, 360)),
                        ],
                        dim="lon",
                        data_vars="minimal",
                    )
                else:
                    ds_subset = ds_subset.sel(
                        lon=slice(lon_range[0] % 360, lon_range[1] % 360)
                    )
            if lat_range is not None:
                ds_subset = ds_subset.sel(lat=slice(*lat_range))
        self._ds = ds_subset

    def _download_remote_data(self):
        # Download the remote dataset to a temporary file (printing a progress bar), and
        # store the file name in the _temp_file_names attribute.
        temp_save_dir = self._temp_save_dir
        temp_file_name = "temp_data.nc"
        temp_save_path = temp_save_dir / temp_file_name
        delayed_obj = self._ds.to_netcdf(temp_save_path, compute=False)
        with dask.diagnostics.ProgressBar():
            delayed_obj.compute()
        self._temp_file_names = [temp_file_name]

    def _open_temp_data(self, **kwargs):
        # Open the temporary dataset, and store the opened dataset in the _ds attribute.
        # Extends the parent method by specifying chunks for the member_id coordinate.
        kwargs = {
            "chunks": {"member_id": 1, "model": 1, "scenario": 1, "location": 1},
            **kwargs,
        }
        super()._open_temp_data(**kwargs)

    def _process_data(self):
        # Extends the parent method to add renaming, unit conversion and (depending on
        # the requested data frequency) temporal averaging.
        realizations = self._subset["realizations"]
        frequency = self._frequency
        ds_processed = self._ds.copy()
        # Calculate total precipitation from convective and large-scale precipitation
        if "PRECC" in ds_processed.data_vars and "PRECL" in ds_processed.data_vars:
            ds_processed["PRECT"] = ds_processed["PRECC"] + ds_processed["PRECL"]
            ds_processed = ds_processed.drop_vars(["PRECC", "PRECL"])
        # Index data variables by an integer realization coordinate instead of the
        # member_id coordinate (which is a string), and add model (and if necessary
        # scenario) dimensions.
        ds_processed = ds_processed.swap_dims({"member_id": "realization"})
        ds_processed["realization"] = realizations
        ds_processed[["TREFHT", "PRECT"]] = ds_processed[
            ["TREFHT", "PRECT"]
        ].expand_dims({"model": np.array(self.available_models, dtype="object")})
        if "scenario" not in ds_processed:
            ds_processed[["TREFHT", "PRECT"]] = ds_processed[
                ["TREFHT", "PRECT"]
            ].expand_dims(
                {"scenario": np.array(self.available_scenarios, dtype="object")}
            )
        # Add time bounds to the dataset (if not already present)
        if "time_bnds" not in ds_processed:
            ds_processed = ds_processed.bounds.add_time_bounds(
                method="freq", freq="day" if frequency == "daily" else "month"
            )
        # Make time bounds a data variable instead of a coordinate if necessary, and
        # format in order to match the conventions of xcdat.
        ds_processed = ds_processed.reset_coords("time_bnds")
        if "nbnd" in ds_processed.dims:
            ds_processed = ds_processed.rename_dims({"nbnd": "bnds"})
        # Center time coordinates (if necessary)
        ds_processed = center_times(ds_processed)
        # Convert temperature from Kelvin to Celsius.
        ds_processed["temperature"] = ds_processed["TREFHT"] - 273.15
        ds_processed["temperature"].attrs.update(long_name="Temperature")
        ds_processed["temperature"].attrs.update(units="°C")
        # Convert total precipitation from m/s to mm/day.
        ds_processed["precipitation"] = (ds_processed["PRECT"]) * (1000 * 60 * 60 * 24)
        ds_processed["precipitation"].attrs.update(long_name="Precipitation")
        ds_processed["precipitation"].attrs.update(units="mm/day")
        ds_processed = ds_processed.drop_vars(["TREFHT", "PRECT"])
        # Use capital letters for variable long names (for consistent plotting).
        ds_processed["time"].attrs.update(long_name="Time")
        ds_processed["lon"].attrs.update(long_name="Longitude")
        ds_processed["lat"].attrs.update(long_name="Latitude")
        # Add axis labels to attributes of the time, longitude and latitude coordinates.
        ds_processed["time"].attrs.update(axis="T")
        ds_processed["lon"].attrs.update(axis="X")
        ds_processed["lat"].attrs.update(axis="Y")
        # Take yearly averages if yearly data requested.
        if frequency == "yearly":
            ds_processed = ds_processed.climepi.yearly_average()
        self._ds = ds_processed
        super()._process_data()


class LENS2DataGetter(CESMDataGetter):
    """
    Class for accessing and downloading CESM2 LENS data.

    Data are taken from the AWS server (https://registry.opendata.aws/ncar-cesm2-lens/).
    Terms of use can be found at https://www.ucar.edu/terms-of-use/data.

    Available years that can be specified in the `subset` argument of the class
    constructor range from 1850 to 2100, and 100 realizations (here labelled as 0 to 99)
    are available for a single scenario ("ssp370") and model ("cesm2"). The remotely
    stored data can be lazily opened as an xarray dataset and processed without
    downloading (`download=False` option in the `get_data` method).

    See the base class (`climepi.climdata._data_getter_class.ClimateDataGetter`) for
    further details.
    """

    data_source = "lens2"
    available_models = ["cesm2"]
    available_years = np.arange(1850, 2101)
    available_scenarios = ["ssp370"]
    available_realizations = np.arange(100)
    remote_open_possible = True

    def _find_remote_data(self):
        # Use intake to find and (lazily) open the remote data, then combine into a
        # single dataset and store in the _ds attribute.
        frequency = self._frequency
        if frequency == "yearly":
            frequency = "monthly"
        catalog_url = (
            "https://raw.githubusercontent.com/NCAR/cesm2-le-aws"
            "/main/intake-catalogs/aws-cesm2-le.json"
        )
        catalog = intake.open_esm_datastore(catalog_url)
        print("\n")
        catalog_subset = catalog.search(
            variable=["TREFHT", "PRECC", "PRECL"], frequency=frequency
        )
        ds_dict_in = catalog_subset.to_dataset_dict(
            storage_options={"anon": True},
            xarray_combine_by_coords_kwargs={"coords": "minimal", "compat": "override"},
        )
        print("\n")
        ds_cmip6_in = xr.concat(
            [
                ds_dict_in[f"atm.historical.{frequency}.cmip6"],
                ds_dict_in[f"atm.ssp370.{frequency}.cmip6"],
            ],
            dim="time",
        )
        ds_smbb_in = xr.concat(
            [
                ds_dict_in[f"atm.historical.{frequency}.smbb"],
                ds_dict_in[f"atm.ssp370.{frequency}.smbb"],
            ],
            dim="time",
        )
        ds_in = xr.concat([ds_cmip6_in, ds_smbb_in], dim="member_id")
        self._ds = ds_in

    def _subset_remote_data(self):
        # Add realization subsetting to the parent method
        realizations = self._subset["realizations"]
        ds_subset = self._ds.copy()
        ds_subset = ds_subset.isel(member_id=realizations)
        self._ds = ds_subset
        super()._subset_remote_data()


class ARISEDataGetter(CESMDataGetter):
    """
    Class for accessing and downloading CESM2 ARISE data.

    Data are taken from the AWS server
    (https://registry.opendata.aws/ncar-cesm2-arise/). Terms of use can be found at
    https://www.ucar.edu/terms-of-use/data.

    Available years that can be specified in the `subset` argument of the class
    constructor range from 2035 to 2069 for feedback simulations (scenario "sai15"),
    and 2000 to 2100 for reference simulations without climate intervention (scenario
    "ssp245"). 10 realizations (here labelled as 0 to 9) are available for each
    scenario. The remotely stored data can be lazily opened as an xarray dataset and
    processed without downloading (`download=False` option in the `get_data` method).

    See the base class (`climepi.climdata._data_getter_class.ClimateDataGetter`) for
    further details.
    """

    data_source = "arise"
    available_models = ["cesm2"]
    available_years = np.arange(2035, 2070)
    available_scenarios = ["ssp245", "sai15"]
    available_realizations = np.arange(10)
    remote_open_possible = True

    def _find_remote_data(self):
        # Running to_datset_dict() on the catalog subset doesn't seem to work for
        # reference/kerchunk format data in AWS, so open the datasets manually.
        frequency = self._frequency
        years = self._subset["years"]
        realizations = self._subset["realizations"]
        member_ids = [f"{(i + 1):03d}" for i in realizations]
        scenarios = self._subset["scenarios"]
        if frequency in ["monthly", "yearly"]:
            catalog_frequency = "month_1"
        elif frequency == "daily":
            catalog_frequency = "day_1"
        else:
            raise ValueError(f"Frequency {frequency} is not supported.")

        version = _get_data_version()

        urls = []

        for scenario in scenarios:
            if scenario == "ssp245":
                catalog_url = (
                    "https://github.com/idm-oxford/climate-epidemics/raw/"
                    f"{version}/data/catalogs/cesm2-waccm-ssp245.json"
                )
            elif scenario == "sai15":
                catalog_url = (
                    "https://github.com/idm-oxford/climate-epidemics/raw/"
                    f"{version}/data/catalogs/cesm2-arise-sai-1.5.json"
                )
            else:
                raise ValueError(f"Scenario {scenario} is not supported.")
            catalog = intake.open_esm_datastore(catalog_url)

            catalog_subset = catalog.search(
                variable=["TREFHT", "PRECT"],
                frequency=catalog_frequency,
                member_id=member_ids,
            )
            datasets = [
                {
                    "url": url,
                    "start_year": int(url.split(".")[-2][:4]),
                    "end_year": int(url.split(".")[-2].split("-")[1][:4]),
                    "member_id": member_id,
                }
                for url, member_id in zip(
                    catalog_subset.df.path,
                    catalog_subset.df.member_id,
                    strict=True,
                )
            ]
            urls.extend(
                [
                    dataset["url"]
                    for dataset in datasets
                    if dataset["member_id"] in member_ids
                    and np.any(
                        (np.array(years) >= dataset["start_year"])
                        & (np.array(years) <= dataset["end_year"])
                    )
                ]
            )
        _preprocess = functools.partial(
            _preprocess_arise_dataset,
            frequency=frequency,
            years=years,
            realizations=realizations,
        )
        ds_in = xr.open_mfdataset(
            urls,
            chunks={},
            preprocess=_preprocess,
            engine="kerchunk",
            data_vars="minimal",
            parallel=True,
            join="inner",
            coords="minimal",  # will become xarray default
            compat="override",  # will become xarray default
            storage_options={
                "remote_options": {"anon": True},
                "target_options": {"anon": True},
            },
        )
        self._ds = ds_in


class GLENSDataGetter(CESMDataGetter):
    """Class for downloading CESM1 GLENS data from the UCAR server."""

    available_models = ["cesm1"]
    lon_res = 1.25
    lat_res = 180 / 191
    data_source = "glens"
    available_years = np.arange(2010, 2100)
    available_scenarios = ["rcp85", "sai"]
    available_realizations = np.arange(20)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._urls = None

    def get_data(self, *args, **kwargs):
        scenarios = self._subset["scenarios"]
        if len(scenarios) == 1:
            return super().get_data(*args, **kwargs)
        try:
            # Will raise an error unless data have been downloaded and force_remake is
            # false (allowing us to avoid the below loop over scenarios when we don't
            # need to download the data)
            return super().get_data(*args, **{**kwargs, "download": False})
        except ValueError:
            pass
        for scenario in scenarios:
            print(f"Getting data for scenario {scenario}...")
            data_getter_curr = deepcopy(self)
            data_getter_curr._subset["scenarios"] = [scenario]
            data_getter_curr._file_names = None
            data_getter_curr._file_name_da = None
            data_getter_curr.get_data(*args, **kwargs)
        return super().get_data(
            *args, **{**kwargs, "force_remake": False, "download": False}
        )

    def _find_remote_data(self):
        frequency = self._frequency
        years = self._subset["years"]
        realizations = self._subset["realizations"]
        member_ids = [f"{(i + 1):03d}" for i in realizations]
        scenario = np.array(self._subset["scenarios"]).item()

        if frequency in ["monthly", "yearly"]:
            data_vars = ["TREFHT", "PRECC", "PRECL"]
        elif frequency == "daily":
            data_vars = ["TREFHT", "PRECT"]
        else:
            raise ValueError(f"Frequency {frequency} is not supported.")

        urls = []

        for data_var in data_vars:
            if scenario == "rcp85" and frequency in ["monthly", "yearly"]:
                catalog_url = (
                    "https://tds.ucar.edu/thredds/catalog/esgcet/343/ucar.cgd.ccsm4."
                    f"GLENS.Control.atm.proc.monthly_ave.{data_var}.v1.xml"
                )
                data_base_url = (
                    "https://data-osdf.rda.ucar.edu/ncar/rda/d651064/GLENS/"
                    f"Control/atm/proc/tseries/monthly/{data_var}/"
                )
            elif scenario == "rcp85" and frequency == "daily":
                catalog_url = (
                    "https://tds.ucar.edu/thredds/catalog/esgcet/495/ucar.cgd.ccsm4."
                    f"GLENS.Control.atm.proc.daily.ave.{data_var}.v1.xml"
                )
                data_base_url = (
                    "https://data-osdf.rda.ucar.edu/ncar/rda/d651064/GLENS/"
                    f"Control/atm/proc/tseries/daily/{data_var}/"
                )
            elif scenario == "sai" and frequency in ["monthly", "yearly"]:
                catalog_url = (
                    "https://tds.ucar.edu/thredds/catalog/esgcet/349/ucar.cgd.ccsm4."
                    f"GLENS.Feedback.atm.proc.monthly_ave.{data_var}.v1.xml"
                )
                data_base_url = (
                    "https://data-osdf.rda.ucar.edu/ncar/rda/d651064/GLENS/"
                    f"Feedback/atm/proc/tseries/monthly/{data_var}/"
                )
            elif scenario == "sai" and frequency == "daily":
                catalog_url = (
                    "https://tds.ucar.edu/thredds/catalog/esgcet/352/ucar.cgd.ccsm4."
                    f"GLENS.Feedback.atm.proc.daily_ave.{data_var}.v1.xml"
                )
                data_base_url = (
                    "https://data-osdf.rda.ucar.edu/ncar/rda/d651064/GLENS/"
                    f"Feedback/atm/proc/tseries/daily/{data_var}/"
                )
            else:
                raise ValueError(
                    f"Scenario {scenario} and/or frequency {frequency} not supported."
                )
            while True:
                try:
                    catalog = siphon.catalog.TDSCatalog(catalog_url)
                    break
                except requests.exceptions.HTTPError as e:
                    print(f"HTTP error opening catalog {catalog_url}: {e}")
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
            dataset_names = [
                dataset.url_path.split("/")[-1] for dataset in catalog.datasets.values()
            ]
            datasets = [
                {
                    "url": data_base_url + name,
                    "start_year": int(name.split(".")[-2][:4]),
                    "end_year": int(name.split(".")[-2].split("-")[1][:4]),
                    "member_id": name.split(".")[5],
                }
                for name in dataset_names
            ]
            urls.extend(
                [
                    dataset["url"]
                    for dataset in datasets
                    if dataset["member_id"] in member_ids
                    and np.any(
                        (np.array(years) >= dataset["start_year"])
                        & (np.array(years) <= dataset["end_year"])
                    )
                ]
            )
        self._urls = urls

    def _subset_remote_data(self):
        pass

    def _download_remote_data(self):
        # Download the remote data files to a temporary directory.
        urls = self._urls
        temp_save_dir = self._temp_save_dir
        temp_file_names = []

        for url in urls:
            url_parts = url.split("/")
            download_file_name = url_parts[-1]
            base_url = "/".join(url_parts[:-1])
            pup = pooch.create(
                base_url=base_url,
                path=temp_save_dir,
                registry={download_file_name: None},
                retry_if_failed=5,
            )
            pup.fetch(
                download_file_name,
                progressbar=True,
            )
            temp_file_names.append(download_file_name)
        self._temp_file_names = temp_file_names

    def _open_temp_data(self, **kwargs):
        # Need to preprocess the downloaded data files.
        frequency = self._frequency
        years = self._subset["years"]
        realizations = self._subset["realizations"]
        preprocess = functools.partial(
            _preprocess_glens_dataset,
            frequency=frequency,
            years=years,
            realizations=realizations,
        )
        kwargs = {"preprocess": preprocess, "join": "inner", **kwargs}
        super()._open_temp_data(**kwargs)

    def _process_data(self):
        super()._subset_remote_data()
        super()._process_data()


def _preprocess_arise_dataset(ds, frequency=None, years=None, realizations=None):
    try:
        scenario = (
            "sai15"
            if ds.attrs["case"].split(".")[-2] == "SSP245-TSMLT-GAUSS-DEFAULT"
            else "ssp245"
            if ".".join(ds.attrs["case"].split(".")[-3:-1]) == "CMIP6-SSP2-4.5-WACCM"
            else None
        )
    except (IndexError, KeyError):
        scenario = None
    if scenario is None:
        raise ValueError(
            f"Failed to parse scenario from case attribute '{ds.attrs['case']}'"
        )
    member_ids = [f"{(i + 1):03d}" for i in realizations]
    member_id = ds.attrs["case"].split(".")[-1]
    assert member_id in member_ids, f"Unexpected member_id {member_id}"
    data_var = [v for v in ds.data_vars if v in ["TREFHT", "PRECT"]][0]
    ds = ds[[data_var]]  # drops time_bnds (readded later)
    ds[data_var] = ds[data_var].expand_dims(
        member_id=[member_id],
        scenario=np.array([scenario], dtype="object"),
    )
    # times seem to be at end of interval for monthly data, so shift
    if frequency in ["monthly", "yearly"]:
        old_time = ds["time"]
        ds = ds.assign_coords(time=ds.get_index("time").shift(-1, freq="MS"))
        ds["time"].encoding = old_time.encoding
        ds["time"].attrs = old_time.attrs
    # Subset to requested years now to avoid concatenation/merging issues
    ds = ds.isel(time=np.isin(ds.time.dt.year, years))
    return ds


def _preprocess_glens_dataset(ds, frequency=None, years=None, realizations=None):
    try:
        scenario_str = ds.attrs["case"].split(".")[-2]
        scenario = (
            "rcp85"
            if scenario_str.lower() == "control"
            else "sai"
            if scenario_str.lower() == "feedback"
            else None
        )
    except (IndexError, KeyError):
        scenario = None
    if scenario is None:
        raise ValueError(
            f"Failed to parse scenario from case attribute '{ds.attrs['case']}'"
        )
    member_ids = [f"{(i + 1):03d}" for i in realizations]
    member_id = ds.attrs["case"].split(".")[-1]
    assert member_id in member_ids, f"Unexpected member_id {member_id}"
    data_var = [v for v in ds.data_vars if v in ["TREFHT", "PRECC", "PRECL", "PRECT"]][
        0
    ]
    ds = ds[[data_var]]  # drops time_bnds (avoids performance issues)
    ds[data_var] = ds[data_var].expand_dims(
        member_id=[member_id],
        scenario=np.array([scenario], dtype="object"),
    )
    # Times seem to be at end of interval, so shift
    old_time = ds["time"]
    if frequency in ["monthly", "yearly"]:
        ds = ds.assign_coords(time=ds.get_index("time").shift(-1, freq="MS"))
    elif frequency == "daily":
        ds = ds.assign_coords(time=ds.get_index("time").shift(-1, freq="D"))
    else:
        raise ValueError(f"Frequency {frequency} is not supported.")
    ds["time"].encoding = old_time.encoding
    ds["time"].attrs = old_time.attrs
    # Subset to requested years now to avoid concatenation/merging issues
    ds = ds.isel(time=np.isin(ds.time.dt.year, years))
    return ds
