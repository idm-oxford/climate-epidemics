"""Module for accessing and downloading CESM LENS2 data."""

import dask.diagnostics
import fsspec
import intake
import numpy as np
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

    remote_open_possible = True
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
        # Extends the parent method by using the chunks attribute of the original remote
        # dataset (unless overridden by the user in the kwargs argument).
        kwargs = {"chunks": self._ds.chunks.mapping, **kwargs}
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
        ].expand_dims({"model": self.available_models})
        if "scenario" not in ds_processed:
            ds_processed[["TREFHT", "PRECT"]] = ds_processed[
                ["TREFHT", "PRECT"]
            ].expand_dims({"scenario": self.available_scenarios})
        # Add time bounds to the dataset (if not already present)
        ds_processed = ds_processed.bounds.add_missing_bounds(axes=["T"])
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
        ds_dict_in = catalog_subset.to_dataset_dict(storage_options={"anon": True})
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
    constructor range from 2035 to 2069, and 10 realizations (here labelled as 0 to 9)
    are available for two scenarios ("ssp245" and "sai15") and one model ("cesm2"). The
    remotely stored data can be lazily opened as an xarray dataset and processed without
    downloading (`download=False` option in the `get_data` method).

    See the base class (`climepi.climdata._data_getter_class.ClimateDataGetter`) for
    further details.
    """

    data_source = "arise"
    available_models = ["cesm2"]
    available_years = np.arange(2035, 2070)
    available_scenarios = ["ssp245", "sai15"]
    available_realizations = np.arange(10)

    def _find_remote_data(self):
        # Running to_datset_dict() on the catalog subset doesn't seem to work for
        # reference/kerchunk format data in AWS, so open the datasets manually.
        frequency = self._frequency
        available_years = self.available_years
        realizations = self._subset["realizations"]
        member_ids = [f"{(i + 1):03d}" for i in realizations]
        scenarios = self._subset["scenarios"]
        if frequency in ["monthly", "yearly"]:
            catalog_frequency = "month_1"
        elif frequency == "daily":
            catalog_frequency = "day_1"
        else:
            raise ValueError(f"Frequency {frequency} is not supported.")

        def _preprocess(_ds):
            _member_id = _ds.attrs["case"].split(".")[-1]
            assert _member_id in member_ids, f"Unexpected member_id {_member_id}"
            _data_var = [v for v in _ds.data_vars if v in ["TREFHT", "PRECT"]][0]
            if frequency == "daily":
                # time_bnds supplied with daily data seem to be incorrect
                _ds = _ds[[_data_var]]
            elif frequency == "monthly":
                # monthly data seem to have time at the right of the bounds (which may
                # lead to cutting the last value when subsetting)
                _ds = _ds[[_data_var, "time_bnds"]]
                _ds = center_times(_ds)
            _ds[_data_var] = _ds[_data_var].expand_dims(member_id=[_member_id])
            return _ds

        version = _get_data_version()
        ds_list = []

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
            s3_urls = catalog_subset.df.path.to_list()
            mappers = [
                fsspec.filesystem(
                    "reference",
                    fo=url,
                    remote_protocol="s3",
                    remote_options={"anon": True},
                    target_options={"anon": True},
                ).get_mapper("")
                for url in s3_urls
            ]
            ds_curr = xr.open_mfdataset(
                mappers,
                engine="zarr",
                preprocess=_preprocess,
                backend_kwargs={"consolidated": False},
                data_vars="minimal",
                chunks={},
            )
            # Subset to available years (sims were run for different years within/
            # between scenarios)
            ds_curr = ds_curr.sel(
                time=slice(str(available_years[0]), str(available_years[-1]))
            )
            # Add scenario dimension over which to concatenate
            ds_curr[["TREFHT", "PRECT"]] = ds_curr[["TREFHT", "PRECT"]].expand_dims(
                {"scenario": [scenario]}
            )
            ds_list.append(ds_curr)
        ds_in = xr.concat(
            ds_list, dim="scenario", data_vars="minimal", combine_attrs="drop_conflicts"
        )
        self._ds = ds_in


class GLENSDataGetter(CESMDataGetter):
    """Class for downloading CESM1 GLENS data from the UCAR server."""

    available_models = ["cesm1"]
    lon_res = 1.25
    lat_res = 180 / 191
    data_source = "arise"
    available_years = np.arange(2010, 2100)
    available_scenarios = ["rcp85", "sai"]
    available_realizations = np.arange(20)

    def _find_remote_data(self):
        frequency = self._frequency
        years = self._subset["years"]
        realizations = self._subset["realizations"]
        member_ids = [f"{(i + 1):03d}" for i in realizations]
        scenarios = self._subset["scenarios"]
        api_token = self._api_token

        def _preprocess(_ds):
            _member_id = _ds.attrs["case"].split(".")[-1]
            assert _member_id in member_ids, f"Unexpected member_id {_member_id}"
            _data_var = [
                v for v in _ds.data_vars if v in ["TREFHT", "PRECC", "PRECL", "PRECT"]
            ][0]
            _ds = _ds[[_data_var, "time_bnds"]]
            _ds = center_times(_ds)
            _ds[_data_var] = _ds[_data_var].expand_dims(member_id=[_member_id])
            return _ds

        ds_list = []

        for scenario in scenarios:
            if scenario == "rcp85" and frequency in ["monthly", "yearly"]:
                catalog_urls = [
                    "https://tds.ucar.edu/thredds/catalog/esgcet/343/ucar.cgd.ccsm4."
                    f"GLENS.Control.atm.proc.monthly_ave.{data_var}.v1.xml"
                    for data_var in ["TREFHT", "PRECC", "PRECL"]
                ]
            elif scenario == "rcp85" and frequency == "daily":
                catalog_urls = [
                    "https://tds.ucar.edu/thredds/catalog/esgcet/495/ucar.cgd.ccsm4."
                    f"GLENS.Control.atm.proc.daily.ave.{data_var}.v1.xml"
                    for data_var in ["TREFHT", "PRECT"]
                ]
            elif scenario == "sai" and frequency in ["monthly", "yearly"]:
                catalog_urls = [
                    "https://tds.ucar.edu/thredds/catalog/esgcet/349/ucar.cgd.ccsm4."
                    f"GLENS.Feedback.atm.proc.monthly_ave.{data_var}.v1.xml"
                    for data_var in ["TREFHT", "PRECC", "PRECL"]
                ]
            elif scenario == "sai" and frequency == "daily":
                catalog_urls = [
                    "https://tds.ucar.edu/thredds/catalog/esgcet/352/ucar.cgd.ccsm4."
                    f"GLENS.Feedback.atm.proc.daily_ave.{data_var}.v1.xml"
                    for data_var in ["TREFHT", "PRECT"]
                ]
            else:
                raise ValueError(
                    f"Scenario {scenario} and/or frequency {frequency} not supported."
                )
            dataset_names = []
            for cat_url in catalog_urls:
                dataset_names.extend(
                    [
                        dataset.url_path
                        for dataset in siphon.catalog.TDSCatalog(
                            cat_url
                        ).datasets.values()
                    ]
                )
            datasets = [
                {
                    "name": name,
                    "url": "simplecache::"
                    f"https://tds.ucar.edu/thredds/fileServer/{name}"
                    f"?api-token={api_token}",
                    "start_year": int(name.split(".")[-2][:4]),
                    "end_year": int(name.split(".")[-2].split("-")[1][:4]),
                    "member_id": name.split(".")[5],
                }
                for name in dataset_names
            ]
            datasets = [
                dataset
                for dataset in datasets
                if np.any(
                    (np.array(years) >= dataset["start_year"])
                    & (np.array(years) <= dataset["end_year"])
                )
                and dataset["member_id"] in member_ids
            ]
            urls = [dataset["url"] for dataset in datasets]
            ds_curr = xr.open_mfdataset(
                urls,
                data_vars="minimal",
                preprocess=_preprocess,
                combine="nested",
                engine="h5netcdf",
                chunks={},
            )
            ds_list.append(ds_curr)
        ds_in = xr.concat(ds_list, dim="scenario", data_vars="minimal")
        self._ds = ds_in
