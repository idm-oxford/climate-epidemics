import numpy as np
import numpy.testing as npt
import xarray as xr
import xarray.testing as xrt

from climepi import ClimEpiDatasetAccessor  # noqa
from tests.fixtures import generate_dataset


def test__init__():
    ds = generate_dataset()
    accessor = ClimEpiDatasetAccessor(ds)
    assert accessor._obj.identical(ds)


def test_sel_geo():
    ds = generate_dataset(lon_0_360=False)
    location = "Miami"
    lat = 25.7617
    lon = -80.1918
    result = ds.climepi.sel_geo(location=location)
    lat_result = result.lat.values
    lon_result = result.lon.values
    lat_expected = ds.lat.sel(lat=lat, method="nearest").values
    lon_expected = ds.lon.sel(lon=lon, method="nearest").values
    npt.assert_allclose(lat_result, lat_expected)
    npt.assert_allclose(lon_result, lon_expected)
