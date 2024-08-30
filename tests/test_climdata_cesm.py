"""
Unit tests for the CESMDataGetter class in the _cesm.py module of the climdata
subpackage.
"""

from unittest.mock import patch

import intake_esm
import numpy as np
import xarray as xr
import xarray.testing as xrt

from climepi.climdata._cesm import CESMDataGetter


def test_find_remote_data():
    """
    Test the _find_remote_data method of the CESMDataGetter class. The conversion of
    the intake_esm catalog to a dataset dictionary is mocked to avoid opening the
    actual remote data.
    """

    frequency = "monthly"
    ds = xr.Dataset(
        data_vars={
            var: xr.DataArray(np.random.rand(6, 4), dims=["time", "member_id"])
            for var in ["TREFHT", "PRECC", "PRECL"]
        },
        coords={
            "time": xr.DataArray(np.arange(6), dims="time"),
            "member_id": xr.DataArray(np.arange(4), dims="member_id"),
        },
    )

    def _mock_to_dataset_dict(catalog_subset, storage_options=None):
        assert sorted(catalog_subset.df.path.tolist()) == sorted(
            [
                "s3://ncar-cesm2-lens/atm/"
                + f"{frequency}/cesm2LE-{forcing}-{assumption}-{var}.zarr"
                for forcing in ["historical", "ssp370"]
                for assumption in ["cmip6", "smbb"]
                for var in ["TREFHT", "PRECC", "PRECL"]
            ]
        )
        assert storage_options == {"anon": True}
        return {
            "atm." + forcing + ".monthly." + assumption: ds.isel(
                time=3 * (forcing == "ssp370") + np.arange(3),
                member_id=2 * (assumption == "smbb") + np.arange(2),
            )
            for forcing in ["historical", "ssp370"]
            for assumption in ["cmip6", "smbb"]
        }

    data_getter = CESMDataGetter(frequency="monthly")

    with patch.object(
        intake_esm.core.esm_datastore, "to_dataset_dict", _mock_to_dataset_dict
    ):
        data_getter._find_remote_data()

    xrt.assert_identical(data_getter._ds, ds)
