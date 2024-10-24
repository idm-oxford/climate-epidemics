"""
Unit tests for the _base module of the climdata subpackage.
"""

from unittest.mock import patch

import pytest

from climepi import climdata


@pytest.mark.parametrize("data_source", ["test", "lens2", "isimip"])
def test_get_climate_data(data_source):
    """
    Unit test for the get_climate_data function.
    """
    frequency = "hourly"
    subset = {
        "scenarios": ["overcast", "sunny"],
        "models": ["length", "inswinger"],
        "realizations": [1, 2],
        "years": [2015, 2016, 2018, 2100],
        "locations": "gabba",
        "lon_range": None,
        "lat_range": None,
    }
    save_dir = "."
    download = "probably"
    force_remake = "perhaps"
    max_subset_wait_time = 30

    kwargs_in = {
        "data_source": data_source,
        "frequency": frequency,
        "subset": subset,
        "save_dir": save_dir,
        "download": download,
        "force_remake": force_remake,
        "max_subset_wait_time": max_subset_wait_time,
    }

    if data_source == "test":
        with pytest.raises(ValueError):
            climdata.get_climate_data(**kwargs_in)
        return

    if data_source == "lens2":
        to_patch = "climepi.climdata._base.CESMDataGetter"
    elif data_source == "isimip":
        to_patch = "climepi.climdata._base.ISIMIPDataGetter"
    with patch(to_patch, autospec=True) as mock_data_getter:
        result = climdata.get_climate_data(**kwargs_in)
        if data_source == "lens2":
            mock_data_getter.assert_called_once_with(
                frequency=frequency,
                subset=subset,
                save_dir=save_dir,
            )
        elif data_source == "isimip":
            mock_data_getter.assert_called_once_with(
                frequency=frequency,
                subset=subset,
                save_dir=save_dir,
                subset_check_interval=10,
                max_subset_wait_time=max_subset_wait_time,
            )
        else:
            raise ValueError("Should not be here.")
        mock_data_getter.return_value.get_data.assert_called_once_with(
            download=download, force_remake=force_remake
        )
        assert result == mock_data_getter.return_value.get_data.return_value


def test_get_climate_data_file_names():
    """
    Test the get_climate_data_file_names function.
    """
    data_source = "isimip"
    frequency = "monthly"
    subset = {
        "scenarios": ["overcast"],
        "models": ["length", "inswinger"],
        "realizations": [3],
        "years": [2015],
        "locations": ["gabba", "mcg"],
    }
    result = climdata.get_climate_data_file_names(
        data_source, frequency=frequency, subset=subset
    )
    expected = [
        "isimip_monthly_2015_" + comb + ".nc"
        for comb in [
            "gabba_overcast_length_3",
            "gabba_overcast_inswinger_3",
            "mcg_overcast_length_3",
            "mcg_overcast_inswinger_3",
        ]
    ]
    assert sorted(result) == sorted(expected)


@pytest.mark.parametrize("data_source", ["lens2", "isimip", "test"])
def test_get_data_getter(data_source):
    """
    Test the _get_data_getter function.
    """
    frequency = "hourly"
    subset = {
        "scenarios": ["overcast", "sunny"],
        "models": ["length", "inswinger"],
        "realizations": [1, 2],
        "years": [2015, 2016, 2018, 2100],
        "locations": "gabba",
        "lon_range": None,
        "lat_range": None,
    }
    save_dir = "."
    download = "probably"
    force_remake = "perhaps"
    max_subset_wait_time = 30
    if data_source == "test":
        with pytest.raises(ValueError):
            climdata._base._get_data_getter(
                data_source,
                frequency=frequency,
                subset=subset,
                save_dir=save_dir,
                download=download,
                force_remake=force_remake,
                max_subset_wait_time=max_subset_wait_time,
            )
        return
    result = climdata._base._get_data_getter(
        data_source,
        frequency=frequency,
        subset=subset,
        save_dir=save_dir,
        max_subset_wait_time=max_subset_wait_time,
    )
    if data_source == "lens2":
        assert isinstance(result, climdata._cesm.CESMDataGetter)
        assert (
            getattr(result, "_max_subset_wait_time", "should not be here")
            == "should not be here"
        )
    elif data_source == "isimip":
        assert isinstance(result, climdata._isimip.ISIMIPDataGetter)
        assert result._max_subset_wait_time == max_subset_wait_time
    else:
        raise ValueError("Should not be here.")
    assert result._frequency == frequency
    assert result._subset == subset
    assert str(result._save_dir) == save_dir


# @pytest.mark.parametrize("timeout_error", ["none", "some", "all"])
# def test_get_climate_data_location_list(timeout_error):
#     """
#     Test the _get_climate_data_location_list function.
#     """
#     subset = {
#         "location": ["gabba", "waca", "mcg"],
#         "scenarios": ["overcast", "sunny"],
#     }
#     if timeout_error == "none":
#         timeout_error_dict = {"gabba": False, "waca": False, "mcg": False}
#     elif timeout_error == "some":
#         timeout_error_dict = {"gabba": False, "waca": True, "mcg": False}
#     elif timeout_error == "all":
#         timeout_error_dict = {"gabba": True, "waca": True, "mcg": True}

#     ds_all = generate_dataset(data_var=["marnus"], extra_dims={"scenarios": 2})
#     ds_all["marnus"].values = np.random.rand(*ds_all["marnus"].shape)
#     ds_all["scenarios"] = subset["scenarios"]

#     location_lat_lon_map = {
#         "gabba": (ds_all["lat"].values[0], ds_all["lon"].values[3]),
#         "waca": (ds_all["lat"].values[3], ds_all["lon"].values[3]),
#         "mcg": (ds_all["lat"].values[2], ds_all["lon"].values[1]),
#     }

#     def mock_get_climate_data(*args, **kwargs):
#         location = kwargs["subset"]["location"]
#         if timeout_error_dict[location]:
#             raise TimeoutError
#         assert isinstance(location, str)
#         lat, lon = location_lat_lon_map[location]
#         return ds_all.sel(lat=[lat], lon=[lon], method="nearest")

#     with patch.object(climdata._base, "get_climate_data", new=mock_get_climate_data):
#         if timeout_error == "all":
#             with pytest.raises(TimeoutError), pytest.warns(UserWarning):
#                 climdata._base._get_climate_data_location_list("test", subset=subset)
#             return
#         if timeout_error == "some":
#             with pytest.warns(UserWarning):
#                 result = climdata._base._get_climate_data_location_list(
#                     "test", subset=subset
#                 )
#         else:
#             result = climdata._base._get_climate_data_location_list(
#                 "test", subset=subset
#             )

#     assert "location" in result.dims
#     expected_locations_retrived = [
#         x for x in subset["location"] if not timeout_error_dict[x]
#     ]
#     assert list(result["location"].values) == expected_locations_retrived
#     assert "lon" not in result.dims
#     assert "lat" not in result.dims
#     xrt.assert_identical(result["time_bnds"], ds_all["time_bnds"])
#     for location in expected_locations_retrived:
#         lat, lon = location_lat_lon_map[location]
#         xrt.assert_identical(
#             result[["marnus", "lon_bnds", "lat_bnds"]]
#             .sel(location=location)
#             .drop_vars("location"),
#             ds_all[["marnus", "lon_bnds", "lat_bnds"]].sel(
#                 lat=lat, lon=lon, method="nearest"
#             ),
#         )
