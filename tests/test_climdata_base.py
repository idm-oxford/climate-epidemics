"""Unit tests for the _base module of the climdata subpackage."""

from unittest.mock import patch

import pytest

from climepi import climdata


@pytest.mark.parametrize("data_source", ["test", "lens2", "arise", "glens", "isimip"])
def test_get_climate_data(data_source):
    """Unit test for the get_climate_data function."""
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
    api_token = "test_token"
    full_download = True

    kwargs_in = {
        "data_source": data_source,
        "frequency": frequency,
        "subset": subset,
        "save_dir": save_dir,
        "download": download,
        "force_remake": force_remake,
        "max_subset_wait_time": max_subset_wait_time,
        "full_download": full_download,
    }

    if data_source == "test":
        with pytest.raises(ValueError):
            climdata.get_climate_data(**kwargs_in)
        return

    to_patch = f"climepi.climdata._base.{data_source.upper()}DataGetter"
    with patch(to_patch, autospec=True) as mock_data_getter:
        result = climdata.get_climate_data(**kwargs_in)
        if data_source in ["lens2", "arise"]:
            mock_data_getter.assert_called_once_with(
                frequency=frequency,
                subset=subset,
                save_dir=save_dir,
            )
        elif data_source == "glens":
            mock_data_getter.assert_called_once_with(
                frequency=frequency,
                subset=subset,
                save_dir=save_dir,
                full_download=full_download,
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
    """Test the get_climate_data_file_names function."""
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
    """Test the _get_data_getter function."""
    frequency = "hourly"
    subset = {
        "scenarios": ["overcast", "sunny"],
        "models": ["length", "inswinger"],
        "realizations": [1, 2],
        "years": [2015, 2016, 2018, 2100],
        "locations": "gabba",
        "lon": 153,
        "lat": -27,
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
