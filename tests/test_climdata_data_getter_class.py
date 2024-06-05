"""
Unit tests for the ClimateDataGetter class in the _data_getter_class module of the
climdata subpackage.
"""

import itertools
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from climepi.climdata._data_getter_class import CACHE_DIR, ClimateDataGetter


def test_init():
    """
    Test the __init__ method of the ClimateDataGetter class.
    """
    data_getter = ClimateDataGetter(
        frequency="daily",
        subset={"models": ["googly"], "location": "gabba"},
    )
    for attr, value in (
        ("data_source", None),
        ("remote_open_possible", False),
        ("available_years", None),
        ("available_scenarios", None),
        ("available_models", None),
        ("available_realizations", None),
        ("lon_res", None),
        ("lat_res", None),
        ("_frequency", "daily"),
        (
            "_subset",
            {
                "years": None,
                "scenarios": None,
                "models": ["googly"],
                "realizations": None,
                "location": "gabba",
                "lon_range": None,
                "lat_range": None,
            },
        ),
        ("_temp_save_dir", CACHE_DIR / "temp"),
        ("_temp_file_names", None),
        ("_ds_temp", None),
        ("_save_dir", CACHE_DIR),
        ("_file_name_dict", None),
        ("_file_names", None),
    ):
        assert getattr(data_getter, attr) == value, f"Attribute {attr} is not {value}."


@pytest.mark.parametrize(
    "years,year_str_expected,warning",
    [
        ([2015], "2015_to_2015", False),
        ([2015, 2016], "2015_to_2016", False),
        (np.arange(2015, 2200), "2015_to_2199", False),
        ([2015, 2016, 2018, 2100], "2015_2016_2018_2100", False),
        (np.arange(2015, 2200, step=2), "2015_by_2_to_2199", False),
        (
            np.append(np.arange(2015, 2026), 3000),
            "_".join([str(y) for y in np.append(np.arange(2015, 2026), 3000)]),
            True,
        ),
    ],
)
@pytest.mark.parametrize(
    "location,lon_range,lat_range,loc_str_expected",
    [
        ("gabba", None, None, "gabba"),
        ("Trent Bridge", [8, 15], None, "Trent_Bridge"),
        (None, [8, 15], None, "lon_8_to_15"),
        (None, None, [7, 44], "lat_7_to_44"),
        (None, [7, 72], [6, 17], "lon_7_to_72_lat_6_to_17"),
    ],
)
def test_file_name_dict(
    years, year_str_expected, warning, location, lon_range, lat_range, loc_str_expected
):
    """
    Test the _file_name_dict method of the ClimateDataGetter class.
    """
    scenarios = ["overcast", "sunny"]
    models = ["length", "inswinger", "bouncer"]
    realizations = np.arange(1, 3)
    data_getter = ClimateDataGetter(
        subset={
            "scenarios": scenarios,
            "models": models,
            "realizations": realizations,
            "years": years,
            "location": location,
            "lon_range": lon_range,
            "lat_range": lat_range,
        },
    )
    data_getter.data_source = "broad"
    assert data_getter._file_name_dict is None
    if warning:
        with pytest.warns(UserWarning):
            result = data_getter.file_name_dict
    else:
        result = data_getter.file_name_dict
    assert data_getter._file_name_dict == result
    for scenario, model, realization in itertools.product(
        scenarios, models, realizations
    ):
        file_name_result = result[scenario][model][realization]
        file_name_expected = (
            f"broad_monthly_{year_str_expected}_{loc_str_expected}"
            + f"_{scenario}_{model}_{realization}.nc"
        )
        assert file_name_result == file_name_expected, (
            f"File name is {file_name_result}, expected filename is "
            f"{file_name_expected}."
        )


def test_file_names():
    """
    Test the file_names method of the ClimateDataGetter class.
    """
    data_getter = ClimateDataGetter(
        frequency="monthly",
        subset={
            "scenarios": ["overcast", "sunny"],
            "models": ["length", "inswinger"],
            "realizations": np.arange(1, 3),
            "years": [2015, 2016, 2018, 2100],
            "location": "gabba",
        },
    )
    data_getter.data_source = "broad"
    assert data_getter._file_names is None
    result = data_getter.file_names
    assert data_getter._file_names == result
    expected = [
        "broad_monthly_2015_2016_2018_2100_gabba_" + comb + ".nc"
        for comb in [
            "overcast_length_1",
            "overcast_length_2",
            "overcast_inswinger_1",
            "overcast_inswinger_2",
            "sunny_length_1",
            "sunny_length_2",
            "sunny_inswinger_1",
            "sunny_inswinger_2",
        ]
    ]
    assert sorted(result) == sorted(expected)


class TestGetData:
    """
    Class to test the get_data method of the ClimateDataGetter class.
    """

    subset = {
        "years": [2015],
        "scenarios": ["overcast"],
        "models": ["length"],
        "realizations": [1],
    }
    data_source = "warner"

    def test_get_data_already_downloaded(self):
        """
        Test the get_data method of the ClimateDataGetter class retrieved locally
        available data if already downloaded (_open_local_data is tested in detail
        separately).
        """
        data_getter = ClimateDataGetter(subset=self.subset)
        data_getter.data_source = self.data_source

        def _side_effect():
            data_getter._ds = ["flipper"]

        data_getter._open_local_data = MagicMock(side_effect=_side_effect)
        result = data_getter.get_data()
        assert result == ["flipper"]
        assert data_getter._ds == ["flipper"]

    def test_get_data_download(self):
        """
        Test the get_data method of the ClimateDataGetter works correctly when data is
        not already downloaded.
        """
        data_getter = ClimateDataGetter(subset=self.subset)
        data_getter.data_source = self.data_source

        def _open_local_data_side_effect():
            if data_getter._ds is None:
                raise FileNotFoundError

        def _open_temp_data_side_effect():
            data_getter._ds_temp = ["googly"]
            data_getter._ds = ["googly"]

        def _process_data_side_effect():
            data_getter._ds = ["flipper"]

        data_getter._open_local_data = MagicMock(
            side_effect=_open_local_data_side_effect
        )
        data_getter._find_remote_data = MagicMock()
        data_getter._subset_remote_data = MagicMock()
        data_getter._download_remote_data = MagicMock()
        data_getter._open_temp_data = MagicMock(side_effect=_open_temp_data_side_effect)
        data_getter._process_data = MagicMock(side_effect=_process_data_side_effect)
        data_getter._save_processed_data = MagicMock()
        data_getter._delete_temporary = MagicMock()
        result = data_getter.get_data(download=True)
        assert result == ["flipper"]
        assert data_getter._ds == ["flipper"]
