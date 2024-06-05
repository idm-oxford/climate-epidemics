"""
Unit tests for the ClimateDataGetter class in the _data_getter_class module of the
climdata subpackage.
"""

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
