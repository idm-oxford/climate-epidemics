"""
Unit tests for the _examples.py module of the epimod subpackage.
"""

from unittest.mock import patch

import pooch
import pytest

from climepi import climdata


@patch.dict(
    climdata._examples.EXAMPLES,
    {
        "shot": {
            "data_source": "bat",
            "frequency": "ball",
            "subset": "cover",
            "formatted_data_downloadable": True,
        },
        "leave": {
            "data_source": "bat",
            "frequency": "ball",
            "subset": "cover",
        },
    },
    clear=True,
)
@patch.object(climdata, "get_climate_data", return_value="not real")
@patch.object(climdata._examples, "_fetch_formatted_example_dataset")
@pytest.mark.parametrize("name", ["shot", "leave"])
@pytest.mark.parametrize("force_remake", [False, True])
def test_get_example_dataset(
    mock_fetch_formatted_example_dataset, mock_get_climate_data, name, force_remake
):
    """
    Test the get_example_dataset method.
    """
    data_dir = "not/a/real/dir"
    ds = climdata.get_example_dataset(
        name, data_dir=data_dir, force_remake=force_remake
    )
    assert ds == "not real"
    if name != "shot" or force_remake:
        mock_fetch_formatted_example_dataset.assert_not_called()
    else:
        mock_fetch_formatted_example_dataset.assert_called_once_with(name, data_dir)
    mock_get_climate_data.assert_called_once_with(
        data_source="bat",
        frequency="ball",
        subset="cover",
        save_dir=data_dir,
        download=True,
        force_remake=force_remake,
    )


@patch.dict(
    climdata._examples.EXAMPLES,
    {
        "leave": {
            "data_source": "bat",
            "frequency": "ball",
            "subset": "cover",
            "formatted_data_downloadable": False,
        },
    },
    clear=True,
)
def test_get_example_details():
    """
    Test the _get_example_details method.
    """
    example_details = climdata._examples._get_example_details("leave")
    assert example_details == climdata._examples.EXAMPLES["leave"]
    with pytest.raises(ValueError, match="Available examples are"):
        climdata._examples._get_example_details("googly")


def test_get_data_dir():
    """
    Test the _get_data_dir method.
    """
    assert (
        climdata._examples._get_data_dir("leave", "not/a/real/dir") == "not/a/real/dir"
    )
    with patch("pathlib.Path.exists", return_value=False):
        assert climdata._examples._get_data_dir("leave", None) == pooch.os_cache(
            "climepi/examples/leave"
        )


@patch.dict(
    climdata._examples.EXAMPLES,
    {
        "leave": {
            "data_source": "lens2",
            "frequency": "ball",
            "subset": {
                "locations": ["cover"],
                "years": [2015],
                "scenarios": ["full"],
                "models": ["root"],
                "realizations": [1, 2],
            },
            "formatted_data_downloadable": False,
        },
    },
    clear=True,
)
@patch("pooch.core.Pooch", autospec=True)
def test_fetch_formatted_example_dataset(mock_pooch):
    """
    Test the _fetch_formatted_example_dataset method.
    """

    name = "leave"
    data_dir = "not/a/real/dir"

    climdata._examples._fetch_formatted_example_dataset(name, data_dir)
    mock_pooch.assert_called_once()
    assert mock_pooch.call_args.kwargs["base_url"] == (
        "https://github.com/will-s-hart/climate-epidemics/"
        + "raw/main/data/examples/leave/"
    )
    assert str(mock_pooch.call_args.kwargs["path"]) == data_dir
    mock_pooch.return_value.load_registry.assert_called_once()
    registry_path_used = mock_pooch.return_value.load_registry.call_args.args[0]
    assert registry_path_used.parent.is_dir()
    assert str(registry_path_used).endswith(
        "climepi/climdata/_example_registry_files/leave.txt"
    )
    assert mock_pooch.return_value.fetch.call_count == 2
    mock_pooch.return_value.fetch.assert_any_call(
        "lens2_ball_2015_cover_full_root_1.nc"
    )
    mock_pooch.return_value.fetch.assert_any_call(
        "lens2_ball_2015_cover_full_root_2.nc"
    )


@patch.dict(
    climdata._examples.EXAMPLES,
    {
        "leave": {"on": "length"},
    },
    clear=True,
)
def test_get_registry_file_path():
    """
    Test the _get_registry_file_path method.
    """
    result = climdata._examples._get_registry_file_path("leave")
    assert str(result).endswith("climepi/climdata/_example_registry_files/leave.txt")


@patch.object(pooch, "make_registry")
def test_make_example_registry(mock_pooch_make_registry):
    """
    Test the _make_example_registry method.
    """
    name = "leave"
    data_dir = "not/a/real/dir"
    climdata._examples._make_example_registry(name, data_dir=data_dir)
    mock_pooch_make_registry.assert_called_once()
    assert str(mock_pooch_make_registry.call_args.args[0]) == data_dir
    assert str(mock_pooch_make_registry.call_args.args[1]).endswith(
        "climepi/climdata/_example_registry_files/leave.txt"
    )
    assert not mock_pooch_make_registry.call_args.kwargs["recursive"]


@patch.dict(climdata._examples.EXAMPLES, {"shot": {}, "leave": {}}, clear=True)
@patch.object(climdata._examples, "get_example_dataset")
@patch.object(climdata._examples, "_make_example_registry")
@pytest.mark.parametrize("force_remake", [False, True])
def test_make_all_examples(
    mock_make_example_registry, mock_get_example_dataset, force_remake
):
    """
    Test the make_all_examples method.
    """

    def _mock_get_example_dataset_side_effect(_name, **kwargs):
        # Mock attempt to download the "leave" example dataset timing out
        if _name == "leave" and kwargs["force_remake"]:
            raise TimeoutError

    mock_get_example_dataset.side_effect = _mock_get_example_dataset_side_effect
    if force_remake:
        with pytest.raises(TimeoutError):
            climdata._examples.make_all_examples(force_remake=force_remake)
        mock_make_example_registry.assert_called_once_with("shot")
    else:
        climdata._examples.make_all_examples(force_remake=force_remake)
        assert mock_make_example_registry.call_count == 2
        mock_make_example_registry.assert_any_call("shot")
        mock_make_example_registry.assert_any_call("leave")
    assert mock_get_example_dataset.call_count == 2
    mock_get_example_dataset.assert_any_call("shot", force_remake=force_remake)
    mock_get_example_dataset.assert_any_call("leave", force_remake=force_remake)