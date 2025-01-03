"""Unit tests for the _app_classes_methods module of the app subpackage."""

from unittest.mock import patch

import pytest

import climepi.app._app_classes_methods as app_classes_methods
from climepi import epimod


@patch("climepi.app._app_classes_methods.climdata.get_example_dataset")
def test_load_clim_data_func(mock_get_example_dataset):
    """
    Unit test for the _load_clim_data_func function.

    This function is currently a thin wrapper around the climdata.get_example_dataset
    function. This test therefore simply checks that get_example_dataset is called
    with the correct arguments and that the return value is passed through.
    """
    mock_get_example_dataset.return_value = "mocked_dataset"
    result = app_classes_methods._load_clim_data_func("some_example_name", "some/dir")
    mock_get_example_dataset.assert_called_once_with(
        "some_example_name", base_dir="some/dir"
    )
    assert result == "mocked_dataset"


@patch("climepi.app._app_classes_methods.epimod.get_example_model")
def test_get_epi_model_func(mock_get_example_model):
    """Unit test for the _get_epi_model_func function."""
    # Test with example_name provided
    mock_get_example_model.return_value = "mocked_model"
    result_named = app_classes_methods._get_epi_model_func(
        example_name="some_example_name"
    )
    mock_get_example_model.assert_called_once_with("some_example_name")
    assert result_named == "mocked_model"
    # Test with temperature_range provided
    result_temp_range = app_classes_methods._get_epi_model_func(
        temperature_range=(15, 30)
    )
    assert isinstance(result_temp_range, epimod.SuitabilityModel)
    assert result_temp_range.temperature_range == (15, 30)
    # Check error if either both or neither of example_name and temperature_range are
    # provided
    with pytest.raises(
        ValueError,
        match="Exactly one of example_name and temperature_range must be provided",
    ):
        app_classes_methods._get_epi_model_func(
            example_name="another_name", temperature_range=(0, 10)
        )
    with pytest.raises(
        ValueError,
        match="Exactly one of example_name and temperature_range must be provided",
    ):
        app_classes_methods._get_epi_model_func()
