"""Unit tests for the _app_classes_methods module of the app subpackage."""

import pathlib
from unittest.mock import patch

import numpy as np
import pytest
import xarray.testing as xrt

import climepi.app._app_classes_methods as app_classes_methods
from climepi import epimod
from climepi.testing.fixtures import generate_dataset


@patch("climepi.app._app_classes_methods.climdata.get_example_dataset", autospec=True)
def test_load_clim_data_func(mock_get_example_dataset):
    """Unit test for the _load_clim_data_func function."""
    mock_get_example_dataset.return_value = "mocked_dataset"
    result = app_classes_methods._load_clim_data_func("some_example_name", "some/dir")
    mock_get_example_dataset.assert_called_once_with(
        "some_example_name", base_dir="some/dir"
    )
    assert result == "mocked_dataset"
    # Check cached version is returned if the same example_name and base_dir are
    # provided
    mock_get_example_dataset.return_value = "another_mocked_dataset"
    result_cached = app_classes_methods._load_clim_data_func(
        "some_example_name", "some/dir"
    )
    assert result_cached == "mocked_dataset"
    mock_get_example_dataset.assert_called_once()


@patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
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


@patch("climepi.app._app_classes_methods._compute_to_file_reopen", autospec=True)
@patch.object(pathlib.Path, "unlink", autospec=True)
def test_run_epi_model_func(mock_path_unlink, mock_compute_to_file_reopen):
    """Unit test for the _run_epi_model_func function."""

    def _mock_compute_to_file_reopen(ds_in, save_path):
        return ds_in

    mock_compute_to_file_reopen.side_effect = _mock_compute_to_file_reopen

    ds_clim = generate_dataset(data_var="temperature", frequency="monthly")
    ds_clim["temperature"].values = 30 * np.random.rand(*ds_clim["temperature"].shape)
    epi_model = epimod.SuitabilityModel(temperature_range=(15, 30))

    result = app_classes_methods._run_epi_model_func(
        ds_clim,
        epi_model,
        return_yearly_portion_suitable=True,
        save_path=pathlib.Path("some/dir/ds_out.nc"),
    )
    xrt.assert_identical(
        result, epi_model.run(ds_clim, return_yearly_portion_suitable=True)
    )
    assert mock_compute_to_file_reopen.call_count == 2
    mock_path_unlink.assert_called_once_with(pathlib.Path("some/dir/ds_suitability.nc"))


@pytest.mark.parametrize("temporal", ["daily", "monthly", "yearly"])
@pytest.mark.parametrize("spatial", ["single", "list", "grid"])
@pytest.mark.parametrize("ensemble", ["single", "multiple"])
@pytest.mark.parametrize("scenario", ["single", "multiple"])
@pytest.mark.parametrize("model", ["single", "multiple"])
def test_get_scope_dict(temporal, spatial, ensemble, scenario, model):
    """Unit test for the _get_scope_dict function."""
    ds = generate_dataset(
        data_var="temperature",
        frequency=temporal,
        extra_dims={"realization": 3, "scenario": 2, "model": 2},
    )
    if spatial == "single":
        ds = ds.isel(lat=0, lon=0)
    elif spatial == "list":
        ds = ds.climepi.sel_geo(["lords", "gabba"])
    if ensemble == "single":
        ds = ds.isel(realization=0, drop=True)
    if scenario == "single":
        ds = ds.isel(scenario=0)
    if model == "single":
        ds = ds.isel(model=[0])
    result = app_classes_methods._get_scope_dict(ds)
    expected = {
        "temporal": temporal,
        "spatial": spatial,
        "ensemble": ensemble,
        "scenario": scenario,
        "model": model,
    }
    assert result == expected
