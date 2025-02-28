"""Unit tests for the _app_construction module of the app subpackage."""

import time
from unittest.mock import patch

import dask
import panel as pn
import pytest
from playwright.sync_api import expect

import climepi.app._app_construction as app_construction
from climepi.testing.fixtures import generate_dataset

dask.config.set(scheduler="synchronous")  # enforce synchronous scheduler

PORT = [6000]


@pytest.fixture
def port():
    """
    Return a port number.

    (Taken from https://panel.holoviz.org/how_to/test/uitests.html)
    """
    PORT[0] += 1
    return PORT[0]


@pytest.fixture(autouse=True)
def server_cleanup():
    """
    Clean up server state after each test.

    (Taken from https://panel.holoviz.org/how_to/test/uitests.html)
    """
    try:
        yield
    finally:
        pn.state.reset()


@patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
@patch("climepi.app._app_classes_methods.climdata.get_example_dataset", autospec=True)
@patch("climepi.app._app_classes_methods.climdata.EXAMPLE_NAMES", ["data"])
@patch.dict(
    "climepi.app._app_classes_methods.climdata.EXAMPLES",
    {"data": {}},
)
@patch(
    "climepi.app._app_classes_methods.epimod.EXAMPLE_NAMES",
    ["model"],
)
@patch.dict(
    "climepi.app._app_classes_methods.epimod.EXAMPLES",
    {"model": {}},
)
def test_run_app(mock_get_example_dataset, _, capsys, port, page):
    """
    Test the run_app function.

    Uses the Playwright library to test the app.
    """
    ds = generate_dataset(
        data_var="temperature", frequency="monthly", extra_dims={"realization": 2}
    ).climepi.sel_geo("SCG")
    mock_get_example_dataset.return_value = ds

    server = app_construction.run_app(port=port, threaded=True, show=False)
    time.sleep(0.1)
    captured = capsys.readouterr()
    assert "Setting up the app" in captured.out
    assert "Using the Dask single-machine scheduler" in captured.out
    assert "Set-up complete. Press Ctrl+C to stop the app" in captured.out

    page.goto(f"http://localhost:{port}/climepi_app")
    page.get_by_role("button", name="Load data").wait_for()
    time.sleep(0.1)
    captured = capsys.readouterr()
    assert "Session created" in captured.out
    expect(page).to_have_title("climepi app")

    page.get_by_role("button", name="Load data").click()
    page.get_by_text("Data loaded").wait_for()
    mock_get_example_dataset.assert_called_once_with("data", base_dir=None)

    server.stop()
    time.sleep(0.1)
    captured = capsys.readouterr()
    assert "Cleaning up sessions" in captured.out
    assert "Session cleaned up successfully (deleted temporary file(s))" in captured.out
    assert "Stopping app" in captured.out
