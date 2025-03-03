"""Unit tests for the _app_construction module of the app subpackage."""

import time
from unittest.mock import MagicMock, patch

import dask
import panel as pn
import pytest
from playwright.sync_api import expect

import climepi.app._app_classes_methods as app_classes_methods
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


@pytest.fixture(autouse=True)
def cache_cleanup():
    """Clean up the cache after each test."""
    pn.state.clear_caches()


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
    Unit test for the run_app method.

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
    captured = capsys.readouterr()
    assert "Session created" in captured.out
    expect(page).to_have_title("climepi app")

    page.get_by_role("button", name="Load data").click()
    page.get_by_text("Data loaded").wait_for()
    mock_get_example_dataset.assert_called_once_with("data", base_dir=None)

    assert pn.state.cache["controllers"] != {}
    server.stop()
    time.sleep(0.1)
    assert pn.state.cache["controllers"] == {}
    captured = capsys.readouterr()
    assert "Cleaning up sessions" in captured.out
    assert "Session cleaned up successfully (deleted temporary file(s))" in captured.out
    assert "Stopping app" in captured.out


@patch("climepi.app._app_construction.pn.serve", autospec=True)
@patch("climepi.app._app_construction.Client", autospec=True)
@pytest.mark.parametrize(
    "start,threaded,dask_distributed",
    [
        (False, False, False),
        (True, False, True),
        (False, True, False),
        (True, True, True),
    ],
)
def test_run_app_opts(
    mock_client, mock_serve, start, threaded, dask_distributed, capsys
):
    """Unit test for the run_app method with different keyword arguments."""
    server = app_construction.run_app(
        start=start, threaded=threaded, dask_distributed=dask_distributed
    )
    mock_serve.assert_called_once()
    mock_serve_kwargs = mock_serve.call_args[1]
    if threaded:
        # serve called without start kwarg (since threaded server always started
        # immediately, and passing start can cause unexpected behavior)
        assert mock_serve_kwargs == {"threaded": True}
        assert server.start.call_count == 0
        assert server.io_loop.start.call_count == 0
    else:
        # serve always called with start=False and server started later if requested
        assert mock_serve_kwargs == {"start": False, "threaded": False}
        assert server.start.call_count == (1 if start else 0)
        assert server.io_loop.start.call_count == (1 if start else 0)
    captured = capsys.readouterr()
    assert "Setting up the app" in captured.out
    assert "Set-up complete. Press Ctrl+C to stop the app" in captured.out
    if dask_distributed:
        assert pn.state.cache["dask_client"] == mock_client.return_value
        assert "Client connected to Dask local cluster" in captured.out
    else:
        assert "dask_client" not in pn.state.cache
        assert "Using the Dask single-machine scheduler" in captured.out
    mock_client.return_value.futures = "some futures"
    server.stop()
    captured = capsys.readouterr()
    assert "Cleaning up sessions" in captured.out
    assert "Stopping app" in captured.out
    if dask_distributed:
        mock_client.return_value.cancel.assert_called_once_with(
            "some futures", force=True
        )
        mock_client.return_value.close.assert_called_once()
        assert "dask_client" not in pn.state.cache
        assert "Closing Dask client" in captured.out
    else:
        mock_client.assert_not_called()
        assert "Closing Dask client" not in captured.out


@patch("climepi.app._app_construction._layout", autospec=True)
@patch("climepi.app._app_construction.pn", autospec=True)
def test_session(mock_panel, mock_layout):
    """Unit test for the _session method."""
    mock_panel.state = MagicMock()
    mock_panel.state.curdoc.session_context.id = "an_id"
    mock_panel.state.cache = {}
    mock_layout.return_value = ("a template", "a controller")
    template = app_construction._session(
        clim_dataset_example_names=["data"], epi_model_example_names=["model"]
    )
    assert template == "a template"
    assert mock_panel.state.cache["controllers"]["an_id"] == "a controller"
    mock_panel.state.on_session_destroyed.assert_called_once_with(
        app_construction._session_destroyed
    )


@patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
def test_layout(_, capsys):
    """Unit test for the _layout method."""
    template, controller = app_construction._layout()
    assert isinstance(template, pn.template.BootstrapTemplate)
    assert isinstance(controller, app_classes_methods.Controller)
    assert template.sidebar[0] == controller.data_controls
