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
    Test the run_app method.

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

    assert len(pn.state.cache["controllers"]) == 1
    server.stop()
    time.sleep(0.1)
    assert "controllers" not in pn.state.cache
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


def test_get_logger(capsys):
    """Unit test for the get_logger method."""
    logger = app_construction.get_logger("test")
    logger.info("Test message")
    captured = capsys.readouterr()
    assert "INFO | test | Test message" in captured.out
    cached_logger = app_construction.get_logger("test")
    assert cached_logger is logger


@patch("climepi.app._app_construction._layout", autospec=True)
@patch("climepi.app._app_construction.pn", autospec=True)
def test_session(mock_panel, mock_layout):
    """Unit test for the _session method."""
    mock_panel.state = MagicMock()
    mock_panel.state.curdoc.session_context.id = "an_id"
    mock_panel.state.cache = {}
    mock_layout.return_value = ("a template", "a controller")
    template = app_construction._session(
        clim_dataset_example_base_dir=None,
        clim_dataset_example_names=["data"],
        enable_custom_clim_dataset=True,
        custom_clim_data_dir=None,
        epi_model_example_names=["model"],
        enable_custom_epi_model=False,
    )
    assert template == "a template"
    assert mock_panel.state.cache["controllers"]["an_id"] == "a controller"
    mock_panel.state.on_session_destroyed.assert_called_once_with(
        app_construction._session_destroyed
    )


@patch("climepi.app._app_construction.Client", autospec=True)
def test_setup_dask(mock_client, capsys):
    """Unit test for the _setup_dask method."""
    app_construction._setup_dask(dask_distributed=False)
    mock_client.assert_not_called()
    assert "dask_client" not in pn.state.cache
    assert "Using the Dask single-machine scheduler" in capsys.readouterr().out

    app_construction._setup_dask(dask_distributed=True)
    mock_client.assert_called_once_with("tcp://127.0.0.1:64719")
    assert pn.state.cache["dask_client"] == mock_client.return_value
    assert "Client connected to Dask local cluster" in capsys.readouterr().out

    mock_client.side_effect = OSError()
    with pytest.raises(OSError, match="Error connecting to Dask local cluster"):
        app_construction._setup_dask(dask_distributed=True)


@patch("climepi.app._app_classes_methods.epimod.get_example_model", autospec=True)
def test_layout(_):
    """Unit test for the _layout method."""
    template, controller = app_construction._layout(
        clim_dataset_example_base_dir=None,
        clim_dataset_example_names=None,
        enable_custom_clim_dataset=True,
        custom_clim_data_dir=None,
        epi_model_example_names=None,
        enable_custom_epi_model=False,
    )
    assert isinstance(template, pn.template.BootstrapTemplate)
    assert isinstance(controller, app_classes_methods.Controller)
    assert template.sidebar[0] == controller.data_controls


def test_cleanup_session(capsys):
    """Unit test for the _cleanup_session method."""
    mock_controller = MagicMock()
    pn.state.cache["controllers"] = {"an_id": mock_controller}
    app_construction._cleanup_session("an_id")
    mock_controller.cleanup_temp_file.assert_called_once()
    captured = capsys.readouterr()
    assert "Session cleaned up successfully (deleted temporary file(s))" in captured.out


def test_session_destroyed(capsys):
    """Unit test for the _session_destroyed method."""
    mock_controller = MagicMock()
    pn.state.cache["controllers"] = {"an_id": mock_controller}
    session_context = MagicMock()
    session_context.id = "an_id"
    app_construction._session_destroyed(session_context)
    mock_controller.cleanup_temp_file.assert_called_once()
    captured = capsys.readouterr()
    assert "Session cleaned up successfully (deleted temporary file(s))" in captured.out


def test_shutdown(capsys):
    """Unit test for the _shutdown method."""
    mock_controller1 = MagicMock()
    mock_controller2 = MagicMock()
    pn.state.cache["controllers"] = {"id1": mock_controller1, "id2": mock_controller2}
    mock_client = MagicMock()
    mock_client.return_value.futures = "some futures"
    pn.state.cache["dask_client"] = mock_client
    app_construction._shutdown()
    mock_controller1.cleanup_temp_file.assert_called_once()
    mock_controller2.cleanup_temp_file.assert_called_once()
    mock_client.cancel.assert_called_once_with(mock_client.futures, force=True)
    mock_client.close.assert_called_once()
    captured = capsys.readouterr()
    assert captured.out.count("Session cleaned up successfully") == 2
    assert "Closing Dask client" in captured.out
    assert "Stopping app" in captured.out
    assert pn.state.cache == {}


@patch("climepi.app._app_construction.signal", autospec=True)
@patch("climepi.app._app_construction._shutdown", autospec=True)
def test_set_shutdown(mock_shutdown, mock_signal):
    """Unit test for the _set_shutdown method."""
    server = MagicMock()
    server.is_alive.return_value = True

    def _original_stop():
        server.is_alive.return_value = False
        return "stopped"

    server.stop.side_effect = _original_stop

    mock_original_sigint = MagicMock()
    mock_original_sigterm = MagicMock()

    def _getsignal(signalnum):
        if signalnum == mock_signal.SIGINT:
            return mock_original_sigint
        if signalnum == mock_signal.SIGTERM:
            return mock_original_sigterm
        raise ValueError(f"Unexpected argument {signalnum} passed to signal.getsignal")

    mock_signal.getsignal.side_effect = _getsignal

    def _signal(signalnum, handler):
        if signalnum == mock_signal.SIGINT:
            mock_signal.SIGINT = handler
        elif signalnum == mock_signal.SIGTERM:
            mock_signal.SIGTERM = handler
        else:
            raise ValueError(f"Unexpected argument {signalnum} passed to signal.signal")

    mock_signal.signal.side_effect = _signal

    app_construction._set_shutdown(server, threaded=True)
    assert server.is_alive()
    out = server.stop()
    assert out == "stopped"
    assert not server.is_alive()
    mock_shutdown.assert_called_once()

    out2 = server.stop()
    assert out2 == "stopped"
    mock_shutdown.assert_called_once()

    mock_signal.SIGINT("some signum", "some frame")
    mock_original_sigint.assert_called_once_with("some signum", "some frame")
    assert mock_shutdown.call_count == 2
    mock_signal.SIGTERM("another signum", "another frame")
    mock_original_sigterm.assert_called_once_with("another signum", "another frame")
    assert mock_shutdown.call_count == 3
