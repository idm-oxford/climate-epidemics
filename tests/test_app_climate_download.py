"""Regression tests for choosing, retrieving and using a climate subset."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import panel as pn
import pytest
import tornado.httpclient
import tornado.httpserver
import tornado.testing
import tornado.web
import xarray as xr

from climepi.app import _climate_download as download
from climepi.climdata._data_getter_class import ClimateDataGetter


def _run_async(coroutine):
    # Playwright keeps its own loop running in the main thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


@pytest.fixture
def dialog():
    """Create and release a local climate download dialog."""
    picker = download.ClimateDownload(lambda *_: None)
    yield picker
    picker.close()


def test_source_selection_and_year_constraints(dialog):
    """Available scenarios and years follow the current source combination."""
    assert list(dialog._source.options) == [
        "LENS2",
        "ARISE-SAI",
        "GLENS",
        "ISIMIP3b",
        "ISIMIP3a",
        "ISIMIP4a",
        "ISIMIP4b",
        "ERA5",
    ]
    assert (
        "https://registry.opendata.aws/ncar-cesm2-lens/" in dialog._source_info.object
    )
    dialog._source.value = "isimip"
    assert dialog._scenarios.options == ["ssp126", "ssp370", "ssp585", "ssp245"]
    assert len(dialog._models.options) == 10

    dialog._source.value = "isimip4b"
    dialog._scenarios.value = ["esm-hist"]
    assert (dialog._start.start, dialog._end.end) == (1850, 2021)
    dialog._start.value, dialog._end.value = 2000, 2001
    dialog._longitude.value, dialog._latitude.value = 1, 52
    dialog._scenarios.value = ["esm-hist", "esm-scen7-vl"]
    assert (dialog._start.value, dialog._end.value) == (2000, 2001)
    with pytest.raises(ValueError, match="no years in common"):
        dialog._request()

    dialog._source.value = "era5"
    assert dialog._scenarios.options == ["historical"]
    assert dialog._models.value == ["era5"]
    assert dialog._realizations.value == [0]

    dialog._source.value = "arise"
    assert (dialog._start.start, dialog._end.end) == (2015, 2100)
    dialog._realizations.value = [0, 8]
    assert dialog._end.end == 2069
    dialog._scenarios.value = ["ssp245", "sai15"]
    assert (dialog._start.start, dialog._end.end) == (2035, 2069)

    dialog._source.value = "glens"
    assert "https://rda.ucar.edu/datasets/d651064/" in dialog._source_info.object
    dialog._scenarios.value = ["sai"]
    assert (dialog._start.start, dialog._end.end) == (2020, 2099)
    dialog._scenarios.value = ["sai", "rcp85"]
    dialog._realizations.value = [7]
    assert (dialog._start.start, dialog._end.end) == (2020, 2030)


def test_unlisted_source_details():
    """A data source without display details is shown under its identifier."""

    class _NewDataGetter(ClimateDataGetter):
        data_source = "new"

    assert download._source_details(_NewDataGetter).name == "NEW"
    assert download._source_information(_NewDataGetter) == ""


def test_climate_geometry_validation(dialog):
    """Check required coordinates and preserve the requested bounding box."""
    with pytest.raises(ValueError, match="latitude"):
        dialog._request()
    dialog._longitude.value = -1.25
    dialog._latitude.value = 51.75
    assert dialog._request().subset["locations"] == ["Point"]
    assert dialog._request().subset["lons"] == [-1.25]

    dialog._geometry.value = "box"
    dialog._west.value = 10
    dialog._east.value = -5
    dialog._south.value = 49
    dialog._north.value = 58
    with pytest.raises(ValueError, match="west below east"):
        dialog._request()
    dialog._west.value = -15
    assert dialog._request().subset["lon_range"] == (-15, -5)
    assert dialog._request().subset["lat_range"] == (49, 58)

    dialog._west.value = -180
    dialog._east.value = 180
    assert "lon_range" not in dialog._request().subset


def test_retrieve_named_subset_and_keep_adopted_file(dialog, monkeypatch):
    """Export the requested data and retain its file while used by the app."""
    provider_started = threading.Event()
    finish_provider = threading.Event()
    captured = {}
    location = "../../Oxford, United Kingdom"

    def get_data(self, *, download):
        captured.update(self._subset)
        assert not download
        provider_started.set()
        assert finish_provider.wait(10)
        return xr.Dataset(
            {"temperature": ("location", [15.2])}, coords={"location": ["Point"]}
        )

    monkeypatch.setattr(download._base.LENS2DataGetter, "get_data", get_data)
    address = "Oxford, Oxfordshire, England, United Kingdom"
    monkeypatch.setattr(
        download,
        "geocode",
        lambda _: SimpleNamespace(latitude=51.75, longitude=-1.25, address=address),
    )
    dialog._geometry.value = "place"
    dialog._place.value = location
    received = []
    dialog._on_loaded = lambda dataset, label: received.append((dataset, label))

    async def retrieve():
        task = asyncio.create_task(dialog._retrieve())
        try:
            assert await asyncio.to_thread(provider_started.wait, 10)
            assert dialog._status.value == "Retrieving climate data…"
            assert dialog._prepare.disabled
            await dialog._retrieve()
        finally:
            finish_provider.set()
            await task

    _run_async(retrieve())
    # The address found is shown so the user can check it is the intended place.
    assert dialog._status.value == f"Climate subset ready. Location found: {address}."
    assert not dialog._prepare.disabled
    assert captured["locations"] == ["Point"]
    assert captured["lats"] == [51.75]
    assert captured["lons"] == [-1.25]
    assert dialog._result is not None
    path = dialog._result.path
    assert path.name == "climepi-lens2-monthly-2020-2024.nc"
    assert f'href="{dialog._result.url}"' in dialog._download.object

    dialog._use_result()
    dataset, label = received[0]
    try:
        assert label == "LENS2 · 2020–2024 · monthly"
        dialog.close()
        assert path.is_file()
        assert dataset.location.item() == location
        assert dataset.location.attrs["geocoded_address"] == address
        assert float(dataset.temperature.compute().item()) == 15.2
    finally:
        dataset.close()
        dataset.close()  # a repeated close must not release the file twice
    assert not path.parent.exists()


def test_retry_after_provider_failure_and_retire_unused_subset(dialog, monkeypatch):
    """Allow a provider retry and release the file when an adoption is rejected."""

    def time_out(self, *, download):
        assert download
        raise TimeoutError("source still processing")

    monkeypatch.setattr(download._base.ISIMIPDataGetter, "get_data", time_out)
    dialog._source.value = "isimip"
    dialog._longitude.value = 1
    dialog._latitude.value = 52
    _run_async(dialog._retrieve())
    assert "Try the same request again" in dialog._status.value
    assert not dialog._prepare.disabled

    def successful_retry(self, *, download):
        return xr.Dataset({"temperature": ("time", [10.0])})

    monkeypatch.setattr(download._base.ISIMIPDataGetter, "get_data", successful_retry)
    _run_async(dialog._retrieve())
    assert dialog._status.value == "Climate subset ready."
    assert dialog._result is not None
    path = dialog._result.path

    def reject(dataset, label):
        raise RuntimeError("other job active")

    dialog._on_loaded = reject
    dialog._use_result()
    assert "Could not load this subset" in dialog._status.value
    dialog.close()
    assert not path.parent.exists()


def test_editing_selection_invalidates_previous_download(dialog, monkeypatch):
    """A failed replacement request cannot export the old selection."""
    monkeypatch.setattr(
        download._base.LENS2DataGetter,
        "get_data",
        lambda self, *, download: xr.Dataset({"temperature": ("time", [10.0])}),
    )
    dialog._longitude.value = 1
    dialog._latitude.value = 52
    _run_async(dialog._retrieve())
    assert dialog._result is not None
    previous_file = dialog._result.path
    assert dialog._actions.visible

    dialog._longitude.value = 2
    assert not dialog._actions.visible
    assert not previous_file.exists()

    def fail(self, *, download):
        raise OSError("source unavailable")

    monkeypatch.setattr(download._base.LENS2DataGetter, "get_data", fail)
    _run_async(dialog._retrieve())
    assert "Climate retrieval failed" in dialog._status.value
    assert dialog._result is None
    assert dialog._download.object == ""
    assert not dialog._actions.visible


def test_oversized_subset_is_refused(dialog, monkeypatch):
    """A request above the size limit reports the size instead of writing it."""
    monkeypatch.setattr(download, "_MAX_SUBSET_BYTES", 8)
    monkeypatch.setattr(
        download._base.LENS2DataGetter,
        "get_data",
        lambda self, *, download: xr.Dataset({"temperature": ("time", [1.0, 2.0])}),
    )
    dialog._longitude.value = 1
    dialog._latitude.value = 52
    _run_async(dialog._retrieve())
    assert "above the 0 GB limit" in dialog._status.value
    assert dialog._status.tone == "warning"
    assert dialog._result is None
    assert not dialog._actions.visible


def test_use_result_explains_rejected_subset(dialog, monkeypatch):
    """The app's reason for rejecting a subset is shown to the user."""
    monkeypatch.setattr(
        download._base.LENS2DataGetter,
        "get_data",
        lambda self, *, download: xr.Dataset({"temperature": ("time", [10.0])}),
    )
    dialog._longitude.value = 1
    dialog._latitude.value = 52
    _run_async(dialog._retrieve())
    assert dialog._result is not None
    path = dialog._result.path

    def busy(dataset, label):
        raise ValueError("Wait for the current operation to finish and try again.")

    dialog._on_loaded = busy
    dialog._use_result()
    assert dialog._status.value == (
        "Could not use this subset in the app: "
        "Wait for the current operation to finish and try again."
    )
    dialog.close()
    assert not path.parent.exists()


def test_download_handler_streams_until_retired():
    """Retrieved files are streamed over HTTP until they are retired."""
    directory = TemporaryDirectory()
    path = Path(directory.name) / "subset.nc"
    path.write_bytes(bytes(range(256)) * 5000)  # larger than one streamed piece
    result = download._Download(directory, path, "subset")
    route = rf"/{download.DOWNLOAD_ROUTE}/([\w-]+)"

    async def fetch():
        app = tornado.web.Application([(route, download.DownloadHandler)])
        sock, port = tornado.testing.bind_unused_port()
        server = tornado.httpserver.HTTPServer(app)
        server.add_sockets([sock])
        client = tornado.httpclient.AsyncHTTPClient()
        base = f"http://127.0.0.1:{port}"
        try:
            response = await client.fetch(f"{base}/{result.url}")
            assert response.body == path.read_bytes()
            assert response.headers["Content-Disposition"] == (
                'attachment; filename="subset.nc"'
            )
            result.retire()
            retired = await client.fetch(f"{base}/{result.url}", raise_error=False)
            assert retired.code == 404
        finally:
            server.stop()

    _run_async(fetch())
    assert not path.exists()


def test_download_open_file_failure_releases_file():
    """A file that cannot be opened does not keep its directory in use."""
    directory = TemporaryDirectory()
    path = Path(directory.name) / "subset.nc"
    result = download._Download(directory, path, "subset")
    with pytest.raises(FileNotFoundError):
        result.open_file()
    result.retire()
    assert result.open_file() is None
    assert not Path(directory.name).exists()


def test_choice_group_toggle():
    """Test selecting and clearing every option of a choice group."""
    members = pn.widgets.CheckBoxGroup(label="Members", options=[0, 1, 2], value=[0])
    toggle = download._choice_group(members, "members")[0][2]
    assert toggle.label == "Select all members"
    toggle.clicks += 1
    assert members.value == [0, 1, 2]
    assert toggle.label == "Clear all members"
    toggle.clicks += 1
    assert members.value == []


def test_request_validation(dialog):
    """Test that incomplete requests are explained before anything is retrieved."""
    dialog._latitude.value, dialog._longitude.value = 52, 1
    dialog._scenarios.value = []
    with pytest.raises(ValueError, match="at least one scenario"):
        dialog._request()
    dialog._scenarios.value = ["ssp370"]
    dialog._start.value, dialog._end.value = 2030, 2025
    with pytest.raises(ValueError, match="Select years in order"):
        dialog._request()
    dialog._start.value = 2020
    dialog._geometry.value = "place"
    dialog._place.value = "  "
    with pytest.raises(ValueError, match="Enter a place name"):
        dialog._request()
    _run_async(dialog._retrieve())
    assert dialog._status.value == "Enter a place name."
    assert dialog._status.tone == "warning"
    dialog._use_result()  # nothing has been retrieved yet
    assert dialog._status.value == "Enter a place name."


@pytest.mark.parametrize(
    "geocode_error,data,message",
    [
        (True, None, "Place not found"),
        (False, xr.Dataset({"temperature": ("time", [])}), "contain no data"),
    ],
)
def test_unavailable_subset(dialog, monkeypatch, geocode_error, data, message):
    """Test that a place that cannot be found, or an empty subset, is explained."""

    def geocode(_):
        if geocode_error:
            raise ValueError("Could not geocode query")
        return SimpleNamespace(latitude=0, longitude=0, address="Null Island")

    monkeypatch.setattr(download, "geocode", geocode)
    monkeypatch.setattr(
        download._base.LENS2DataGetter, "get_data", lambda self, *, download: data
    )
    dialog._geometry.value = "place"
    dialog._place.value = "Nowhere"
    _run_async(dialog._retrieve())
    assert message in dialog._status.value
    assert dialog._result is None


def test_downloaded_source_files_removed(dialog, monkeypatch):
    """Test that source files downloaded for a subset are removed afterwards."""
    save_dirs = []

    def get_data(self, *, download):
        assert download
        self._save_dir.mkdir(parents=True)
        save_dirs.append(self._save_dir)
        time = xr.date_range("2020-01-01", periods=2, freq="YS")
        return xr.Dataset(
            {
                "temperature": ("time", [1.0, 2.0]),
                "time_bnds": (("time", "bnds"), [[time[0], time[1]]] * 2),
            },
            coords={"time": time},
        )

    monkeypatch.setattr(download._base.ISIMIPDataGetter, "get_data", get_data)
    dialog._source.value = "isimip"
    dialog._latitude.value, dialog._longitude.value = 52, 1
    _run_async(dialog._retrieve())
    assert not save_dirs[0].exists()
    assert dialog._result is not None
    with dialog._result.open_dataset() as dataset:
        assert isinstance(dataset.time_bnds.data, np.ndarray)


def test_close_during_retrieval(dialog, monkeypatch):
    """Test that a subset retrieved after the session closes is removed."""
    started, release = threading.Event(), threading.Event()
    paths = []
    prepare = download._prepare_download

    def prepare_download(request):
        started.set()
        assert release.wait(10)
        result = prepare(request)
        paths.append(result.path)
        return result

    monkeypatch.setattr(download, "_prepare_download", prepare_download)
    monkeypatch.setattr(
        download._base.LENS2DataGetter,
        "get_data",
        lambda self, *, download: xr.Dataset({"temperature": ("time", [1.0])}),
    )
    dialog._latitude.value, dialog._longitude.value = 52, 1

    async def retrieve():
        task = asyncio.create_task(dialog._retrieve())
        assert await asyncio.to_thread(started.wait, 10)
        dialog.close()
        release.set()
        await task

    _run_async(retrieve())
    assert dialog._result is None
    assert not paths[0].exists()


def test_open_dataset_failure_releases_file(tmp_path):
    """Test that a subset which cannot be read does not keep its file in use."""
    directory = TemporaryDirectory()
    path = Path(directory.name) / "subset.nc"
    time = xr.date_range("2020-01-01", periods=2, freq="YS")
    xr.Dataset(
        {"time_bnds": (("time", "bnds"), [[time[0], time[1]]] * 2)},
        coords={"time": time},
    ).to_netcdf(path)
    result = download._Download(directory, path, "subset")
    with (
        patch.object(xr.DataArray, "load", side_effect=OSError("unreadable")),
        pytest.raises(OSError, match="unreadable"),
    ):
        result.open_dataset()
    result.retire()
    assert not path.exists()
