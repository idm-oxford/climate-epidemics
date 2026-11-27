"""Regression tests for choosing, retrieving and using a climate subset."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from climepi.app import _climate_download as download


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
    """Available models and years follow the current source combination."""
    assert set(dialog._sources) >= {"lens2", "arise", "glens", "isimip"}
    assert (
        "https://registry.opendata.aws/ncar-cesm2-lens/" in dialog._source_info.object
    )
    dialog._source.value = "isimip"
    dialog._models.value = ["canesm5"]
    dialog._scenarios.value = ["ssp126", "ssp245"]
    assert "canesm5" not in dialog._models.options
    assert dialog._models.value == ["gfdl-esm4"]

    dialog._source.value = "arise"
    assert (dialog._start.start, dialog._end.end) == (2015, 2100)
    dialog._realizations.value = [0, 8]
    assert dialog._end.end == 2069
    dialog._scenarios.value = ["ssp245", "sai15"]
    assert (dialog._start.start, dialog._end.end) == (2035, 2069)

    dialog._source.value = "glens"
    assert "https://rda.ucar.edu/datasets/d651064/" in dialog._source_info.object
    dialog._scenarios.value = ["sai", "rcp85"]
    dialog._realizations.value = [7]
    assert (dialog._start.start, dialog._end.end) == (2020, 2030)


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
    monkeypatch.setattr(
        download,
        "geocode",
        lambda _: SimpleNamespace(latitude=51.75, longitude=-1.25),
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
    assert dialog._status.value == "Climate subset ready."
    assert not dialog._prepare.disabled
    assert captured["locations"] == ["Point"]
    assert captured["lats"] == [51.75]
    assert captured["lons"] == [-1.25]
    path = dialog._download.file
    assert path.name == "climepi-lens2-monthly-2020-2024.nc"

    dialog._use_result()
    dataset, label = received[0]
    try:
        assert label == "LENS2 · 2020–2024 · monthly"
        dialog.close()
        assert path.is_file()
        assert dataset.location.item() == location
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
    path = dialog._download.file

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
    previous_file = dialog._download.file
    assert dialog._actions.visible

    dialog._longitude.value = 2
    assert not dialog._actions.visible
    assert not previous_file.exists()

    def fail(self, *, download):
        raise OSError("source unavailable")

    monkeypatch.setattr(download._base.LENS2DataGetter, "get_data", fail)
    _run_async(dialog._retrieve())
    assert "Climate retrieval failed" in dialog._status.value
    assert dialog._download.file is None
    assert not dialog._actions.visible


def test_write_netcdf_keeps_names_longer_than_encoded_width(tmp_path):
    """A stale fixed string width from a multi-file dataset does not truncate."""
    dataset = xr.Dataset(
        {"temperature": ("model", [1.0, 2.0])},
        coords={"model": ["gfdl-esm4", "canesm5"]},
    )
    dataset.model.encoding["dtype"] = np.dtype("<U7")
    download.write_netcdf(dataset, tmp_path / "out.nc")
    with xr.open_dataset(tmp_path / "out.nc") as reopened:
        assert reopened.model.values.tolist() == ["gfdl-esm4", "canesm5"]
    assert dataset.model.encoding["dtype"] == np.dtype("<U7")


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
    assert dialog._download.file is None
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
    path = dialog._download.file

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
