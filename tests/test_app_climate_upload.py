"""Unit tests for uploading climate data files in the app."""

import asyncio
import socket
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest
import tornado.escape
import tornado.httpclient
import tornado.httpserver
import tornado.testing
import tornado.web
import xarray as xr
import xarray.testing as xrt

from climepi.app import _climate_upload as upload
from climepi.testing.fixtures import generate_dataset

_ROUTE = rf"/{upload.UPLOAD_ROUTE}/([\w-]+)"


def _run_async(coroutine):
    # Playwright keeps its own loop running in the main thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def _serve_uploads(check):
    # Run a coroutine check against an upload server and HTTP client.
    async def run():
        app = tornado.web.Application([(_ROUTE, upload.UploadHandler)])
        sock, port = tornado.testing.bind_unused_port()
        server = tornado.httpserver.HTTPServer(app)
        server.add_sockets([sock])
        client = tornado.httpclient.AsyncHTTPClient()
        try:
            await check(client, f"http://127.0.0.1:{port}", port)
        finally:
            server.stop()

    _run_async(run())


@pytest.fixture
def folder():
    """Return an upload folder, closed after the test."""
    result = upload._UploadFolder()
    yield result
    result.close()


@pytest.fixture
def dialog():
    """Return an upload dialog that records what it hands over."""
    received = []
    result = upload.ClimateUpload(lambda opener, name: received.append((opener, name)))
    result.received = received
    yield result
    result.close()


def _write_upload(folder, data, name):
    # Save a file as the upload handler would and return its ID.
    path = folder.path / f"{uuid4().hex}.nc"
    path.write_bytes(data)
    return folder.add(path, name)


def test_open_netcdf_combines_files(tmp_path):
    """Test that open_netcdf combines files and reports when it is closed."""
    ds = generate_dataset(data_var=["temperature", "precipitation"])
    paths = [tmp_path / "temperature.nc", tmp_path / "precipitation.nc"]
    bounds = ["time_bnds", "lat_bnds", "lon_bnds"]
    ds[["temperature", *bounds]].to_netcdf(paths[0])
    ds[["precipitation", *bounds]].to_netcdf(paths[1])
    closed = []
    result = upload.open_netcdf(paths, on_close=lambda: closed.append(True))
    try:
        xrt.assert_allclose(result.compute(), ds)
        assert not result.time_bnds.chunks  # loaded for writing results
        assert result.temperature.chunks
    finally:
        result.close()
        result.close()
    assert closed == [True]


@patch.object(xr, "open_mfdataset", autospec=True)
def test_open_netcdf_drops_string_widths(mock_open_mfdataset):
    """Test that open_netcdf keeps longer names from being truncated when saved."""
    source = xr.Dataset(coords={"model": ("model", ["gfdl-esm4"])})
    source["model"].encoding["dtype"] = np.dtype("<U9")
    mock_open_mfdataset.return_value = source
    result = upload.open_netcdf(["a.nc"], on_close=lambda: None)
    assert "dtype" not in result["model"].encoding
    mock_open_mfdataset.assert_called_once_with(
        ["a.nc"], chunks={}, data_vars="minimal", coords="minimal", compat="override"
    )


def test_open_netcdf_failure_closes_files(tmp_path):
    """Test that open_netcdf closes the files if the bounds cannot be read."""
    path = tmp_path / "data.nc"
    generate_dataset().to_netcdf(path)
    closed = []
    with (
        patch.object(xr.DataArray, "load", side_effect=OSError("unreadable")),
        pytest.raises(OSError, match="unreadable"),
    ):
        upload.open_netcdf([path], on_close=lambda: closed.append(True))
    assert not closed
    path.unlink()  # possible on Windows only once the file is closed


def test_open_uploads(tmp_path):
    """Test that uploaded files are checked, and deleted once no longer needed."""
    good, bad = tmp_path / "good.nc", tmp_path / "bad.nc"
    generate_dataset().to_netcdf(good)
    bad.write_bytes(b"not a NetCDF file")
    with pytest.raises(ValueError, match="folder/bad.nc is not a NetCDF file"):
        upload._open_uploads([good, bad], ["folder/good.nc", "folder/bad.nc"])
    assert not good.exists() and not bad.exists()

    generate_dataset().to_netcdf(good)
    dataset = upload._open_uploads([good], ["good.nc"])
    assert good.exists()
    dataset.close()
    assert not good.exists()


@pytest.mark.parametrize(
    "names,expected",
    [
        (["data.nc"], "data.nc"),
        (["folder/data.nc"], "data.nc"),
        (["folder/a.nc", "folder/sub/b.nc"], "folder"),
        (["a.nc", "b.nc"], "2 files"),
        (["one/a.nc", "two/b.nc"], "2 files"),
    ],
)
def test_upload_name(names, expected):
    """Test that uploaded data are named after the chosen file or folder."""
    assert upload._upload_name(names) == expected


def test_upload_folder(folder):
    """Test that an upload folder hands over requested files and deletes others."""
    assert folder.url == f"climepi_upload/{folder.token}"
    assert upload._UPLOADS[folder.token] is folder
    first = _write_upload(folder, b"1", "a.nc")
    second = _write_upload(folder, b"2", "b.nc")
    paths, names = folder.take([second])
    assert names == ["b.nc"]
    assert [path.read_bytes() for path in paths] == [b"2"]
    assert list(folder.path.iterdir()) == paths

    # Missing uploads are reported and nothing is kept.
    third = _write_upload(folder, b"3", "c.nc")
    with pytest.raises(ValueError, match="did not finish uploading"):
        folder.take([third, first])
    with pytest.raises(ValueError, match="did not finish uploading"):
        folder.take([])
    assert list(folder.path.iterdir()) == paths

    _write_upload(folder, b"4", "d.nc")
    folder.discard()
    assert list(folder.path.iterdir()) == paths

    folder.close()
    assert folder.token not in upload._UPLOADS
    assert not folder.path.exists()
    folder.path.mkdir()
    assert _write_upload(folder, b"5", "e.nc") is None
    folder.close()


def test_upload_handler_saves_files(folder):
    """Test that the upload handler saves files and replies with their IDs."""
    data = bytes(range(256)) * 5000  # arrives in several pieces

    async def check(client, base, _port):
        response = await client.fetch(
            f"{base}/{folder.url}?name=folder%2Fdata.nc", method="POST", body=data
        )
        file_id = tornado.escape.json_decode(response.body)["id"]
        paths, names = folder.take([file_id])
        assert names == ["folder/data.nc"]
        assert paths[0].read_bytes() == data

        long_name = "x" * 300
        response = await client.fetch(
            f"{base}/{folder.url}?name={long_name}", method="POST", body=b"1"
        )
        file_id = tornado.escape.json_decode(response.body)["id"]
        assert folder.take([file_id])[1] == ["x" * 255]

    _serve_uploads(check)


def test_upload_handler_refusals(folder):
    """Test that the upload handler refuses unexpected uploads and keeps no files."""

    async def check(client, base, _port):
        async def status(url, **kwargs):
            kwargs = {"method": "POST", "body": b"data", **kwargs}
            response = await client.fetch(url, raise_error=False, **kwargs)
            return response.code

        assert await status(f"{base}/{upload.UPLOAD_ROUTE}/unknown") == 404
        assert await status(f"{base}/{folder.url}", method="GET", body=None) == 405

        async def chunks(write):
            await write(b"data")

        chunked = await status(f"{base}/{folder.url}", body=None, body_producer=chunks)
        assert chunked == 411
        with patch.object(upload.shutil, "disk_usage") as disk_usage:
            disk_usage.return_value.free = upload._FREE_SPACE_MARGIN + 3
            assert await status(f"{base}/{folder.url}") == 507

        # A session that ends during an upload does not keep the file.
        original = upload.UploadHandler.data_received

        def close_session(handler, chunk):
            original(handler, chunk)
            upload._UPLOADS.pop(folder.token)

        with patch.object(upload.UploadHandler, "data_received", close_session):
            assert await status(f"{base}/{folder.url}") == 404

    _serve_uploads(check)
    assert not list(folder.path.iterdir())


def test_upload_handler_abandoned_upload(folder):
    """Test that the upload handler deletes the file of an abandoned upload."""

    async def check(_client, _base, port):
        with socket.create_connection(("127.0.0.1", port)) as connection:
            connection.sendall(
                f"POST /{folder.url} HTTP/1.1\r\nHost: localhost\r\n"
                "Content-Length: 1000\r\n\r\npartial".encode()
            )
            for _ in range(100):
                await asyncio.sleep(0.01)
                if list(folder.path.iterdir()):
                    break
            assert len(list(folder.path.iterdir())) == 1
        for _ in range(100):
            await asyncio.sleep(0.01)
            if not list(folder.path.iterdir()):
                break

    _serve_uploads(check)
    assert not list(folder.path.iterdir())


def test_file_picker_messages():
    """Test that the file picker passes messages from the browser to its owner."""
    received = []
    picker = upload.FilePicker(received.append, url="climepi_upload/token")
    picker._handle_msg({"type": "start", "count": 2})
    assert received == [{"type": "start", "count": 2}]


def test_upload_dialog_progress(dialog):
    """Test that the upload dialog reports what the browser is doing."""
    for message, status, tone in [
        ({"type": "start", "count": 1}, "Uploading 1 file…", "info"),
        ({"type": "start", "count": 3}, "Uploading 3 files…", "info"),
        ({"type": "empty"}, "No NetCDF (.nc) files were chosen.", "warning"),
        ("not a message", "No NetCDF (.nc) files were chosen.", "warning"),
        ({"type": "other"}, "No NetCDF (.nc) files were chosen.", "warning"),
    ]:
        dialog._handle_picker(message)
        assert (dialog._status.value, dialog._status.tone) == (status, tone)

    _write_upload(dialog._folder, b"1", "a.nc")
    dialog._handle_picker({"type": "error", "message": "Upload of b.nc failed."})
    assert (dialog._status.value, dialog._status.tone) == (
        "Upload of b.nc failed.",
        "danger",
    )
    assert not list(dialog._folder.path.iterdir())
    assert not dialog.received


def test_upload_dialog_uses_files(dialog, tmp_path):
    """Test that the upload dialog hands finished uploads over to the app."""
    ds = generate_dataset()
    ds.to_netcdf(tmp_path / "data.nc")
    file_id = _write_upload(
        dialog._folder, (tmp_path / "data.nc").read_bytes(), "folder/data.nc"
    )
    dialog.modal.show()
    dialog._handle_picker({"type": "start", "count": 1})
    dialog._handle_picker({"type": "done", "files": [file_id]})
    assert not dialog.modal.open
    assert dialog._status.value == ""
    [(opener, name)] = dialog.received
    assert name == "data.nc"
    dataset = opener()
    xrt.assert_allclose(dataset.compute(), ds)
    dataset.close()
    assert not list(dialog._folder.path.iterdir())


@pytest.mark.parametrize("files", [None, [1], ["unknown"]])
def test_upload_dialog_incomplete_upload(dialog, files):
    """Test that the upload dialog explains an upload that did not finish."""
    _write_upload(dialog._folder, b"1", "a.nc")
    dialog._handle_picker({"type": "done", "files": files})
    assert dialog._status.value == "Some files did not finish uploading. Try again."
    assert dialog._status.tone == "danger"
    assert not list(dialog._folder.path.iterdir())
    assert not dialog.received


def test_upload_dialog_refused_files():
    """Test that files the app cannot take now are deleted and the reason shown."""

    def refuse(_opener, _name):
        raise ValueError("Wait for the current operation to finish and try again.")

    dialog = upload.ClimateUpload(refuse)
    try:
        file_id = _write_upload(dialog._folder, b"1", "a.nc")
        dialog.modal.show()
        dialog._handle_picker({"type": "done", "files": [file_id]})
        assert dialog.modal.open
        assert dialog._status.value == (
            "Could not use these files: Wait for the current operation to finish and "
            "try again."
        )
        assert not list(dialog._folder.path.iterdir())
    finally:
        dialog.close()
    assert not dialog._folder.path.exists()
