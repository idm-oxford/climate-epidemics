"""Unit tests for opening climate data files saved on the app's machine."""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from climepi.app import _local_files as local_files
from climepi.testing.fixtures import generate_dataset


def _run_async(coroutine):
    # Playwright keeps its own loop running in the main thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


@pytest.fixture
def dialog():
    """Return a local files dialog that records what it hands over."""
    received = []
    result = local_files.LocalFiles(
        lambda opener, name: received.append((opener, name))
    )
    result.received = received
    return result


@pytest.fixture
def folder(tmp_path):
    """Return a folder of NetCDF files, with the dataset they hold."""
    ds = generate_dataset(data_var=["temperature", "precipitation"])
    bounds = ["time_bnds", "lat_bnds", "lon_bnds"]
    path = tmp_path / "london"
    path.mkdir()
    ds[["temperature", *bounds]].to_netcdf(path / "temperature.nc")
    ds[["precipitation", *bounds]].to_netcdf(path / "precipitation.nc")
    (path / "README.txt").write_text("Not climate data")
    return path, ds


def test_open_netcdf_combines_files(tmp_path):
    """Test that open_netcdf combines files and reports when it is closed."""
    ds = generate_dataset(data_var=["temperature", "precipitation"])
    paths = [tmp_path / "temperature.nc", tmp_path / "precipitation.nc"]
    bounds = ["time_bnds", "lat_bnds", "lon_bnds"]
    ds[["temperature", *bounds]].to_netcdf(paths[0])
    ds[["precipitation", *bounds]].to_netcdf(paths[1])
    closed = []
    result = local_files.open_netcdf(paths, on_close=lambda: closed.append(True))
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
    result = local_files.open_netcdf(["a.nc"])
    assert "dtype" not in result["model"].encoding
    mock_open_mfdataset.assert_called_once_with(
        ["a.nc"], chunks={}, data_vars="minimal", coords="minimal", compat="override"
    )
    result.close()


def test_open_netcdf_failure_closes_files(tmp_path):
    """Test that open_netcdf closes the files if the bounds cannot be read."""
    path = tmp_path / "data.nc"
    generate_dataset().to_netcdf(path)
    closed = []
    with (
        patch.object(xr.DataArray, "load", side_effect=OSError("unreadable")),
        pytest.raises(OSError, match="unreadable"),
    ):
        local_files.open_netcdf([path], on_close=lambda: closed.append(True))
    assert not closed
    path.unlink()  # possible on Windows only once the file is closed


@pytest.mark.parametrize("folders", [False, True])
def test_dialog_command_macos(folders):
    """Test the AppleScript dialog that lists chosen paths on macOS."""
    with patch.object(local_files.sys, "platform", "darwin"):
        command = local_files._dialog_command(folders)
    assert command[0] == "osascript"
    assert command[1::2] == ["-e"] * (len(command) // 2)
    script = command[2::2]
    assert script[0] == "activate"
    assert script[1].startswith(
        "set chosen to choose folder" if folders else "set chosen to choose file"
    )
    assert script[1].endswith("with multiple selections allowed")
    assert ('of type {"nc", "nc4"}' in script[1]) is not folders
    assert script[-1] == "return chosenPaths"


@pytest.mark.parametrize(
    "folders,dialog_class", [(False, "OpenFileDialog"), (True, "FolderBrowserDialog")]
)
def test_dialog_command_windows(folders, dialog_class):
    """Test the PowerShell dialog that lists chosen paths on Windows."""
    with patch.object(local_files.sys, "platform", "win32"):
        command = local_files._dialog_command(folders)
    assert command[:4] == ["powershell", "-NoProfile", "-STA", "-EncodedCommand"]
    script = base64.b64decode(command[4]).decode("utf-16-le")
    assert "[Text.Encoding]::UTF8" in script
    assert f"System.Windows.Forms.{dialog_class}" in script
    assert "ShowDialog($owner)" in script


def test_dialog_command_linux():
    """Test the zenity dialog that lists chosen paths on other systems."""
    with (
        patch.object(local_files.sys, "platform", "linux"),
        patch.object(local_files.shutil, "which", return_value="/usr/bin/zenity"),
    ):
        files = local_files._dialog_command(False)
        folders = local_files._dialog_command(True)
    with (
        patch.object(local_files.sys, "platform", "linux"),
        patch.object(local_files.shutil, "which", return_value=None),
    ):
        missing = local_files._dialog_command(False)
    start = ["/usr/bin/zenity", "--file-selection", "--multiple", "--separator=\n"]
    assert files[:4] == start
    assert "--file-filter=NetCDF files | *.nc *.nc4" in files
    assert "--directory" not in files
    assert folders[:4] == start
    assert "--directory" in folders
    assert missing[0] == "zenity"


def test_choose_paths():
    """Test that the paths printed by the file dialog are returned."""
    with patch.object(local_files.subprocess, "run", autospec=True) as mock_run:
        mock_run.return_value.stdout = "/data/a.nc\n/my data/b.nc\n\n"
        paths = local_files._choose_paths(True)
    assert paths == [Path("/data/a.nc"), Path("/my data/b.nc")]
    mock_run.assert_called_once_with(
        local_files._dialog_command(True),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_netcdf_files(tmp_path):
    """Test that chosen files and folders give NetCDF files, after checks."""
    for name in ["b.nc", "A.NC4", ".hidden.nc"]:
        generate_dataset().to_netcdf(tmp_path / name)
    (tmp_path / "notes.txt").write_text("Not climate data")
    (tmp_path / "folder.nc").mkdir()
    assert local_files._netcdf_files([tmp_path]) == [
        tmp_path / "A.NC4",
        tmp_path / "b.nc",
    ]
    assert local_files._netcdf_files([tmp_path / "b.nc"]) == [tmp_path / "b.nc"]

    with pytest.raises(ValueError, match="There are no NetCDF \\(.nc\\) files in"):
        local_files._netcdf_files([tmp_path / "folder.nc"])
    with pytest.raises(ValueError, match="There is no file or folder at"):
        local_files._netcdf_files([tmp_path / "missing.nc"])
    (tmp_path / "notes.nc").write_text("Not climate data")
    with pytest.raises(ValueError, match="notes.nc is not a NetCDF file"):
        local_files._netcdf_files([tmp_path])


@pytest.mark.parametrize(
    "paths,expected",
    [
        (["/data/a.nc"], "a.nc"),
        (["/data/london"], "london"),
        (["/"], str(Path("/"))),
        (["/data/a.nc", "/data/b.nc"], "a.nc and b.nc"),
        (["/data/a.nc", "/data/b.nc", "/data/c.nc"], "a.nc and 2 more"),
    ],
)
def test_data_name(paths, expected):
    """Test that the data are named after the chosen files or folders."""
    assert local_files._data_name([Path(path) for path in paths]) == expected


def test_local_files_choose_folder(dialog, folder):
    """Test that a folder chosen in the file dialog is opened where it is."""
    path, ds = folder
    during = []

    def choose(folders):
        during.append((folders, dialog.choosing, dialog._choose_files_button.disabled))
        return [path]

    dialog.modal.show()
    with patch.object(local_files, "_choose_paths", side_effect=choose):
        _run_async(dialog._choose_folders())
    assert during == [(True, True, True)]
    assert not dialog.choosing
    assert not dialog._choose_files_button.disabled
    assert not dialog.modal.open
    assert dialog._status.value == ""
    [(opener, name)] = dialog.received
    assert name == "london"
    dataset = opener()
    xrt.assert_allclose(dataset.compute(), ds)
    dataset.close()
    assert sorted(child.name for child in path.iterdir()) == [
        "README.txt",
        "precipitation.nc",
        "temperature.nc",
    ]


def test_local_files_choose_files(dialog, folder):
    """Test that files chosen in the file dialog are opened, or nothing if none."""
    path, _ds = folder
    files = [path / "temperature.nc", path / "precipitation.nc"]
    with patch.object(local_files, "_choose_paths", return_value=[]) as mock_choose:
        _run_async(dialog._choose_files())
    mock_choose.assert_called_once_with(False)
    assert dialog._status.value == ""
    assert not dialog.received

    with patch.object(local_files, "_choose_paths", return_value=files):
        _run_async(dialog._choose_files())
    [(_opener, name)] = dialog.received
    assert name == "temperature.nc and precipitation.nc"


def test_local_files_no_dialog(dialog):
    """Test that the dialog suggests entering a path if no file dialog shows."""
    with patch.object(
        local_files, "_choose_paths", side_effect=FileNotFoundError("zenity")
    ):
        _run_async(dialog._choose_files())
    assert not dialog.choosing
    assert dialog._status.value == (
        "The app could not show a file dialog on this computer. Enter a path instead."
    )
    assert dialog._status.tone == "warning"


def test_local_files_typed_path(dialog, folder, monkeypatch):
    """Test opening a path entered by the user, as copied from a file manager."""
    path, _ds = folder
    dialog._open_typed()
    assert dialog._status.value == "Enter the path of a NetCDF file or folder."
    assert dialog._status.tone == "warning"

    dialog._path.value_input = "missing"
    dialog._path.param.trigger("enter_pressed")
    assert dialog._status.value.startswith("There is no file or folder at")
    assert not dialog.received

    monkeypatch.chdir(path.parent)
    dialog.modal.show()
    dialog._path.value_input = ' "london" '
    dialog._open_typed()
    assert not dialog.modal.open
    assert dialog._status.value == ""
    [(_opener, name)] = dialog.received
    assert name == "london"


def test_local_files_refused():
    """Test that files the app cannot take now are explained in the dialog."""

    def refuse(_opener, _name):
        raise ValueError("Wait for the current operation to finish and try again.")

    dialog = local_files.LocalFiles(refuse, busy=False)
    with patch.object(local_files, "_netcdf_files", return_value=[]):
        dialog.modal.show()
        dialog._open([Path("data.nc")])
    assert dialog.modal.open
    assert dialog._status.value == (
        "Wait for the current operation to finish and try again."
    )
