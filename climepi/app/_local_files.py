"""Opening climate data files saved on the machine running a local app session."""

import asyncio
import base64
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import panel as pn
import param
import xarray as xr

from climepi.app._app_ui import MODAL_CSS, StatusMessage
from climepi.climdata._utils import _drop_string_width_encodings

# How NetCDF classic and NetCDF-4 (HDF5) files begin.
_NETCDF_SIGNATURES = (b"CDF", b"\x89HDF\r\n\x1a\n")
_NETCDF_SUFFIXES = (".nc", ".nc4")
_FILES_TITLE = "Choose NetCDF files"
_FOLDERS_TITLE = "Choose folders of NetCDF files"
# Windows dialogs are created in PowerShell, kept in front of the browser.
_WINDOWS_DIALOG_START = """
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$owner = New-Object System.Windows.Forms.Form -Property @{ TopMost = $true }
"""
_WINDOWS_FILES_DIALOG = f"""
$dialog = New-Object System.Windows.Forms.OpenFileDialog -Property @{{
    Multiselect = $true
    Title = '{_FILES_TITLE}'
    Filter = 'NetCDF files (*.nc, *.nc4)|*.nc;*.nc4|All files (*.*)|*.*'
}}
if ($dialog.ShowDialog($owner) -eq 'OK') {{ $dialog.FileNames }}
"""
_WINDOWS_FOLDER_DIALOG = """
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{
    Description = 'Choose a folder of NetCDF files'
}
if ($dialog.ShowDialog($owner) -eq 'OK') { $dialog.SelectedPath }
"""


def open_netcdf(
    paths: Sequence[Path], on_close: Callable[[], None] | None = None
) -> xr.Dataset:
    """
    Open NetCDF files lazily as one dataset.

    Parameters
    ----------
    paths : sequence of pathlib.Path
        Files to open. Several files are combined by their coordinates.
    on_close : callable, optional
        Called once the returned dataset is closed (not if opening fails).

    Returns
    -------
    xarray.Dataset
        The combined dataset.
    """
    source = xr.open_mfdataset(
        paths, chunks={}, data_vars="minimal", coords="minimal", compat="override"
    )
    try:
        # Results written from the data would otherwise truncate longer names.
        _drop_string_width_encodings(source)
        # Bounds must be in memory when results are written to file.
        if "time_bnds" in source:
            source.time_bnds.load()
        dataset = source.copy(deep=False)
    except BaseException:
        source.close()
        raise

    def close() -> None:
        try:
            source.close()
        finally:
            if on_close is not None:
                on_close()

    # xarray runs this at most once per dataset.
    dataset.set_close(close)
    return dataset


def _dialog_command(folders: bool) -> list[str]:
    # Return a command showing the system's dialog for choosing files or folders,
    # which prints each chosen path on its own line.
    if sys.platform == "darwin":
        choose = (
            f'choose folder with prompt "{_FOLDERS_TITLE}"'
            if folders
            else f'choose file with prompt "{_FILES_TITLE}" of type {{"nc", "nc4"}}'
        )
        script = [
            "activate",
            f"set chosen to {choose} with multiple selections allowed",
            'set chosenPaths to ""',
            "repeat with chosenItem in chosen",
            "set chosenPaths to chosenPaths & POSIX path of chosenItem & linefeed",
            "end repeat",
            "return chosenPaths",
        ]
        return ["osascript", *(arg for line in script for arg in ("-e", line))]
    if sys.platform == "win32":
        script = _WINDOWS_DIALOG_START + (
            _WINDOWS_FOLDER_DIALOG if folders else _WINDOWS_FILES_DIALOG
        )
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        return ["powershell", "-NoProfile", "-STA", "-EncodedCommand", encoded]
    command = [
        shutil.which("zenity") or "zenity",
        "--file-selection",
        "--multiple",
        "--separator=\n",
    ]
    if folders:
        return [*command, "--directory", f"--title={_FOLDERS_TITLE}"]
    return [
        *command,
        f"--title={_FILES_TITLE}",
        "--file-filter=NetCDF files | *.nc *.nc4",
        "--file-filter=All files | *",
    ]


def _choose_paths(folders: bool) -> list[Path]:
    # Wait for the user to choose files or folders, returning none if cancelled.
    result = subprocess.run(
        _dialog_command(folders),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def _netcdf_files(paths: Iterable[Path]) -> list[Path]:
    # Collect chosen files, and the NetCDF files directly in chosen folders, after
    # checking that each is a NetCDF file.
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(
                (
                    child
                    for child in path.iterdir()
                    if child.suffix.lower() in _NETCDF_SUFFIXES
                    and not child.name.startswith(".")
                    and child.is_file()
                ),
                key=lambda child: child.name.casefold(),
            )
            if not found:
                raise ValueError(
                    f"There are no NetCDF (.nc) files in {path.name or path}."
                )
            files.extend(found)
        elif path.exists():
            files.append(path)
        else:
            raise ValueError(f"There is no file or folder at {path}.")
    for file in files:
        with file.open("rb") as stream:
            if not stream.read(8).startswith(_NETCDF_SIGNATURES):
                raise ValueError(f"{file.name} is not a NetCDF file.")
    return files


def _data_name(paths: Sequence[Path]) -> str:
    # Name the data after the chosen files or folders.
    names = [path.name or str(path) for path in paths]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]} and {len(names) - 1} more"


class LocalFiles(param.Parameterized):
    """Build the local app's dialog for opening NetCDF files where they are saved."""

    choosing = param.Boolean(default=False, doc="Whether a file dialog is open.")

    def __init__(
        self,
        on_open: Callable[[Callable[[], xr.Dataset], str], None],
        *,
        busy: Any = False,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._on_open = on_open
        unavailable = pn.bind(
            lambda busy, choosing: busy or choosing, busy, self.param.choosing
        )
        self._choose_files_button = pn.widgets.Button(
            label="Choose files",
            icon="file",
            color="primary",
            variant="outline",
            margin=(5, 5, 5, 10),
            disabled=unavailable,
            on_click=self._choose_files,
        )
        self._choose_folder_button = pn.widgets.Button(
            label="Choose folder",
            icon="folder",
            color="primary",
            variant="outline",
            margin=5,
            disabled=unavailable,
            on_click=self._choose_folders,
        )
        self._path = pn.widgets.TextInput(
            placeholder="Or enter the path of a file or folder",
            sizing_mode="stretch_width",
        )
        self._path.param.watch(self._open_typed, "enter_pressed")
        self._open_path = pn.widgets.Button(
            label="Open",
            color="primary",
            variant="outline",
            disabled=unavailable,
            on_click=self._open_typed,
        )
        self._status = StatusMessage(
            tone="neutral", margin=(6, 5, 0, 5), sizing_mode="stretch_width"
        )
        content = pn.Column(
            pn.pane.Markdown("## Open local files"),
            pn.pane.Markdown(
                "Choose NetCDF (.nc) files, or a folder containing them, on this "
                "computer. Several files are combined into one dataset by their "
                "coordinates, such as one file per variable or period. The data need "
                "time, latitude and longitude coordinates.\n\n"
                "The app reads the files where they are, so nothing is copied or "
                "changed. Files saved with **Retrieve online data** can be opened here."
            ),
            pn.FlexBox(
                self._choose_files_button, self._choose_folder_button, gap="0.75rem"
            ),
            pn.Row(self._path, self._open_path, sizing_mode="stretch_width"),
            self._status,
            sizing_mode="stretch_width",
        )
        self.modal = pn.Modal(content, stylesheets=[MODAL_CSS])
        self.button = self.modal.create_button(
            "show",
            label="Open local files",
            color="primary",
            variant="outline",
            icon="folder-open",
            margin=(6, 0),
            sizing_mode="stretch_width",
        )

    def _set_status(self, message: str, tone: str) -> None:
        self._status.param.update(value=message, tone=tone)

    async def _choose_files(self, _event: Any = None) -> None:
        # Open NetCDF files chosen in the system's file dialog.
        await self._choose(folders=False)

    async def _choose_folders(self, _event: Any = None) -> None:
        # Open the NetCDF files in folders chosen in the system's file dialog.
        await self._choose(folders=True)

    async def _choose(self, *, folders: bool) -> None:
        # The dialog belongs to the app's process, which runs on this computer, so
        # it gives the files' paths where a browser would give only their contents.
        self.choosing = True
        self._set_status(
            "Choose in the file dialog, which may open behind this window.", "info"
        )
        try:
            paths = await asyncio.to_thread(_choose_paths, folders)
        except OSError:
            self._set_status(
                "The app could not show a file dialog on this computer. Enter a "
                "path instead.",
                "warning",
            )
            return
        finally:
            self.choosing = False
        if paths:
            self._open(paths)
        else:
            self._set_status("", "neutral")

    def _open_typed(self, _event: Any = None) -> None:
        # Open the file or folder at the path entered, as copied from a file manager.
        text = (self._path.value_input or "").strip().strip("\"'")
        if not text:
            self._set_status("Enter the path of a NetCDF file or folder.", "warning")
            return
        self._open([Path(text).expanduser().absolute()])

    def _open(self, paths: list[Path]) -> None:
        # Check the chosen files, then hand them to the app to open in the background.
        try:
            files = _netcdf_files(paths)
            self._on_open(partial(open_netcdf, files), _data_name(paths))
        except (OSError, ValueError) as exc:
            self._set_status(str(exc), "warning")
            return
        self._set_status("", "neutral")
        self.modal.hide()
