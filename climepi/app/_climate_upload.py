"""Climate data file uploads for local app sessions."""

import secrets
import shutil
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any, BinaryIO, cast

import panel as pn
import param
import tornado.web
import xarray as xr
from tornado.http1connection import HTTP1Connection

from climepi.app._app_ui import MODAL_CSS, OUTLINE_ACTION_CSS, StatusMessage
from climepi.climdata._utils import _drop_string_width_encodings

# Uploaded files are received at this route (see UploadHandler and run_app).
UPLOAD_ROUTE = "climepi_upload"
# Refuse an upload that would leave less free disk space than this.
_FREE_SPACE_MARGIN = 1024**3
# How NetCDF classic and NetCDF-4 (HDF5) files begin.
_NETCDF_SIGNATURES = (b"CDF", b"\x89HDF\r\n\x1a\n")
_PICKER_ESM = """
const NETCDF = /\\.nc4?$/i;
const ICON_START = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
    + 'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    + 'stroke-linejoin="round" aria-hidden="true">';
const FILE_ICON = ICON_START + '<path d="M14 3v4a1 1 0 0 0 1 1h4"/>'
    + '<path d="M17 21h-10a2 2 0 0 1 -2 -2v-14a2 2 0 0 1 2 -2h7l5 5v11a2 2 0 0 1 '
    + '-2 2z"/></svg>';
const FOLDER_ICON = ICON_START + '<path d="M5 4h4l3 3h7a2 2 0 0 1 2 2v8a2 2 0 0 1 '
    + '-2 2h-14a2 2 0 0 1 -2 -2v-11a2 2 0 0 1 2 -2"/></svg>';

// Send one file to the upload route, resolving with the ID the server gives it.
function send(url, file, onProgress) {
    const name = file.webkitRelativePath || file.name;
    return new Promise((resolve, reject) => {
        const request = new XMLHttpRequest();
        request.open('POST', `${url}?name=${encodeURIComponent(name)}`);
        request.responseType = 'json';
        request.upload.addEventListener('progress', (event) => onProgress(event.loaded));
        request.addEventListener('load', () => {
            if (request.status === 200) {
                resolve(request.response.id);
            } else if (request.status === 507) {
                reject(new Error(`There is not enough disk space to upload ${name}.`));
            } else {
                reject(new Error(`Upload of ${name} failed (error ${request.status}).`));
            }
        });
        request.addEventListener('error', () => reject(new Error(
            `Upload of ${name} failed. Check that the app is still running.`
        )));
        request.send(file);
    });
}

export function render({model}) {
    const actions = document.createElement('div');
    actions.className = 'file-picker';
    const progress = document.createElement('progress');
    progress.max = 1;
    progress.hidden = true;
    const buttons = [];
    let uploading = false;

    const update = () => {
        for (const button of buttons) {
            button.disabled = model.disabled || uploading;
        }
    };

    const upload = async (chosen) => {
        const files = chosen.filter((file) => NETCDF.test(file.name));
        if (!files.length) {
            model.send_msg({type: 'empty'});
            return;
        }
        const total = files.reduce((sum, file) => sum + file.size, 0) || 1;
        let sent = 0;
        uploading = true;
        progress.value = 0;
        progress.hidden = false;
        update();
        model.send_msg({type: 'start', count: files.length});
        try {
            const ids = [];
            for (const file of files) {
                ids.push(await send(model.url, file, (loaded) => {
                    progress.value = (sent + loaded) / total;
                }));
                sent += file.size;
            }
            model.send_msg({type: 'done', files: ids});
        } catch (error) {
            model.send_msg({type: 'error', message: error.message});
        } finally {
            uploading = false;
            progress.hidden = true;
            update();
        }
    };

    for (const [label, icon, folder] of [
        ['Choose files', FILE_ICON, false],
        ['Choose folder', FOLDER_ICON, true],
    ]) {
        const input = document.createElement('input');
        input.type = 'file';
        input.multiple = true;
        input.hidden = true;
        if (folder) {
            input.webkitdirectory = true;
        } else {
            input.accept = '.nc,.nc4';
        }
        input.addEventListener('change', () => {
            const files = [...input.files];
            input.value = '';
            upload(files);
        });
        const button = document.createElement('button');
        button.type = 'button';
        button.innerHTML = icon;
        button.append(label);
        button.addEventListener('click', () => input.click());
        buttons.push(button);
        actions.append(button, input);
    }
    update();
    model.on('disabled', update);
    const root = document.createElement('div');
    root.className = 'upload';
    root.append(actions, progress);
    return root;
}
"""
_PICKER_CSS = """
.upload { width: 100%; }
.file-picker {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
}
progress {
    width: 100%;
    margin-top: 0.75rem;
    accent-color: var(--design-primary-color, #1759a6);
}
"""


def open_netcdf(paths: Sequence[Path], on_close: Callable[[], None]) -> xr.Dataset:
    """
    Open NetCDF files lazily as one dataset.

    Parameters
    ----------
    paths : sequence of pathlib.Path
        Files to open. Several files are combined by their coordinates.
    on_close : callable
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
            on_close()

    # xarray runs this at most once per dataset.
    dataset.set_close(close)
    return dataset


def _delete(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _open_uploads(paths: list[Path], names: list[str]) -> xr.Dataset:
    # Open uploaded files, which are deleted once they are no longer needed.
    try:
        for path, name in zip(paths, names, strict=True):
            with path.open("rb") as file:
                if not file.read(8).startswith(_NETCDF_SIGNATURES):
                    raise ValueError(f"{name} is not a NetCDF file.")
        return open_netcdf(paths, on_close=partial(_delete, paths))
    except BaseException:
        _delete(paths)
        raise


def _upload_name(names: list[str]) -> str:
    # Name uploaded data after the chosen file or folder.
    paths = [PurePosixPath(name) for name in names]
    if len(paths) == 1:
        return paths[0].name
    folders = {path.parts[0] for path in paths if len(path.parts) > 1}
    if len(folders) == 1 and all(len(path.parts) > 1 for path in paths):
        return folders.pop()
    return f"{len(paths)} files"


# Upload destinations, by the unguessable token in their URL.
_UPLOADS: dict[str, "_UploadFolder"] = {}


class _UploadFolder:
    """Where a session's uploads are saved until they are opened or discarded."""

    def __init__(self) -> None:
        # Files may still be open on Windows if a session ends mid-operation.
        self._directory = TemporaryDirectory(
            prefix="climepi-upload-", ignore_cleanup_errors=True
        )
        self.path = Path(self._directory.name)
        self.token = secrets.token_urlsafe(16)
        # Finished uploads that are not yet opened: saved path and name in the browser.
        self._files: dict[str, tuple[Path, str]] = {}
        self._lock = Lock()
        _UPLOADS[self.token] = self

    @property
    def url(self) -> str:
        """Return the upload address, relative to the app page."""
        return f"{UPLOAD_ROUTE}/{self.token}"

    def add(self, path: Path, name: str) -> str | None:
        """Keep a finished upload and return its ID, or None once closed."""
        with self._lock:
            if _UPLOADS.get(self.token) is not self:
                return None
            self._files[path.stem] = (path, name)
        return path.stem

    def take(self, ids: list[str]) -> tuple[list[Path], list[str]]:
        """Hand over the finished uploads with these IDs, and delete any others."""
        with self._lock:
            files, self._files = self._files, {}
        chosen = [files.pop(file_id) for file_id in ids if file_id in files]
        _delete(path for path, _ in files.values())
        if not ids or len(chosen) < len(ids):
            _delete(path for path, _ in chosen)
            raise ValueError("Some files did not finish uploading. Try again.")
        return [path for path, _ in chosen], [name for _, name in chosen]

    def discard(self) -> None:
        """Delete finished uploads that were not handed over."""
        with self._lock:
            files, self._files = self._files, {}
        _delete(path for path, _ in files.values())

    def close(self) -> None:
        """Stop accepting uploads and delete any saved files."""
        with self._lock:
            _UPLOADS.pop(self.token, None)
            self._files.clear()
        self._directory.cleanup()


@tornado.web.stream_request_body
class UploadHandler(tornado.web.RequestHandler):
    """Save a climate data file that the browser uploads for a session."""

    SUPPORTED_METHODS = ("POST",)
    _file: BinaryIO | None = None
    _saved = False

    def prepare(self) -> None:
        """Check that the upload is expected and fits, then open its file."""
        folder = _UPLOADS.get(self.path_args[0])
        if folder is None:
            raise tornado.web.HTTPError(404)
        size = int(self.request.headers.get("Content-Length", -1))
        if size < 0:
            raise tornado.web.HTTPError(411)
        if size > shutil.disk_usage(folder.path).free - _FREE_SPACE_MARGIN:
            raise tornado.web.HTTPError(507)
        # The file is written as it arrives, so it may exceed the buffered limit.
        cast(HTTP1Connection, self.request.connection).set_max_body_size(size)
        self._folder = folder
        self._path = folder.path / f"{secrets.token_hex(16)}.nc"
        self._file = self._path.open("wb")

    def data_received(self, chunk: bytes) -> None:
        """Write the next piece of the file."""
        assert self._file is not None
        self._file.write(chunk)

    def post(self, _token: str) -> None:
        """Keep the finished file and reply with its ID."""
        assert self._file is not None
        self._file.close()
        name = self.get_query_argument("name", "")[:255]
        file_id = self._folder.add(self._path, name)
        if file_id is None:
            raise tornado.web.HTTPError(404)
        self._saved = True
        self.write({"id": file_id})

    def on_finish(self) -> None:
        """Delete the file of a refused upload."""
        self._discard()

    def on_connection_close(self) -> None:
        """Delete the file of an upload that the browser abandoned."""
        self._discard()

    def _discard(self) -> None:
        # Remove a partial file, or one that arrived after its session ended.
        if self._file is not None and not self._saved:
            self._file.close()
            self._path.unlink(missing_ok=True)


class FilePicker(pn.custom.JSComponent):
    """Upload chosen NetCDF files, or those in a chosen folder, over HTTP."""

    url = param.String(default="")
    disabled = param.Boolean(default=False, allow_refs=True)

    _esm = _PICKER_ESM
    _stylesheets = [OUTLINE_ACTION_CSS, _PICKER_CSS]

    def __init__(self, on_message: Callable[[Any], None], **params: Any) -> None:
        super().__init__(**params)
        self._on_message = on_message

    def _handle_msg(self, data: Any) -> None:
        self._on_message(data)


class ClimateUpload:
    """Build the local app's dialog for using climate data files from the user."""

    def __init__(
        self,
        on_uploaded: Callable[[Callable[[], xr.Dataset], str], None],
        *,
        busy: Any = False,
    ) -> None:
        self._on_uploaded = on_uploaded
        self._folder = _UploadFolder()
        self._picker = FilePicker(
            self._handle_picker,
            url=self._folder.url,
            disabled=busy,
            margin=(10, 5),
            sizing_mode="stretch_width",
        )
        self._status = StatusMessage(
            tone="neutral", margin=(6, 5, 0, 5), sizing_mode="stretch_width"
        )
        content = pn.Column(
            pn.pane.Markdown("## Upload climate data"),
            pn.pane.Markdown(
                "Choose NetCDF (.nc) files, or a folder containing them. Several "
                "files are combined into one dataset by their coordinates, such as "
                "one file per variable or period. Files saved with **Choose climate "
                "subset** can be used again here.\n\n"
                "The data need time, latitude and longitude coordinates. Files are "
                "copied to a temporary folder for this session."
            ),
            self._picker,
            self._status,
            sizing_mode="stretch_width",
        )
        self.modal = pn.Modal(content, stylesheets=[MODAL_CSS])
        self.button = self.modal.create_button(
            "show",
            label="Upload climate data",
            color="primary",
            variant="outline",
            icon="upload",
            margin=(6, 0),
            sizing_mode="stretch_width",
        )

    def _set_status(self, message: str, tone: str) -> None:
        self._status.param.update(value=message, tone=tone)

    def _handle_picker(self, message: Any) -> None:
        # Report upload progress from the browser, and use the files once uploaded.
        if not isinstance(message, dict):
            return
        kind = message.get("type")
        if kind == "start":
            count = message.get("count")
            noun = "file" if count == 1 else "files"
            self._set_status(f"Uploading {count} {noun}…", "info")
        elif kind == "done":
            self._use_files(message.get("files"))
        elif kind == "empty":
            self._set_status("No NetCDF (.nc) files were chosen.", "warning")
        elif kind == "error":
            self._folder.discard()
            self._set_status(str(message.get("message", "Upload failed.")), "danger")

    def _use_files(self, ids: Any) -> None:
        # Load uploaded files into the app, which opens them in the background.
        try:
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                raise ValueError("Some files did not finish uploading. Try again.")
            paths, names = self._folder.take(ids)
        except ValueError as exc:
            self._folder.discard()
            self._set_status(str(exc), "danger")
            return
        try:
            self._on_uploaded(partial(_open_uploads, paths, names), _upload_name(names))
        except ValueError as exc:
            _delete(paths)
            self._set_status(f"Could not use these files: {exc}", "warning")
            return
        self._set_status("", "neutral")
        self.modal.hide()

    def close(self) -> None:
        """Stop accepting uploads and delete files that were never opened."""
        self._folder.close()
