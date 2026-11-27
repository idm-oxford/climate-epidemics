"""Climate data retrieval and NetCDF export for local app sessions."""

import asyncio
import logging
import secrets
import shutil
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any, BinaryIO, Callable, Literal, NamedTuple, cast

import panel as pn
import tornado.web
import xarray as xr

from climepi._geocoding import geocode
from climepi.app._app_ui import CONTROL_HELP, MODAL_CSS, StatusMessage
from climepi.app._local_files import open_netcdf
from climepi.climdata import _base
from climepi.climdata._data_getter_class import ClimateDataGetter, SubsetPartial
from climepi.climdata._isimip import SubsetPendingError

_LOG = logging.getLogger(__name__)
# Refuse subsets larger than this before writing them, so an over-broad request
# cannot fill the disk or stall the session.
_MAX_SUBSET_BYTES = 2 * 1024**3
# Retrieved files are served from this route (see DownloadHandler and run_app).
DOWNLOAD_ROUTE = "climepi_download"
_FREQUENCIES = {"Daily": "daily", "Monthly": "monthly", "Yearly": "yearly"}
_GEOMETRIES = {
    "Point by coordinates": "point",
    "Point by place name": "place",
    "Bounding box": "box",
}
_MEMBER_CHOICES_CSS = """
.bk-input-group.bk-inline {
    flex-wrap: wrap;
    gap: 0.25rem 0.5rem;
    height: auto;
}
.bk-input-group.bk-inline > * {
    min-width: 2.4rem;
}
.bk-input-group.bk-inline > *:not(:first-child) { margin-left: 0; }
"""
_DOWNLOAD_LINK_CSS = """
a {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.375rem 0.75rem;
    border: 1px solid var(--design-primary-color, #1759a6);
    border-radius: 0.375rem;
    color: var(--design-primary-color, #1759a6);
    font-size: 0.875rem;
    text-decoration: none;
}
a:hover { background: var(--design-primary-color, #1759a6); color: #fff; }
a:focus-visible { outline: 2px solid var(--design-primary-color, #1759a6); }
"""
_DOWNLOAD_ICON = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-2"/><path d="M7 11l5 5l5 -5"/>'
    '<path d="M12 4l0 12"/></svg>'
)
# Fields share a row on wide dialogs and stack on narrow ones.
_FIELD = {"flex": "1 1 14rem", "min-width": "0"}
_FOOTER = {
    # Keep the action and its outcome in view however far the form scrolls.
    "position": "sticky",
    "bottom": "0",
    "z-index": "1",
    "background": "#fff",
    "border-top": "1px solid var(--climepi-line, #cadcf4)",
    "padding": "0.5rem 0",
    "margin-top": "0.5rem",
}


class _SourceDetails(NamedTuple):
    name: str
    description: str = ""
    url: str = ""


# How each data source is presented. A source without an entry is still offered,
# under its identifier in capitals.
_SOURCE_DETAILS = {
    "lens2": _SourceDetails(
        "LENS2",
        "CESM2 Large Ensemble projections, combining historical simulations with "
        "SSP3-7.0.",
        "https://registry.opendata.aws/ncar-cesm2-lens/",
    ),
    "arise": _SourceDetails(
        "ARISE-SAI",
        "CESM2 simulations comparing stratospheric aerosol injection with an SSP2-4.5 "
        "reference.",
        "https://registry.opendata.aws/ncar-cesm2-arise/",
    ),
    "glens": _SourceDetails(
        "GLENS",
        "CESM1 simulations comparing stratospheric aerosol injection with an RCP8.5 "
        "reference. Full source grids are downloaded before subsetting, so start with "
        "a short period and one ensemble member.",
        "https://rda.ucar.edu/datasets/d651064/",
    ),
    "isimip": _SourceDetails(
        "ISIMIP3b",
        "Bias-adjusted CMIP6 projections for several climate models and scenarios.",
        "https://data.isimip.org/",
    ),
    "isimip3a": _SourceDetails(
        "ISIMIP3a",
        "Observation-based GSWP3-W5E5 historical climate, with a detrended "
        "counterfactual.",
        "https://protocol.isimip.org/",
    ),
    "isimip4a": _SourceDetails(
        "ISIMIP4a",
        "Observation-based GSWP3-ERA5 historical climate.",
        "https://protocol4.isimip.org/",
    ),
    "isimip4b": _SourceDetails(
        "ISIMIP4b",
        "Bias-adjusted CMIP7 projections, with a historical run from 1850 to 2021.",
        "https://protocol4.isimip.org/",
    ),
    "era5": _SourceDetails(
        "ERA5",
        "ECMWF global reanalysis, with daily values aggregated from hourly data.",
        "https://registry.opendata.aws/earthmover-era5/",
    ),
}


def _source_details(source: type[ClimateDataGetter]) -> _SourceDetails:
    # Get how a data source is presented, defaulting to its identifier.
    return _SOURCE_DETAILS.get(
        source.data_source, _SourceDetails(source.data_source.upper())
    )


def _source_information(source: type[ClimateDataGetter]) -> str:
    details = _source_details(source)
    if not details.url:
        return details.description
    return f"{details.description}\n\n[{details.name} source website]({details.url})"


def _row(*items: Any) -> pn.FlexBox:
    for item in items:
        item.param.update(sizing_mode="stretch_width", styles={**item.styles, **_FIELD})
    return pn.FlexBox(*items, gap="0 0.75rem", sizing_mode="stretch_width")


def _choice_group(
    widget: pn.widgets.CheckBoxGroup, noun: str, *, wide: bool = False
) -> pn.Column:
    toggle = pn.widgets.Button(
        variant="outline", color="primary", margin=0, height=28, width=134
    )

    def update_toggle(*_events) -> None:
        selected_all = len(widget.value) == len(widget.options)
        toggle.label = f"Clear all {noun}" if selected_all else f"Select all {noun}"
        toggle.visible = len(widget.options) > 1
        toggle.disabled = widget.disabled

    def toggle_selection(_event) -> None:
        widget.value = (
            [] if len(widget.value) == len(widget.options) else list(widget.options)
        )

    update_toggle()
    widget.param.watch(update_toggle, ["options", "value", "disabled"])
    toggle.on_click(toggle_selection)
    return pn.Column(
        pn.Row(
            pn.pane.Markdown(
                f"**{widget.label}**", margin=(0, 6), stylesheets=["p { margin: 0; }"]
            ),
            pn.HSpacer(),
            toggle,
            margin=(0, 0, 4, 0),
            align="center",
            sizing_mode="stretch_width",
        ),
        pn.Column(
            widget,
            margin=0,
            sizing_mode="stretch_width",
            # Panel's max_height resolves against an auto-height parent and is
            # ignored, so set the scroll limit directly.
            styles={"max-height": "8.5rem", "overflow-y": "auto"},
        ),
        margin=(6, 0),
        sizing_mode="stretch_width",
        styles={
            "border": "1px solid var(--climepi-line)",
            "border-radius": "8px",
            "padding": "8px",
            **({"flex": "1 1 100%"} if wide else _FIELD),
        },
    )


@dataclass(frozen=True)
class _SubsetRequest:
    source: type[ClimateDataGetter]
    frequency: Literal["daily", "monthly", "yearly"]
    subset: SubsetPartial
    place: str | None = None

    @property
    def name(self) -> str:
        years = self.subset["years"]
        period = str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]}"
        return f"{_source_details(self.source).name} · {period} · {self.frequency}"


class _SubsetUnavailable(ValueError):
    """A valid request that cannot be fulfilled with the selected data."""


# Retrieved files that can be downloaded, by the unguessable token in their URL.
_DOWNLOADS: dict[str, "_Download"] = {}


@dataclass
class _Download:
    """A retrieved file, removed once it is retired and nothing is reading it."""

    directory: TemporaryDirectory[str]
    path: Path
    name: str
    # Full address found for a place name, if one was searched for.
    address: str = ""
    _users: int = 0
    _retired: bool = False
    # Session cleanup may close datasets on a worker thread.
    _lock: Lock = field(default_factory=Lock, repr=False)
    token: str = field(default_factory=lambda: secrets.token_urlsafe(16))

    def __post_init__(self) -> None:
        _DOWNLOADS[self.token] = self

    @property
    def url(self) -> str:
        """Return the file's download address, relative to the app page."""
        return f"{DOWNLOAD_ROUTE}/{self.token}"

    def open_file(self) -> BinaryIO | None:
        """Open the file for a download, or return None once it is retired."""
        with self._lock:
            if self._retired:
                return None
            self._users += 1
        try:
            return self.path.open("rb")
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        """Mark a download or dataset opened from the file as finished."""
        self._update(users=-1)

    def open_dataset(self) -> xr.Dataset:
        """Open the file lazily, keeping it until the dataset is closed."""
        dataset = open_netcdf([self.path], on_close=self.release)
        self._update(users=1)
        return dataset

    def retire(self) -> None:
        """Stop offering the file, and remove it once nothing is reading it."""
        _DOWNLOADS.pop(self.token, None)
        self._update(retire=True)

    def _update(self, *, users: int = 0, retire: bool = False) -> None:
        # Count readers, mark the file retired, and remove it once both allow.
        with self._lock:
            self._users += users
            self._retired = self._retired or retire
            if self._retired and not self._users:
                self.directory.cleanup()


class DownloadHandler(tornado.web.RequestHandler):
    """Stream a retrieved climate subset to the browser."""

    async def get(self, token: str) -> None:
        """Send the file for a download token, or 404 if it is no longer offered."""
        download = _DOWNLOADS.get(token)
        file = download.open_file() if download is not None else None
        if download is None or file is None:
            raise tornado.web.HTTPError(404)
        try:
            self.set_header("Content-Type", "application/x-netcdf")
            self.set_header(
                "Content-Disposition", f'attachment; filename="{download.path.name}"'
            )
            self.set_header("Content-Length", str(download.path.stat().st_size))
            # Send the file in pieces so large subsets are never held in memory.
            while chunk := file.read(1 << 20):
                self.write(chunk)
                await self.flush()
        finally:
            file.close()
            download.release()


def _prepare_download(request: _SubsetRequest) -> _Download:
    subset = request.subset.copy()
    address = ""
    if request.place:
        try:
            point = geocode(request.place)
        except ValueError as exc:
            raise _SubsetUnavailable(
                "Place not found. Try a more specific name or enter coordinates."
            ) from exc
        subset["lons"] = [point.longitude]
        subset["lats"] = [point.latitude]
        address = point.address

    directory = TemporaryDirectory(prefix="climepi-subset-")
    years = subset["years"]
    filename = f"climepi-{request.source.data_source}-{request.frequency}-{years[0]}"
    if len(years) > 1:
        filename += f"-{years[-1]}"
    path = Path(directory.name) / f"{filename}.nc"
    try:
        source_dir = path.parent / "source"
        getter = _base._get_data_getter(
            cast(_base.ClimateDataSource, request.source.data_source),
            frequency=request.frequency,
            subset=subset,
            save_dir=source_dir,
            max_subset_wait_time=120,
        )
        with getter.get_data(download=not getter.remote_open_possible) as data:
            if any(data.sizes.get(axis, 1) == 0 for axis in ("time", "lat", "lon")):
                raise _SubsetUnavailable("The selected years or area contain no data.")
            if data.nbytes > _MAX_SUBSET_BYTES:
                raise _SubsetUnavailable(
                    f"This selection is about {data.nbytes / 1024**3:.1f} GB, above "
                    f"the {_MAX_SUBSET_BYTES / 1024**3:.0f} GB limit. Choose fewer "
                    "years or ensemble members, or a smaller area."
                )
            if request.place:
                # Match sel_geo, which records the address found for a place name.
                data = data.assign_coords(location=[request.place])
                data["location"].attrs["geocoded_address"] = address
            # Materialize a single NetCDF even when a provider returns several files.
            data.to_netcdf(path, engine="h5netcdf")
        if source_dir.exists():
            shutil.rmtree(source_dir)
    except BaseException:
        directory.cleanup()
        raise
    return _Download(directory, path, request.name, address)


class ClimateDownload:
    """Build the local app's climate data retrieval dialog."""

    def __init__(self, on_loaded: Callable[[xr.Dataset, str], None]) -> None:
        self._on_loaded = on_loaded
        self._sources = _base._data_getter_classes()
        self._result: _Download | None = None
        self._working = False
        self._closed = False
        self._source = pn.widgets.Select(
            label="Data source",
            options={
                _source_details(getter).name: key
                for key, getter in self._sources.items()
            },
        )
        self._frequency = pn.widgets.Select(
            label="Time resolution", options=_FREQUENCIES, value="monthly"
        )
        self._scenarios = pn.widgets.CheckBoxGroup(label="Scenarios")
        self._models = pn.widgets.CheckBoxGroup(label="Climate models")
        self._realizations = pn.widgets.CheckBoxGroup(
            label="Ensemble members", inline=True, stylesheets=[_MEMBER_CHOICES_CSS]
        )
        self._start = pn.widgets.IntInput(label="First year")
        self._end = pn.widgets.IntInput(label="Last year")
        self._geometry = pn.widgets.Select(
            label="Area", options=_GEOMETRIES, sizing_mode="stretch_width"
        )
        self._longitude = self._coordinate("Longitude (°)", 180)
        self._latitude = self._coordinate("Latitude (°)", 90)
        self._place = pn.widgets.TextInput(
            label="Place name",
            placeholder="e.g. Oxford, United Kingdom",
            max_length=160,
        )
        self._west = self._coordinate("West longitude (°)", 180)
        self._east = self._coordinate("East longitude (°)", 180)
        self._south = self._coordinate("South latitude (°)", 90)
        self._north = self._coordinate("North latitude (°)", 90)
        self._point_fields = _row(self._latitude, self._longitude)
        self._place.sizing_mode = "stretch_width"
        self._place_fields = pn.Column(
            self._place,
            pn.pane.Markdown(
                "Place search uses [OpenStreetMap](https://www.openstreetmap.org/copyright).",
                margin=(0, 5),
            ),
            sizing_mode="stretch_width",
        )
        self._box_fields = pn.Column(
            _row(self._south, self._north),
            _row(self._west, self._east),
            margin=0,
            sizing_mode="stretch_width",
        )
        self._inputs = (
            self._source,
            self._frequency,
            self._scenarios,
            self._models,
            self._realizations,
            self._start,
            self._end,
            self._geometry,
            self._longitude,
            self._latitude,
            self._place,
            self._west,
            self._east,
            self._south,
            self._north,
        )
        self._source_info = pn.pane.Markdown(
            margin=(0, 5), styles=CONTROL_HELP, sizing_mode="stretch_width"
        )
        self._status = StatusMessage(
            tone="neutral", margin=(6, 5, 0, 5), sizing_mode="stretch_width"
        )
        self._filename = pn.pane.Markdown("", margin=(5, 5, 0, 5), styles=CONTROL_HELP)
        self._prepare = pn.widgets.Button(
            label="Retrieve data",
            color="primary",
            icon="cloud-download",
            on_click=self._retrieve,
        )
        # A link to DownloadHandler streams the file, where Panel's FileDownload
        # would send it through the session's websocket and keep it in memory.
        self._download = pn.pane.HTML(
            "", margin=(5, 5), stylesheets=[_DOWNLOAD_LINK_CSS]
        )
        self._use = pn.widgets.Button(
            label="Use dataset in app",
            color="primary",
            variant="outline",
            on_click=self._use_result,
        )
        self._actions = pn.Column(
            self._filename,
            pn.FlexBox(self._download, self._use, gap="0.75rem"),
            visible=False,
            sizing_mode="stretch_width",
        )
        content = pn.Column(
            pn.pane.Markdown("## Retrieve online data"),
            pn.pane.Markdown("Select a period and an area to retrieve a NetCDF file."),
            _row(self._source, self._frequency),
            self._source_info,
            pn.FlexBox(
                _choice_group(self._scenarios, "scenarios"),
                _choice_group(self._models, "models"),
                _choice_group(self._realizations, "members", wide=True),
                gap="0 0.75rem",
                sizing_mode="stretch_width",
            ),
            _row(self._start, self._end),
            self._geometry,
            self._point_fields,
            self._place_fields,
            self._box_fields,
            pn.Column(
                self._prepare,
                self._status,
                self._actions,
                margin=0,
                sizing_mode="stretch_width",
                styles=_FOOTER,
            ),
            sizing_mode="stretch_width",
        )
        self.modal = pn.Modal(content, background_close=False, stylesheets=[MODAL_CSS])
        self.button = self.modal.create_button(
            "show",
            label="Retrieve online data",
            color="primary",
            variant="outline",
            icon="cloud-download",
            margin=(6, 0),
            sizing_mode="stretch_width",
        )
        self._update_source()
        self._update_geometry()
        self._source.param.watch(self._update_source, "value")
        self._scenarios.param.watch(self._update_years, "value")
        self._realizations.param.watch(self._update_years, "value")
        self._geometry.param.watch(self._update_geometry, "value")
        for widget in self._inputs:
            widget.param.watch(self._clear_result, "value")

    @staticmethod
    def _coordinate(label: str, bound: float) -> pn.widgets.FloatInput:
        return pn.widgets.FloatInput(
            label=label, start=-bound, end=bound, value=None, step=0.25, width=175
        )

    def _update_source(self, *_events) -> None:
        source = self._sources[self._source.value]
        self._source_info.object = _source_information(source)
        scenarios = [*source.available_scenarios, *source.optional_scenarios]
        self._scenarios.param.update(options=scenarios, value=scenarios[:1])
        models = source.available_models
        self._models.param.update(options=models, value=models[:1])
        members = source.available_realizations
        self._realizations.param.update(options=members, value=members[:1])
        self._update_years(reset=True)

    def _update_years(self, *_events, reset: bool = False) -> None:
        minimum, maximum = self._sources[self._source.value].year_limits(
            self._scenarios.value,
            self._realizations.value,
        )
        if minimum > maximum:
            # Leave the years as they are; _request explains the problem.
            return
        if reset:
            first = max(min(2020, maximum), minimum)
            last = min(first + 4, maximum)
        else:
            first = max(min(self._start.value or minimum, maximum), minimum)
            last = max(min(self._end.value or minimum, maximum), minimum)
        self._start.param.update(start=minimum, end=maximum, value=first)
        self._end.param.update(start=minimum, end=maximum, value=last)

    def _update_geometry(self, *_events) -> None:
        self._point_fields.visible = self._geometry.value == "point"
        self._place_fields.visible = self._geometry.value == "place"
        self._box_fields.visible = self._geometry.value == "box"

    @staticmethod
    def _coordinate_value(field: pn.widgets.FloatInput) -> float:
        value = field.value
        start, end = field.start, field.end
        assert start is not None and end is not None
        if value is None or not isfinite(value) or not start <= value <= end:
            raise ValueError(
                f"Enter {field.label.lower()} in the range {start} to {end}."
            )
        return value

    def _request(self) -> _SubsetRequest:
        source = self._sources[self._source.value]
        start, end = self._start.value, self._end.value
        scenarios = self._scenarios.value
        models = self._models.value
        members = self._realizations.value
        if not scenarios or not models or not members:
            raise ValueError(
                "Select at least one scenario, climate model and ensemble member."
            )
        minimum, maximum = source.year_limits(scenarios, members)
        if minimum > maximum:
            raise ValueError(
                "The selected scenarios have no years in common. Retrieve them "
                "separately."
            )
        if start is None or end is None or not minimum <= start <= end <= maximum:
            raise ValueError(f"Select years in order from {minimum} to {maximum}.")
        subset: SubsetPartial = {
            "years": list(range(start, end + 1)),
            "scenarios": scenarios.copy(),
            "models": models.copy(),
            "realizations": members.copy(),
        }
        place = None
        if self._geometry.value == "box":
            west, east = (
                self._coordinate_value(field) for field in (self._west, self._east)
            )
            south, north = (
                self._coordinate_value(field) for field in (self._south, self._north)
            )
            if west >= east or south >= north:
                raise ValueError(
                    "Set west below east and south below north for a bounding box."
                )
            if east - west < 360:
                subset["lon_range"] = (west, east)
            if north - south < 180:
                subset["lat_range"] = (south, north)
        else:
            subset["locations"] = ["Point"]
            if self._geometry.value == "point":
                # Validate in the order the fields are shown.
                subset["lats"] = [self._coordinate_value(self._latitude)]
                subset["lons"] = [self._coordinate_value(self._longitude)]
            else:
                place = (self._place.value or "").strip()
                if not place:
                    raise ValueError("Enter a place name.")
        return _SubsetRequest(source, self._frequency.value, subset, place)

    def _set_status(self, message: str, kind: str) -> None:
        self._status.param.update(value=message, tone=kind)

    def _set_working(self, working: bool) -> None:
        self._working = working
        for widget in (*self._inputs, self._prepare, self._use):
            widget.disabled = working
        self._prepare.loading = working

    def _clear_result(self, _event=None) -> None:
        if self._result is not None:
            self._result.retire()
            self._result = None
            self._download.object = ""
            self._actions.visible = False
            self._status.param.update(value="", tone="neutral")

    async def _retrieve(self, _event=None) -> None:
        if self._working or self._closed:
            return
        try:
            request = self._request()
        except ValueError as exc:
            self._set_status(str(exc), "warning")
            return
        self._clear_result()
        self._set_working(True)
        self._set_status("Retrieving climate data…", "info")
        try:
            result = await asyncio.to_thread(_prepare_download, request)
            if self._closed:
                result.retire()
                return
            self._result = result
            self._download.object = (
                f'<a href="{result.url}" download="{result.path.name}">'
                f"{_DOWNLOAD_ICON}Download NetCDF</a>"
            )
            self._filename.object = result.name
            self._actions.visible = True
            found = f" Location found: {result.address}." if result.address else ""
            self._set_status(f"Climate data ready.{found}", "success")
        except _SubsetUnavailable as exc:
            if not self._closed:
                self._set_status(str(exc), "warning")
        except SubsetPendingError as exc:
            if not self._closed:
                self._set_status(
                    "The source is still preparing these data. Select Retrieve data "
                    "again once it has finished. Check progress at:\n"
                    + "\n".join(exc.job_urls),
                    "warning",
                )
        except Exception:
            _LOG.exception("Climate subset retrieval failed")
            if not self._closed:
                self._set_status(
                    "Climate retrieval failed. Check the source and selection, "
                    "then try again.",
                    "danger",
                )
        finally:
            if not self._closed:
                self._set_working(False)

    def _use_result(self, _event=None) -> None:
        if self._result is None or self._working or self._closed:
            return
        dataset = None
        try:
            dataset = self._result.open_dataset()
            self._on_loaded(dataset, self._result.name)
        except ValueError as exc:
            # A busy session, or data the app cannot plot; the message says which.
            if dataset is not None:
                dataset.close()
            self._set_status(f"Could not use these data in the app: {exc}", "warning")
        except Exception:
            if dataset is not None:
                dataset.close()
            _LOG.exception("Could not load retrieved climate data into the app")
            self._set_status("Could not load these data into the app.", "danger")
        else:
            self.modal.hide()

    def close(self) -> None:
        """Release this session's download when no app operation is using it."""
        self._closed = True
        if self._result is not None:
            self._result.retire()
            self._result = None
