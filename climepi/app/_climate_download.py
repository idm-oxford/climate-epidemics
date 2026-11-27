"""Climate subset selection and NetCDF export for local app sessions."""

import asyncio
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from inspect import getdoc
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any, Callable, Literal

import numpy as np
import panel as pn
import xarray as xr

from climepi._geocoding import geocode
from climepi.app._app_ui import CONTROL_HELP, StatusMessage
from climepi.climdata import _base
from climepi.climdata._data_getter_class import ClimateDataGetter, SubsetPartial

_LOG = logging.getLogger(__name__)
# Refuse subsets larger than this before writing them, so an over-broad request
# cannot fill the disk or stall the session.
_MAX_SUBSET_BYTES = 2 * 1024**3
_FREQUENCIES = {"Daily": "daily", "Monthly": "monthly", "Yearly": "yearly"}
_GEOMETRIES = {
    "Point by coordinates": "point",
    "Point by place name": "place",
    "Bounding box": "box",
}
_MODAL_STYLE = """
.dialog-overlay { background: rgb(10 36 75 / 62%); }
.dialog-content {
    width: min(42rem, calc(100vw - 1.5rem));
    max-height: calc(100dvh - 1.5rem);
    padding: 1rem clamp(0.5rem, 3vw, 1.5rem);
    border-radius: 12px;
    color: var(--climepi-ink, #102e55);
    box-shadow: 0 12px 48px rgb(10 36 75 / 35%);
}
"""
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
_STATUS_COLORS = {
    "info": "#f1f6fe",
    "success": "#deedff",
    "warning": "#fff2cf",
    "danger": "#fde8e5",
}


def write_netcdf(dataset: xr.Dataset, path: str | os.PathLike, **kwargs: Any) -> Any:
    """Write a dataset without cutting strings to a stale fixed width."""
    # Multi-file datasets keep the string width encoded in their first file, so
    # longer names from other files (e.g. "gfdl-esm4" after "canesm5") would
    # otherwise be truncated on write.
    dataset = dataset.copy(deep=False)
    for variable in dataset.variables.values():
        encoded = variable.encoding.get("dtype")
        if encoded is not None and np.dtype(encoded).kind == "U":
            del variable.encoding["dtype"]
    return dataset.to_netcdf(path, engine="h5netcdf", **kwargs)


def _source_information(source: type[ClimateDataGetter]) -> str:
    doc = getdoc(source) or ""
    summary = " ".join(doc.partition("\n\n")[0].split())
    link = re.search(r"https?://[^\s)>]+", doc)
    if link:
        url = link.group().rstrip(".,")
        return f"{summary}\n\n[{source.display_name()} source website]({url})"
    return summary


def _choice_group(widget: pn.widgets.CheckBoxGroup, noun: str) -> pn.Column:
    toggle = pn.widgets.Button(
        variant="outline", color="primary", margin=(0, 0, 0, 6), height=28, width=134
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
        pn.pane.Markdown(f"**{widget.label}**", margin=(0, 0, 4, 6)),
        toggle,
        pn.Column(
            widget,
            max_height=132,
            scroll="y-auto",
            margin=0,
            sizing_mode="stretch_width",
        ),
        width=260,
        margin=0,
        styles={
            "border": "1px solid var(--climepi-line)",
            "border-radius": "8px",
            "padding": "8px",
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
        return f"{self.source.display_name()} · {period} · {self.frequency}"


class _SubsetUnavailable(ValueError):
    """A valid request that cannot be fulfilled with the selected data."""


@dataclass
class _Download:
    """A retrieved file, removed once it is retired and no dataset uses it."""

    directory: TemporaryDirectory[str]
    path: Path
    name: str
    _users: int = 0
    _retired: bool = False
    # Session cleanup may close datasets on a worker thread.
    _lock: Lock = field(default_factory=Lock, repr=False)

    def open_dataset(self) -> xr.Dataset:
        source = xr.open_dataset(self.path, chunks={})
        try:
            if "time_bnds" in source:
                source.time_bnds.load()
            dataset = source.copy(deep=False)
        except BaseException:
            source.close()
            raise
        self._update(users=1)

        def close() -> None:
            try:
                source.close()
            finally:
                self._update(users=-1)

        # xarray runs this at most once per dataset.
        dataset.set_close(close)
        return dataset

    def retire(self) -> None:
        self._update(retire=True)

    def _update(self, *, users: int = 0, retire: bool = False) -> None:
        with self._lock:
            self._users += users
            self._retired = self._retired or retire
            if self._retired and not self._users:
                self.directory.cleanup()


def _prepare_download(request: _SubsetRequest) -> _Download:
    subset = request.subset.copy()
    if request.place:
        try:
            point = geocode(request.place)
        except ValueError as exc:
            raise _SubsetUnavailable(
                "Place not found. Try a more specific name or enter coordinates."
            ) from exc
        subset["lons"] = [point.longitude]
        subset["lats"] = [point.latitude]

    directory = TemporaryDirectory(prefix="climepi-subset-")
    years = subset["years"]
    filename = f"climepi-{request.source.data_source}-{request.frequency}-{years[0]}"
    if len(years) > 1:
        filename += f"-{years[-1]}"
    path = Path(directory.name) / f"{filename}.nc"
    try:
        source_dir = path.parent / "source"
        getter = _base._get_data_getter(
            request.source.data_source,
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
                data = data.assign_coords(location=[request.place])
            # Materialize a single NetCDF even when a provider returns several files.
            write_netcdf(data, path)
        if source_dir.exists():
            shutil.rmtree(source_dir)
    except BaseException:
        directory.cleanup()
        raise
    return _Download(directory, path, request.name)


class ClimateDownload:
    """Build the local app's climate data retrieval dialog."""

    def __init__(self, on_loaded: Callable[[xr.Dataset, str], None]) -> None:
        self._on_loaded = on_loaded
        self._sources = _base._data_source_getters()
        self._result: _Download | None = None
        self._working = False
        self._closed = False
        self._source = pn.widgets.Select(
            label="Data source",
            options={
                getter.display_name(): key for key, getter in self._sources.items()
            },
            width=260,
        )
        self._frequency = pn.widgets.Select(
            label="Time resolution", options=_FREQUENCIES, value="monthly", width=260
        )
        self._scenarios = pn.widgets.CheckBoxGroup(label="Scenarios")
        self._models = pn.widgets.CheckBoxGroup(label="Climate models")
        self._realizations = pn.widgets.CheckBoxGroup(
            label="Ensemble members", inline=True, stylesheets=[_MEMBER_CHOICES_CSS]
        )
        self._start = pn.widgets.IntInput(label="First year", width=125)
        self._end = pn.widgets.IntInput(label="Last year", width=125)
        self._geometry = pn.widgets.Select(label="Area", options=_GEOMETRIES)
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
        self._point_fields = pn.FlexBox(self._latitude, self._longitude, gap="0.75rem")
        self._place_fields = pn.Column(
            self._place,
            pn.pane.Markdown(
                "Place search uses [OpenStreetMap](https://www.openstreetmap.org/copyright).",
                margin=(0, 5),
            ),
            sizing_mode="stretch_width",
        )
        self._box_fields = pn.FlexBox(
            self._south, self._north, self._west, self._east, gap="0.75rem"
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
        self._source_note = pn.pane.Markdown(
            "GLENS retrieves the source grids before subsetting. "
            "Start with a short period and one ensemble member.",
            visible=False,
            margin=(0, 5),
        )
        self._source_info = pn.pane.Markdown(
            margin=(0, 5), styles=CONTROL_HELP, sizing_mode="stretch_width"
        )
        self._status = StatusMessage(sizing_mode="stretch_width")
        self._filename = pn.pane.Str("", margin=(5, 5, 0, 5))
        self._prepare = pn.widgets.Button(
            label="Retrieve subset",
            color="primary",
            icon="cloud-download",
            on_click=self._retrieve,
        )
        self._download = pn.widgets.FileDownload(
            label="Download NetCDF",
            icon="download",
            color="primary",
            variant="outline",
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
            pn.pane.Markdown("## Choose climate data"),
            pn.pane.Markdown("Select a period and an area to retrieve a NetCDF file."),
            pn.FlexBox(self._source, self._frequency, gap="0.75rem"),
            self._source_info,
            self._source_note,
            pn.FlexBox(
                _choice_group(self._scenarios, "scenarios"),
                _choice_group(self._models, "models"),
                gap="0.75rem",
            ),
            pn.FlexBox(
                _choice_group(self._realizations, "members"),
                pn.Row(self._start, self._end, margin=0),
                gap="0.75rem",
            ),
            self._geometry,
            self._point_fields,
            self._place_fields,
            self._box_fields,
            self._prepare,
            self._status,
            self._actions,
            sizing_mode="stretch_width",
            max_width=650,
        )
        self.modal = pn.Modal(
            content, background_close=False, stylesheets=[_MODAL_STYLE]
        )
        self.button = self.modal.create_button(
            "show",
            label="Choose climate subset",
            color="primary",
            variant="outline",
            icon="adjustments",
            sizing_mode="stretch_width",
        )
        self._update_source()
        self._update_geometry()
        self._source.param.watch(self._update_source, "value")
        self._scenarios.param.watch(self._update_models, "value")
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
        scenarios = source.scenarios_for_subsetting()
        self._scenarios.param.update(options=scenarios, value=scenarios[:1])
        members = source.available_realizations
        self._realizations.param.update(options=members, value=members[:1])
        self._update_models(reset=True)
        self._update_years(reset=True)
        self._source_note.visible = source.data_source == "glens"

    def _update_models(self, *_events, reset: bool = False) -> None:
        source = self._sources[self._source.value]
        models = source.models_for_scenarios(self._scenarios.value)
        selected = [] if reset else [m for m in self._models.value if m in models]
        self._models.param.update(options=models, value=selected or models[:1])
        self._update_years()

    def _update_years(self, *_events, reset: bool = False) -> None:
        minimum, maximum = self._sources[self._source.value].year_limits(
            self._scenarios.value,
            self._realizations.value,
        )
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
                if source.data_source == "isimip" and east == 180:
                    # The provider normalizes +180 to -180. Stop just before the seam.
                    east = 179.9999
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
        self._status.param.update(
            value=message,
            styles={
                "background": _STATUS_COLORS[kind],
                "border-radius": "8px",
                "padding": "8px 12px",
            },
        )

    def _set_working(self, working: bool) -> None:
        self._working = working
        for widget in (*self._inputs, self._prepare, self._download, self._use):
            widget.disabled = working
        self._prepare.loading = working

    def _clear_result(self, _event=None) -> None:
        if self._result is not None:
            self._result.retire()
            self._result = None
            self._download.param.update(file=None, data=None)
            self._actions.visible = False
            self._status.param.update(value="", styles={})

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
            self._download.param.update(file=result.path, data=None)
            self._filename.object = result.name
            self._actions.visible = True
            self._set_status("Climate subset ready.", "success")
        except _SubsetUnavailable as exc:
            if not self._closed:
                self._set_status(str(exc), "warning")
        except TimeoutError:
            if not self._closed:
                self._set_status(
                    "The source is still preparing this subset. Try the same request again.",
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
            self._set_status(f"Could not use this subset in the app: {exc}", "warning")
        except Exception:
            if dataset is not None:
                dataset.close()
            _LOG.exception("Could not load the climate subset into the app")
            self._set_status("Could not load this subset into the app.", "danger")
        else:
            self.modal.hide()

    def close(self) -> None:
        """Release this session's download when no app operation is using it."""
        self._closed = True
        if self._result is not None:
            self._result.retire()
            self._result = None
