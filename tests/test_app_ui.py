"""Unit tests for the shared visual elements of the app."""

import holoviews as hv
import panel as pn
import param
import pytest

from climepi.app import _app_classes_methods as app_classes_methods
from climepi.app import _app_ui as app_ui


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "From the ISIMIP repository (https://data.isimip.org/).",
            'From the ISIMIP repository (<a href="https://data.isimip.org/" '
            'target="_blank" rel="noopener">data.isimip.org</a>).',
        ),
        (
            "Kaye et al. (https://doi.org/10.1016/S2542-5196(24)00238-9), 2024",
            'Kaye et al. (<a href="https://doi.org/10.1016/S2542-5196(24)00238-9" '
            'target="_blank" rel="noopener">DOI</a>), 2024',
        ),
        ("Ranges <5 & >1 °C", "Ranges &lt;5 &amp; &gt;1 °C"),
    ],
)
def test_linkify(text, expected):
    """Bare URLs become short links without swallowing surrounding punctuation."""
    assert app_ui.linkify(text) == expected


@pytest.mark.parametrize(
    "message,tone",
    [
        ("Data not loaded", "neutral"),
        ("Loading data...", "info"),
        ("Data loaded: isimip_cities_daily", "success"),
        ("Data load failed: missing file", "danger"),
        ("Need to load climate data", "danger"),
        ("Model has not been run", "neutral"),
        ("Model run complete", "success"),
        ("Plot not yet generated", "neutral"),
        ("Plot generated", "success"),
    ],
)
def test_status_tone(message, tone):
    """Operation messages are colored by outcome."""
    assert app_ui.status_tone(message) == tone


def test_status_message_tone():
    """The tone follows bound messages unless it is set explicitly."""

    class Source(param.Parameterized):
        status = param.String(default="Loading data...")

    source = Source()
    inferred = app_ui.StatusMessage(value=source.param.status)
    assert inferred.tone == "info"
    source.status = "Data load failed: missing file"
    assert inferred.tone == "danger"

    explicit = app_ui.StatusMessage(tone="neutral")
    explicit.value = "Place not found."
    assert explicit.tone == "neutral"


def test_plot_workspace_shows_generation():
    """The chart card explains that a plot is on its way."""

    class Controller(param.Parameterized):
        plot_generating = param.Boolean(default=False)

    controller = Controller()
    controller.controls = pn.Row()
    controller.view = pn.Row()
    workspace = app_ui.plot_workspace(controller, "Load data first.")
    prompt = workspace[1][0]
    heading, spinner = prompt[0][1], prompt[0][0]
    instructions = prompt[1]
    assert "No plot yet" in heading.object
    assert instructions.object == "Load data first."
    assert not spinner.visible

    controller.controls.append(pn.pane.Markdown("settings"))
    controller.plot_generating = True
    assert "Generating plot" in heading.object
    assert spinner.visible
    assert prompt.visible

    controller.plot_generating = False
    controller.view.append(pn.pane.Markdown("plot"))
    assert not prompt.visible


def test_style_plot_widgets_keeps_dimension_labels():
    """Selectors keep their dimension names, which the pane uses to link them."""
    holomap = hv.HoloMap(
        {
            (location, model): hv.Curve([1, 2])
            for location in ["London", "Paris"]
            for model in ["gfdl-esm4", "canesm5"]
        },
        kdims=["location", "example model"],
    )
    pane = pn.pane.HoloViews(holomap, widget_layout=pn.Row)
    app_classes_methods._style_plot_widgets(pane.widget_box)
    widgets = list(pane.widget_box)
    assert [widget.label for widget in widgets] == ["location", "example model"]
    assert all(widget.width == 200 for widget in widgets)
    assert all(
        app_classes_methods._CAPITALIZED_LABEL_CSS in widget.stylesheets
        for widget in widgets
    )
