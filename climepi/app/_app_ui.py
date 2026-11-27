"""Shared visual elements for the climepi app."""

import html
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import panel as pn
import param

TEMPLATE_CSS = Path(__file__).with_name("_app.css").read_text(encoding="utf-8")

_HEADING_CSS = """
    h2 {
        color: var(--climepi-ink, #102e55);
        font-size: 1.05rem;
        font-weight: 650;
        line-height: 1.35;
        margin: 0;
    }
"""

_CARD_STYLES = {
    "background": "#fff",
    "border": "1px solid var(--climepi-line, #cadcf4)",
    "border-radius": "12px",
    "box-sizing": "border-box",
    "padding": "18px",
    "min-width": "0",
}

TAB_CSS = """
    .bk-header {
        margin-bottom: 16px;
        border-bottom-color: var(--climepi-line, #cadcf4) !important;
    }
    .bk-headers-wrapper {
        width: 100%;
        gap: 6px;
    }
    .bk-tab {
        font-size: .91rem;
        font-weight: 600;
        color: var(--climepi-ink, #102e55);
        padding: 10px 15px;
        white-space: normal;
        align-content: center;
    }
    :host(.bk-above) .bk-tab {
        border-radius: 8px 8px 0 0;
    }
    :host(.bk-above) .bk-tab.bk-active {
        background: #fff;
        border-color: var(--climepi-line, #cadcf4);
        border-top-color: var(--climepi-accent, #1759a6);
        box-shadow: inset 0 2px var(--climepi-accent, #1759a6);
        color: var(--climepi-accent, #1759a6);
    }
    .bk-tab:focus-visible {
        outline: 2px solid var(--climepi-accent, #1759a6);
    }
    @media (max-width: 575px) {
        .bk-tab {
            flex: 1 1 0;
            font-size: .83rem;
            line-height: 1.25;
            padding: 8px 5px;
        }
    }
"""

CONTROL_HELP = {
    "color": "var(--climepi-muted, #3e5f8b)",
    "font-size": ".83rem",
    "line-height": "1.5",
    "overflow-wrap": "anywhere",
}
LINK_CSS = """
    a {
        color: var(--climepi-accent, #1759a6);
        font-weight: 600;
    }
"""

# Dialogs opened from the sidebar.
MODAL_CSS = """
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
# Links and buttons drawn in HTML, matching Panel's outline primary buttons.
OUTLINE_ACTION_CSS = """
a, button {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.375rem 0.75rem;
    border: 1px solid var(--design-primary-color, #1759a6);
    border-radius: 0.375rem;
    background: transparent;
    color: var(--design-primary-color, #1759a6);
    font: inherit;
    font-size: 0.875rem;
    text-decoration: none;
    cursor: pointer;
}
a:hover, button:enabled:hover {
    background: var(--design-primary-color, #1759a6);
    color: #fff;
}
a:focus-visible, button:focus-visible {
    outline: 2px solid var(--design-primary-color, #1759a6);
}
button:disabled { cursor: default; opacity: 0.6; }
"""

_URL = re.compile(r"https?://[^\s<>]+")


def linkify(text: str) -> str:
    """Escape text and replace bare URLs with short links."""

    def link(match: re.Match[str]) -> str:
        url, trailing = match.group(), ""
        # Leave sentence punctuation and an enclosing bracket outside the link, but
        # keep balanced brackets such as those in some DOIs.
        while url[-1] in ".,;:" or (url[-1] == ")" and url.count(")") > url.count("(")):
            url, trailing = url[:-1], url[-1] + trailing
        host = urlparse(url).netloc.removeprefix("www.")
        label = "DOI" if host == "doi.org" else host
        return f'<a href="{url}" target="_blank" rel="noopener">{label}</a>{trailing}'

    return _URL.sub(link, html.escape(text, quote=False))


def status_tone(message: str) -> str:
    """Classify an operation status message for coloring."""
    text = message.strip().lower()
    if any(word in text for word in ("failed", "error", "need to")):
        return "danger"
    if text.endswith(("...", "…")):
        return "info"
    if text.startswith(("data loaded", "model run complete", "plot generated")):
        return "success"
    return "neutral"


_SOURCE_DIVIDER_CSS = """
    .source-divider {
        display: flex;
        align-items: center;
        gap: 9px;
        color: var(--climepi-muted);
        font-size: .82rem;
    }
    .source-divider::before, .source-divider::after {
        content: "";
        flex: 1;
        border-top: 1px solid var(--climepi-line);
    }
"""


def source_divider() -> pn.pane.HTML:
    """Separate two ways to load climate data."""
    return pn.pane.HTML(
        '<div class="source-divider">or</div>',
        margin=(6, 0),
        sizing_mode="stretch_width",
        stylesheets=[_SOURCE_DIVIDER_CSS],
    )


class StatusMessage(pn.custom.JSComponent):
    """
    Present operation feedback in a persistent screen-reader live region.

    Without an explicit ``tone``, the tone follows the message text.
    """

    value = param.String(default="", allow_refs=True)
    tone = param.ObjectSelector(
        default="neutral", objects=["neutral", "info", "success", "warning", "danger"]
    )

    _esm = """
    // Bare URLs become links, without trailing sentence punctuation.
    const URL_PATTERN = /https?:\\/\\/[^\\s<>]*[^\\s<>.,;:!?)]/g;

    export function render({model}) {
        const status = document.createElement('div');
        status.className = 'operation-status';
        status.setAttribute('role', 'status');
        status.setAttribute('aria-live', 'polite');
        status.setAttribute('aria-atomic', 'true');
        const update = () => {
            const text = model.value;
            const parts = [];
            let end = 0;
            for (const match of text.matchAll(URL_PATTERN)) {
                const link = document.createElement('a');
                link.href = match[0];
                link.target = '_blank';
                link.rel = 'noopener';
                link.textContent = match[0].replace(/^https?:\\/\\//, '');
                parts.push(text.slice(end, match.index), link);
                end = match.index + match[0].length;
            }
            parts.push(text.slice(end));
            // Empty text would keep the :empty rule from hiding the region.
            status.replaceChildren(...parts.filter((part) => part !== ''));
            status.dataset.tone = model.tone;
        };
        update();
        model.on(['value', 'tone'], update);
        return status;
    }
    """
    _stylesheets = [
        """
        .operation-status {
            color: var(--climepi-muted, #3e5f8b);
            font-size: .82rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
            white-space: pre-wrap;
        }
        .operation-status:empty { display: none; }
        .operation-status a { color: inherit; font-weight: 600; }
        .operation-status:not([data-tone="neutral"]) {
            border-left: 3px solid;
            border-radius: 6px;
            padding: 6px 10px;
        }
        .operation-status[data-tone="info"] {
            background: #eef4fd;
            border-color: #1759a6;
            color: #1d4f8f;
        }
        .operation-status[data-tone="info"]::before {
            content: "";
            display: inline-block;
            width: .7em;
            height: .7em;
            margin-right: .5em;
            vertical-align: -.05em;
            border: 2px solid currentColor;
            border-right-color: transparent;
            border-radius: 50%;
            animation: climepi-spin .8s linear infinite;
        }
        .operation-status[data-tone="success"] {
            background: #e7f4eb;
            border-color: #2e7d46;
            color: #1e5631;
        }
        .operation-status[data-tone="warning"] {
            background: #fff4d6;
            border-color: #c48a00;
            color: #6b4a00;
        }
        .operation-status[data-tone="danger"] {
            background: #fdecea;
            border-color: #c0392b;
            color: #8a1c13;
        }
        @keyframes climepi-spin { to { transform: rotate(360deg); } }
        @media (prefers-reduced-motion: reduce) {
            .operation-status[data-tone="info"]::before { animation: none; }
        }
        """
    ]

    def __init__(self, **params: Any) -> None:
        infer_tone = "tone" not in params
        super().__init__(**params)
        if infer_tone:
            self._infer_tone()
            self.param.watch(self._infer_tone, "value")

    def _infer_tone(self, *_events: Any) -> None:
        # Color the message by the outcome its text describes.
        self.tone = status_tone(self.value)


def parameter_controls(owner: param.Parameterized, widgets: dict[str, Any]) -> pn.Param:
    """Lay out the specified widgets at the full available width."""
    return pn.Param(
        owner,
        widgets={
            name: {"sizing_mode": "stretch_width", "margin": (6, 0), **spec}
            if isinstance(spec, dict)
            else spec
            for name, spec in widgets.items()
        },
        parameters=list(widgets),
        show_name=False,
        margin=0,
        sizing_mode="stretch_width",
    )


def control_section(title: str, *controls: Any, **layout: Any) -> pn.Column:
    """Render a labeled group of sidebar controls."""
    defaults = {
        "margin": (0, 0, 14, 0),
        "sizing_mode": "stretch_width",
        "styles": _CARD_STYLES,
    }
    return pn.Column(
        pn.pane.HTML(f"<h2>{title}</h2>", margin=0, stylesheets=[_HEADING_CSS]),
        *controls,
        **(defaults | layout),
    )


def plot_workspace(plot_controller: Any, prerequisites: str) -> pn.FlexBox:
    """Place plot controls and the resulting chart in a wrapping layout."""
    controls, plot_view = plot_controller.controls, plot_controller.view
    options = control_section(
        "Plot settings",
        controls,
        margin=0,
        styles={**_CARD_STYLES, "flex": "1 1 16rem"},
    )
    heading = pn.pane.HTML(margin=(0, 0, 8, 0), stylesheets=[_HEADING_CSS])
    spinner = pn.indicators.LoadingSpinner(
        value=True, size=20, color="primary", margin=(0, 10, 8, 0), align="center"
    )
    instructions = pn.pane.Markdown(margin=0, styles=CONTROL_HELP)
    prompt = pn.Column(
        pn.Row(spinner, heading, margin=0),
        instructions,
        margin=0,
        sizing_mode="stretch_width",
    )

    def update_state(_event=None) -> None:
        generating = plot_controller.plot_generating
        options.visible = bool(controls)
        prompt.visible = generating or not plot_view
        spinner.visible = generating
        heading.object = (
            "<h2>Generating plot…</h2>" if generating else "<h2>No plot yet</h2>"
        )
        instructions.object = (
            "Large selections can take a minute."
            if generating
            else "Set the plot options and select **Generate plot**."
            if options.visible
            else prerequisites
        )

    update_state()
    controls.param.watch(update_state, "objects")
    plot_view.param.watch(update_state, "objects")
    plot_controller.param.watch(update_state, "plot_generating")
    plot = pn.Column(
        prompt,
        plot_view,
        margin=0,
        sizing_mode="stretch_width",
        styles={**_CARD_STYLES, "flex": "3 1 20rem", "overflow-x": "auto"},
    )
    return pn.FlexBox(options, plot, gap="16px", sizing_mode="stretch_width")


def model_workspace(plot_view: Any) -> pn.Column:
    """Render the selected model's suitability chart."""
    return pn.Column(
        plot_view,
        sizing_mode="stretch_width",
        margin=0,
        styles={**_CARD_STYLES, "overflow-x": "auto"},
    )
