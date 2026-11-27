"""Shared visual elements for the climepi app."""

from pathlib import Path
from typing import Any

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
    """Present operation feedback in a persistent screen-reader live region."""

    value = param.String(default="", allow_refs=True)

    _esm = """
    export function render({model}) {
        const status = document.createElement('div');
        status.className = 'operation-status';
        status.setAttribute('role', 'status');
        status.setAttribute('aria-live', 'polite');
        status.setAttribute('aria-atomic', 'true');
        const update = () => { status.textContent = model.value; };
        update();
        model.on('value', update);
        return status;
    }
    """
    _stylesheets = [
        """
        .operation-status {
            color: var(--climepi-muted, #3e5f8b);
            font-size: .82rem;
            overflow-wrap: anywhere;
            white-space: pre-wrap;
        }
        """
    ]


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


def plot_workspace(
    controls: pn.Row,
    plot_view: pn.Row,
    prerequisites: str,
) -> pn.FlexBox:
    """Place plot controls and the resulting chart in a wrapping layout."""
    options = control_section(
        "Plot settings",
        controls,
        margin=0,
        styles={**_CARD_STYLES, "flex": "1 1 16rem"},
    )
    instructions = pn.pane.Markdown(margin=0, styles=CONTROL_HELP)
    prompt = pn.Column(
        pn.pane.HTML(
            "<h2>No plot yet</h2>",
            margin=(0, 0, 8, 0),
            stylesheets=[_HEADING_CSS],
        ),
        instructions,
        margin=0,
        sizing_mode="stretch_width",
    )

    def update_state(_event=None) -> None:
        options.visible = bool(controls)
        prompt.visible = not plot_view
        instructions.object = (
            "Set the plot options and select **Generate plot**."
            if options.visible
            else prerequisites
        )

    update_state()
    controls.param.watch(update_state, "objects")
    plot_view.param.watch(update_state, "objects")
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
