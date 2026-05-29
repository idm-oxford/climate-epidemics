"""Subpackage providing a browser-based front-end application."""

# Attributes provided by _app_construction are loaded lazily so that
# `from climepi import app` does not pay the cost of pulling in panel, bokeh,
# and dask.distributed for users who never launch the UI.
_LAZY_APP_CONSTRUCTION_ATTRS = frozenset({"run_app", "get_logger"})


def __getattr__(name):
    if name in _LAZY_APP_CONSTRUCTION_ATTRS:
        from climepi.app import _app_construction

        return getattr(_app_construction, name)
    raise AttributeError(f"module 'climepi.app' has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_LAZY_APP_CONSTRUCTION_ATTRS})
