"""Utility functions for the climdata subpackage."""

import pooch

from climepi._version import get_versions

FALLBACK_BRANCH = "main"


def _get_data_version() -> str:
    version = pooch.check_version(get_versions()["version"], fallback=FALLBACK_BRANCH)
    if version != FALLBACK_BRANCH:
        version = "v" + version
    return version
