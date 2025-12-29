# Example _version.py, completely non-vendored.
from pathlib import Path

from version_pioneer.api import get_version_dict_wo_exec


def get_version_dict():
    # NOTE: during installation, __file__ is not defined
    # When installed in editable mode, __file__ is defined
    # When installed in standard mode (when built), this file is replaced to a compiled versionfile.
    if "__file__" in globals():
        cwd = Path(__file__).parent
    else:
        cwd = Path.cwd()

    return get_version_dict_wo_exec(
        cwd=cwd,
        style="pep440",
        tag_prefix="v",
    )