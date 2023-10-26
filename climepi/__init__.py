"""
Python package for incorporating climate data into epidemiological models.
"""

from importlib.metadata import version

from ._core import ClimEpiDatasetAccessor  # noqa

# read version from installed package
__version__ = version("climepi")
