"""Subpackage for climate data."""
from climepi.climdata._examples import (  # noqa
    EXAMPLE_NAMES,
    create_example_dataset,
    get_example_dataset,
)
from climepi.climdata._cesm import CESMDataGetter, get_cesm_data  # noqa
