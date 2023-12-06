"""
Module for creating and accessing example CESM LENS2 datasets. If a directory named
'data/cesm_examples' exists in the same parent directory as the climepi package, the
example datasets will be downloaded to and accessed from that directory. Otherwise,
the datasets will be downloaded to and accessed from the OS cache directory. Running
this module as a script will create all example datasets by downloading and formatting
the relevant CESM LENS2 output data.
"""

import pathlib

import numpy as np
import pooch
import xcdat

from climepi.climdata import cesm

# Base directory for example datasets (individual datasets are downloaded to and
# accessed from subdirectories).
BASE_DIR = pathlib.Path(__file__).parents[3] / "data/cesm_examples"
if not BASE_DIR.exists():
    BASE_DIR = pooch.os_cache("climepi/cesm_examples")

# Dictionary of example datasets. Each key gives the example dataset name, and the
# corresponding value should be a dictionary with the following keys/values:
#   data_dir: Directory where the dataset is to be downloaded/accessed.
#   subset: Dictionary of options for subsetting the CESM dataset to pass to
#           CESMDataGetter as keyword arguments (see
#           climepi.climdata.cesm.CESMDataGetter for details).
#   climepi_modes: Dictionary defining the "modes" property of the climepi accessor
#                  for xarray datasets (see the climepi accessor documentation for
#                  details).
EXAMPLES = {
    "world_2020_2060_2100": {
        "data_dir": BASE_DIR / "world_2020_2060_2100",
        "subset": {
            "years": [2020, 2060, 2100],
        },
        "climepi_modes": {
            "spatial": "global",
            "temporal": "monthly",
            "ensemble": "ensemble",
        },
    },
    "cape_town": {
        "data_dir": BASE_DIR / "cape_town",
        "subset": {
            "years": np.arange(2000, 2101),
            "loc_str": "Cape Town",
        },
        "climepi_modes": {
            "spatial": "single",
            "temporal": "monthly",
            "ensemble": "ensemble",
        },
    },
    "europe_small": {
        "data_dir": BASE_DIR / "europe_small",
        "subset": {
            "years": [2020, 2100],
            "lat_range": [35, 72],
            "lon_range": [-25, 65],
            "realizations": np.arange(2),
        },
        "climepi_modes": {
            "spatial": "global",
            "temporal": "monthly",
            "ensemble": "ensemble",
        },
    },
}
EXAMPLE_NAMES = list(EXAMPLES.keys())


def get_example_dataset(name):
    """
    Load a CESM LENS2 example dataset if it exists locally. If in future the formatted
    example datasets are made available for direct download, this function will be
    updated to download the dataset using pooch if it does not exist locally.

    Parameters
    ----------
    name : str
        Name of the example dataset to load. Available examples are defined in the
        EXAMPLES dictionary in the _examples.py module of the climepi.climdata.cesm
        subpackage.

    Returns
    -------
    xarray.Dataset
        Example dataset.
    """
    # Get details of the example dataset.
    example_details = _get_example_details(name)
    data_dir = example_details["data_dir"]
    data_getter = cesm.CESMDataGetter(
        subset=example_details["subset"], save_dir=data_dir
    )
    file_names = data_getter.file_names
    # Create a pooch instance for the example dataset, and try to fetch the files.
    # Currently, the formatted example datasets are not available for direct download,
    # so this will either simply return the local file paths if the dataset exists
    # locally, or otherwise raise an error. However, in future this could be updated
    # to directly download the formatted example datasets if they are made available.
    pup = pooch.create(
        base_url="",
        path=data_dir,
        registry={file_name: None for file_name in file_names},
    )
    try:
        paths = [pup.fetch(file_name) for file_name in file_names]
    except ValueError as exc:
        raise NotImplementedError(
            "The formatted example dataset was not found locally and is not yet"
            + " available to download directly. Use 'create_example_data' to create"
            + " and download the formatted dataset from CESM output data."
        ) from exc
    # Load the dataset and set the 'modes' property of the climepi accessor.
    ds_example = xcdat.open_mfdataset(paths, chunks={})
    ds_example.climepi.modes = example_details["climepi_modes"]
    return ds_example


def create_example_dataset(name):
    """
    Create a CESM LENS2 example dataset from data in the aws server
    (https://ncar.github.io/cesm2-le-aws/model_documentation.html), and download it
    to the local machine. If the example dataset already exists locally, this function
    will return without re-downloading the data.
    """
    # Get details of the example dataset.
    example_details = _get_example_details(name)
    data_dir = example_details["data_dir"]
    subset = example_details["subset"]
    # Create and download the example dataset.
    cesm.get_cesm_data(subset=subset, save_dir=data_dir, download=True)


def _get_example_details(name):
    # Helper function for extracting the details of an example dataset from the
    # EXAMPLES dictionary in this module, and raising a customised error message
    # listing the available examples if the requested example is not found.
    try:
        example_details = EXAMPLES[name]
    except KeyError as exc:
        raise ValueError(
            f"Example data '{name}' not found. Available examples are: "
            f"{', '.join(EXAMPLES.keys())}"
        ) from exc
    return example_details


if __name__ == "__main__":
    for example_name in EXAMPLES:
        create_example_dataset(example_name)
        # get_example_dataset(example_name)
