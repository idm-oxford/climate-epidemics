"""
Module for creating and accessing example CESM LENS2 datasets. Running this module as a
script will create all example datasets by downloading and formatting the relevant CESM
LENS2 output data.
"""

import pathlib

import numpy as np
import pooch
import xcdat

from climepi.climdata import cesm

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


def get_example_dataset(name, data_dir=None):
    """
    Load a CESM LENS2 example dataset if it exists locally. If in future the formatted
    example datasets are made available for direct download, this function will be
    updated to download the dataset using pooch if it does not exist locally.

    Parameters
    ----------
    name : str
        Name of the example dataset to load. Currently available examples are:
        'world_2020_2060_2100' (global monthly data for the years 2020, 2060 and 2100),
        'cape_town' (monthly data for Cape Town between 2000 and 2100), and
        'europe_small' (monthly data for Europe between 2020 and 2100, including only
        the first two of the 100 total ensemble members).
    data_dir : str or pathlib.Path, optional
        Data directory in which to look for the example dataset. If not specified, the
        directory 'data/cesm_examples/{name}' within the same parent directory as the
        climepi package will be used if it exists, and otherwise the OS cache will be
        used.

    Returns
    -------
    xarray.Dataset
        Example dataset.
    """
    # Get details of the example dataset.
    example_details = _get_example_details(name)
    data_dir = _get_data_dir(name, data_dir)
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


def create_example_dataset(name, data_dir=None):
    """
    Create a CESM LENS2 example dataset from data in the aws server
    (https://ncar.github.io/cesm2-le-aws/model_documentation.html), and download it
    to the local machine. If the example dataset already exists locally, this function
    will return without re-downloading the data.

    Parameters
    ----------
    name : str
        Name of the example dataset to create. See `get_example_dataset` for a list of
        available examples.
    data_dir : str or pathlib.Path, optional
        Data directory in which to save the example dataset. If not specified, the
        directory 'data/cesm_examples/{name}' within the same parent directory as the
        climepi package will be used if it exists, and otherwise the OS cache will be
        used.

    Returns
    -------
    None
    """
    # Get details of the example dataset.
    example_details = _get_example_details(name)
    data_dir = _get_data_dir(name, data_dir)
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


def _get_data_dir(name, data_dir):
    # Helper function for getting the directory where the example dataset is to be
    # downloaded/accessed. If no directory is specified, then if a directory named
    # 'data/cesm_examples/{name}' exists in the same parent directory as the climepi
    # package, the example datasets will be downloaded to and accessed from that
    # directory. Otherwise, the datasets will be downloaded to and accessed from the OS
    # cache.
    if data_dir is None:
        base_dir = pathlib.Path(__file__).parents[3] / "data/cesm_examples"
        if not base_dir.exists():
            base_dir = pooch.os_cache("climepi/cesm_examples")
        data_dir = base_dir / name
    return data_dir


if __name__ == "__main__":
    for example_name in EXAMPLES:
        create_example_dataset(example_name)
        # get_example_dataset(example_name)
