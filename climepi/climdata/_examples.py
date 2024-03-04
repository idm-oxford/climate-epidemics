"""
Module for creating and accessing example CESM LENS2 datasets. Running this module as a
script will create all example datasets by downloading and formatting the relevant CESM
LENS2 output data.
"""

import pathlib

import numpy as np
import pooch

from climepi import climdata

# Dictionary of example datasets. Each key gives the example dataset name, and the
# corresponding value should be a dictionary with the following keys/values:
#   data_dir: Directory where the dataset is to be downloaded/accessed.
#   data_source: Data source to retrieve data from. Currently supported sources are
#       'lens2' (for CESM2 LENS data) and 'isimip' (for ISIMIP data).
#   frequency: Frequency of the data to retrieve. Should be one of 'daily', 'monthly' or
#       'yearly' (default is 'monthly').
#   subset: Dictionary of options for subsetting the dataset to pass to
#       climepi.climdata.get_climate_data as keyword arguments (see the docstring of
#       climepi.climdata.get_climate_data for details).
EXAMPLES = {
    "lens2_world": {
        "data_source": "lens2",
        "frequency": "monthly",
        "subset": {
            "years": [2020, 2060, 2100],
        },
    },
    "lens2_cape_town": {
        "data_source": "lens2",
        "frequency": "monthly",
        "subset": {
            "years": np.arange(2000, 2101),
            "loc_str": "Cape Town",
        },
    },
    "lens2_europe_small": {
        "data_source": "lens2",
        "frequency": "monthly",
        "subset": {
            "years": [2020, 2100],
            "lat_range": [35, 72],
            "lon_range": [-25, 65],
            "realizations": np.arange(2),
        },
    },
    "isimip_london": {
        "data_source": "isimip",
        "frequency": "monthly",
        "subset": {
            "loc_str": "London",
        },
    },
}
EXAMPLE_NAMES = list(EXAMPLES.keys())


def get_example_dataset(name, data_dir=None):
    """
    Load an example climate dataset if it exists locally. If the dataset does not exist
    locally, this function will retrieve, download and format the underlying data from
    the relevant server. If in future the formatted example datasets are made available
    for direct download, this function will be updated to download the dataset using
    `pooch` if it does not exist locally.

    Parameters
    ----------
    name : str
        Name of the example dataset to load. Currently available examples are:
        "lens2_world" (CESM LENS2 global monthly data for 2020, 2060, and 2100),
        "lens2_cape_town" (CESM LENS2 monthly data for Cape Town for 2000-2100),
        "lens2_europe_small" (CESM LENS2 monthly data for Europe for 2020 and 2100,
        with a small subset of realizations), and "isimip_london" (ISIMIP monthly data
        for London for 2000-2100).
    data_dir : str or pathlib.Path, optional
        Data directory in which to look for the example dataset. If not specified, the
        directory 'data/examples/{name}' within the same parent directory as the
        `climepi` package will be used if it exists, and otherwise the OS cache will be
        used.

    Returns
    -------
    xarray.Dataset
        Example dataset.
    """
    # Get details of the example dataset.
    example_details = _get_example_details(name)
    data_dir = _get_data_dir(name, data_dir)
    data_source = example_details["data_source"]
    frequency = example_details["frequency"]
    subset = example_details["subset"]
    file_names = climdata.get_climate_data_file_names(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
    )
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
        _ = [pup.fetch(file_name) for file_name in file_names]
    except ValueError:
        print(
            "The formatted example dataset was not found locally and is not currently"
            + " available to download directly. Searching for the raw data and creating"
            + " the example dataset from scratch."
        )
    # Load the dataset
    ds_example = climdata.get_climate_data(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
        save_dir=data_dir,
        download=True,
    )
    return ds_example


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
    # 'data/examples/{name}' exists in the same parent directory as the climepi
    # package, the example datasets will be downloaded to and accessed from that
    # directory. Otherwise, the datasets will be downloaded to and accessed from the OS
    # cache.
    if data_dir is None:
        base_dir = pathlib.Path(__file__).parents[2] / "data/examples"
        if not base_dir.exists():
            base_dir = pooch.os_cache("climepi/examples")
        data_dir = base_dir / name
    return data_dir


if __name__ == "__main__":
    for example_name in EXAMPLES:
        ds = get_example_dataset(example_name)
