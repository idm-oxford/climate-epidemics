"""
Module for creating and accessing example climate projection datasets. Running this
module as a script will create all example datasets by downloading and formatting the
relevant data.
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
#   formatted_data_downloadable: Optional, boolean indicating whether the formatted
#       example dataset is available for direct download. If not specified, it is
#       assumed to be False.
EXAMPLES = {
    "isimip_cities": {
        "data_source": "isimip",
        "frequency": "monthly",
        "subset": {
            "locations": [
                "London",
                "Los Angeles",
                "Paris",
                "Islamabad",
            ],
        },
        "formatted_data_downloadable": True,
    },
    "isimip_cities_daily": {
        "data_source": "isimip",
        "frequency": "daily",
        "subset": {
            "locations": [
                "London",
                "Los Angeles",
                "Paris",
                "Islamabad",
            ],
        },
    },
    "lens2_2020_2060_2100": {
        "data_source": "lens2",
        "frequency": "monthly",
        "subset": {
            "years": [2020, 2060, 2100],
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
    "lens2_cities": {
        "data_source": "lens2",
        "frequency": "monthly",
        "subset": {
            "years": np.arange(2000, 2101),
            "locations": [
                "London",
                "Los Angeles",
                "Paris",
                "Islamabad",
            ],
        },
    },
}
EXAMPLE_NAMES = list(EXAMPLES.keys())


def get_example_dataset(name, data_dir=None, force_remake=False):
    """
    Load an example climate dataset if it exists locally, download the formatted example
    dataset if possible, or retrieve/download/format the raw underlying data from the
    relevant server.

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
        Data directory in which to look for the example dataset at, or download the
        dataset to. If not specified, the directory 'data/examples/{name}' within the
        parent directory of the climepi package will be used if 'data/examples' exists,
        otherwise the OS cache will be used.
    force_remake : bool, optional
        If True, force the download/formatting of the raw underlying data, even if the
        formatted dataset already exists locally and/or is available for direct
        download (default is False).


    Returns
    -------
    xarray.Dataset
        Example dataset.
    """
    data_dir = _get_data_dir(name, data_dir)
    example_details = _get_example_details(name)
    # If the formatted example dataset is available for direct download, download it
    # if neccessary
    if example_details.get("formatted_data_downloadable", False) and not force_remake:
        _fetch_formatted_example_dataset(name, data_dir)
    # Download and format the raw underlying data if neccessary, and return the dataset
    data_source = example_details["data_source"]
    frequency = example_details["frequency"]
    subset = example_details["subset"]
    ds_example = climdata.get_climate_data(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
        save_dir=data_dir,
        download=True,
        force_remake=force_remake,
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
    # 'data/examples' exists in the parent directory of the climepi package, the example
    # datasets will be downloaded to and accessed from 'data/examples/{name}'.
    # Otherwise, the datasets will be downloaded to and accessed from the OS cache.
    if data_dir is None:
        base_dir = pathlib.Path(__file__).parents[2] / "data/examples"
        if not base_dir.exists():
            base_dir = pooch.os_cache("climepi/examples")
        data_dir = base_dir / name
    return data_dir


def _fetch_formatted_example_dataset(name, data_dir):
    # Helper function for fetching the formatted example dataset if available for direct
    # download.
    example_details = _get_example_details(name)
    data_source = example_details["data_source"]
    frequency = example_details["frequency"]
    subset = example_details["subset"]
    file_names = climdata.get_climate_data_file_names(
        data_source=data_source,
        frequency=frequency,
        subset=subset,
    )
    url = (
        "https://github.com/will-s-hart/climate-epidemics/raw/main/data/examples/"
        + name
    )
    pup = pooch.create(
        base_url=url,
        path=data_dir,
        retry_if_failed=3,
    )
    registry_file_path = _get_registry_file_path(name)
    pup.load_registry(registry_file_path)
    _ = [pup.fetch(file_name) for file_name in file_names]


def _get_registry_file_path(name):
    # Helper function for getting the path to the registry file for the example dataset.
    return pathlib.Path(__file__).parent / "_example_registry_files" / f"{name}.txt"


def _make_example_registry(name, data_dir=None):
    # Create a registry file for the example dataset to be used by a pooch instance for
    # downloading the dataset.
    data_dir = _get_data_dir(name, data_dir)
    registry_file_path = _get_registry_file_path(name)
    pooch.make_registry(data_dir, registry_file_path, recursive=False)


def make_all_examples(force_remake=False):
    """
    Create all example datasets by downloading and formatting the relevant data.
    """
    exc = None
    for example_name in EXAMPLES:
        try:
            get_example_dataset(example_name, force_remake=force_remake)
            _make_example_registry(example_name)
        except TimeoutError as _exc:
            exc = _exc
    # Raise a TimeoutError if any of the datasets failed to download
    if exc:
        raise TimeoutError("Some downloads timed out. Please try again later.") from exc


if __name__ == "__main__":
    make_all_examples(force_remake=False)  # pragma: no cover
