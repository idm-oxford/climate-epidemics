"""
Module for creating and accessing example CESM LENS2 datasets. If a directory named
'data/cesm_examples' exists in the same parent directory as the climepi package, the
example datasets will be downloaded to and accessed from that directory. Otherwise,
the datasets will be downloaded to and accessed from the OS cache directory.
"""

import pathlib

import numpy as np
import pooch
import xcdat

from climepi.climdata.cesm import CESMDataDownloader

BASE_DIR = pathlib.Path(__file__).parents[3] / "data/cesm_examples"
if not BASE_DIR.exists():
    BASE_DIR = pooch.os_cache("climepi/cesm_examples")
REALIZATIONS_AVAILABLE = np.arange(100)
EXAMPLES = {
    # "world_2020_2060_2100": {
    #     "data_dir": BASE_DIR / "world_2020_2060_2100",
    #     "subset": {
    #         "realizations": [0, 1],
    #         "years": [2020, 2060, 2100],
    #     },
    # },
    "world_2020_2060_2100": {
        "data_dir": BASE_DIR / "world_2020_2060_2100",
        "subset": {
            "realizations": [0],
            "years": [2020, 2060],
        },
    },
}


def get_example_dataset(name):
    try:
        example_details = EXAMPLES[name]
    except KeyError as exc:
        raise ValueError(
            f"Example data '{name}' not found. Available examples are: "
            f"{', '.join(EXAMPLES.keys())}"
        ) from exc

    data_dir = example_details["data_dir"]
    realizations = example_details["subset"].get("realizations", REALIZATIONS_AVAILABLE)

    file_names = ["realization_" + str(i) + ".nc" for i in realizations]
    pup = pooch.create(
        base_url="",
        path=data_dir,
        registry={file_name: None for file_name in file_names},
    )

    try:
        paths = [pup.fetch(file_name) for file_name in file_names]
    except ValueError as exc:
        raise NotImplementedError(
            "The formatted example dataset was not found locally and is not available",
            " to download. Run 'create_example_data' to download the relevant CESM ",
            " output data and create the formatted dataset.",
        ) from exc

    ds = xcdat.open_mfdataset(paths)
    return ds


def create_example_dataset(name):
    """Create CESM LENS2 example dataset."""
    try:
        example_details = EXAMPLES[name]
    except KeyError as exc:
        raise ValueError(
            f"Example data '{name}' not found. Available examples are: "
            f"{', '.join(EXAMPLES.keys())}"
        ) from exc

    data_dir = example_details["data_dir"]
    subset = example_details["subset"]

    downloader = CESMDataDownloader(data_dir, **subset)
    downloader.download()


if __name__ == "__main__":
    create_example_dataset("world_2020_2060_2100")
    ds = get_example_dataset("world_2020_2060_2100")
    print(ds)
