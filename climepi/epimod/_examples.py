"""
Module for creating and accessing example climate-sensitive epidemiological models.
"""

import pathlib

import numpy as np
import xarray as xr

from climepi import epimod

EXAMPLES = {
    "kaye_ae_aegypti": {  # from https://doi.org/10.1101/2023.08.31.23294902
        "suitability_table_path": str(pathlib.Path(__file__).parent)
        + "/example_data/kaye_ae_aegyptae_niche.nc",
    },
    "mordecai_ae_aegyptae_range": {  # from https://doi.org/10.1371/journal.pntd.0005568
        "temperature_range": [17.8, 34.6],
    },
    "mordecai_ae_aegyptae_table": {  # from https://doi.org/10.1371/journal.pntd.0005568
        "temperature_vals": np.linspace(17.8, 34.6, 100),
        "suitability_vals": np.ones(100),
    },
    "mordecai_ae_albopictus": {
        # from https://doi.org/10.1371/journal.pntd.0005568
        "temperature_range": [16.2, 31.6],
    },
}
EXAMPLE_NAMES = list(EXAMPLES.keys())


def get_example_model(name):
    """
    Returns a climepi.epimod.EpiModel object for an example climate-sensitive
    epidemiological model.

    Parameters:
    -----------
    name : str
        The name of the example model to return. Currently available examples are:
        "kaye_ae_aegypti" (the temperature and rainfall suitability model for
        Ae. aegyptae from https://doi.org/10.1101/2023.08.31.23294902),
        "mordecai_ae_aegyptae" (the temperature suitability model for Ae. aegyptae from
        https://doi.org/10.1371/journal.pntd.0005568), and "mordecai_ae_albopictus" (the
        temperature suitability model for Ae. albopictus from
        https://doi.org/10.1371/journal.pntd.0005568).

    Returns:
    --------
    epi_model : climepi.epimod.EpiModel
        An instance of the EpiModel class representing the example model.
    """
    example_details = _get_example_details(name)
    if "suitability_table_path" in example_details:
        suitability_table = xr.open_dataset(example_details["suitability_table_path"])
        epi_model = epimod.SuitabilityModel(suitability_table=suitability_table)
    elif "temperature_range" in example_details:
        epi_model = epimod.SuitabilityModel(
            temperature_range=example_details["temperature_range"]
        )
    elif (
        "temperature_vals" in example_details and "suitability_vals" in example_details
    ):
        suitability_table = xr.Dataset(
            {
                "suitability": (
                    ["temperature"],
                    example_details["suitability_vals"],
                )
            },
            coords={"temperature": example_details["temperature_vals"]},
        )
        epi_model = epimod.SuitabilityModel(suitability_table=suitability_table)
    return epi_model


def _get_example_details(name):
    # Helper function for extracting the details of an example model from the
    # EXAMPLES dictionary in this module, and raising a customised error message
    # listing the available examples if the requested example is not found.
    try:
        example_details = EXAMPLES[name]
    except KeyError as exc:
        raise ValueError(
            f"Example model '{name}' not found. Available examples are: "
            f"{', '.join(EXAMPLES.keys())}"
        ) from exc
    return example_details
