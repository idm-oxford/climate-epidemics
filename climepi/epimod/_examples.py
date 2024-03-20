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
        + "/example_data/kaye_ae_aegypti_niche.nc",
    },
    "mordecai_ae_aegypti_range": {  # from https://doi.org/10.1371/journal.pntd.0005568
        "temperature_range": [17.8, 34.6],
    },
    "mordecai_ae_aegypti_table": {  # from https://doi.org/10.1371/journal.pntd.0005568
        "temperature_vals": np.arange(18, 37),
        "suitability_vals": np.array(
            [
                0,
                0.02,
                0.04,
                0.09,
                0.16,
                0.27,
                0.40,
                0.56,
                0.72,
                0.86,
                0.96,
                1,
                0.97,
                0.85,
                0.67,
                0.44,
                0.20,
                0.02,
                0,
            ]
        ),
    },
    "mordecai_ae_albopictus_range": {
        # from https://doi.org/10.1371/journal.pntd.0005568
        "temperature_range": [16.2, 31.6],
    },
    "mordecai_ae_albopictus_table": {
        # from https://doi.org/10.1371/journal.pntd.0005568
        "temperature_vals": np.arange(16, 35),
        "suitability_vals": np.array(
            [
                0,
                0.01,
                0.03,
                0.07,
                0.15,
                0.26,
                0.41,
                0.58,
                0.76,
                0.91,
                0.99,
                0.98,
                0.85,
                0.61,
                0.33,
                0.11,
                0.03,
                0.01,
                0,
            ]
        ),
    },
    "ryan_ae_aegypti_975": {
        # from https://doi.org/10.1371/journal.pntd.0007213
        "temperature_range": [21.3, 34.0],
    },
    "ryan_ae_albopictus_975": {
        # from https://doi.org/10.1371/journal.pntd.0007213
        "temperature_range": [19.9, 29.4],
    },
    "villena_an_stephensi_p_falciparum_975": {
        # from https://doi.org/10.1002/ecy.3685
        "temperature_range": [16.0, 36.5],
    },
    "villena_an_stephensi_p_vivax_975": {
        # from https://doi.org/10.1002/ecy.3685
        "temperature_range": [16.6, 31.7],
    },
    "taylor_hlb_range": {
        # from https://doi.org/10.1111/1365-2664.13455
        "temperature_range": [16, 33],
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
        Ae. aegypti from https://doi.org/10.1101/2023.08.31.23294902),
        "mordecai_ae_aegypti" (the temperature suitability model for Ae. aegypti from
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
