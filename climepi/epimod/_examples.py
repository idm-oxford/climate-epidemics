"""Module for creating/accessing example climate-sensitive epidemiological models."""

import pathlib

import numpy as np
import xarray as xr

from climepi.epimod._model_classes import SuitabilityModel

EXAMPLES = {
    "mordecai_ae_aegypti_niche": {  # from https://doi.org/10.1371/journal.pntd.0005568
        "temperature_range": [17.8, 34.6],
    },
    "mordecai_ae_albopictus_niche": {
        # from https://doi.org/10.1371/journal.pntd.0005568
        "temperature_range": [16.2, 31.6],
    },
    "ryan_ae_aegypti_niche": {
        # from https://doi.org/10.1371/journal.pntd.0007213
        "temperature_range": [21.3, 34.0],
    },
    "ryan_ae_albopictus_niche": {
        # from https://doi.org/10.1371/journal.pntd.0007213
        "temperature_range": [19.9, 29.4],
    },
    "kaye_ae_aegypti_niche": {  # from https://doi.org/10.1101/2023.08.31.23294902
        "suitability_table_path": str(pathlib.Path(__file__).parent)
        + "/_example_data/kaye_ae_aegypti_niche.nc",
    },
    "yang_ae_aegypti_niche": {  # from https://doi.org/10.1017/S0950268809002040,
        # range where offspring number is at least 1
        "temperature_range": [13.6, 36.55],
    },
    "mordecai_ae_aegypti_suitability": {  # from https://doi.org/10.1371/journal.pntd.0005568
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
    "mordecai_ae_albopictus_suitability": {
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
    "villena_an_stephensi_p_falciparum_niche": {
        # from https://doi.org/10.1002/ecy.3685
        "temperature_range": [16.0, 36.5],
    },
    "villena_an_stephensi_p_vivax_niche": {
        # from https://doi.org/10.1002/ecy.3685
        "temperature_range": [16.6, 31.7],
    },
    "taylor_hlb_range_niche": {
        # from https://doi.org/10.1111/1365-2664.13455
        "temperature_range": [16, 33],
    },
    "parham_anopheles_niche": {  # from https://doi.org/10.1007/978-1-4419-6064-1_13
        "temperature_range": [12.1606, 40],
        "precipitation_range": [0.001, 50],
    },
}
EXAMPLE_NAMES = list(EXAMPLES.keys())


def get_example_model(name):
    """
    Get an example climate-sensitive epidemiological model.

    Returns a climepi.epimod.EpiModel object for the example model specified by the
    name argument.

    Parameters
    ----------
    name : str
        The name of the example model to return. Currently available examples are:
        "kaye_ae_aegypti" (the temperature and rainfall suitability model for
        Ae. aegypti from https://doi.org/10.1101/2023.08.31.23294902),
        "mordecai_ae_aegypti" (the temperature suitability model for Ae. aegypti from
        https://doi.org/10.1371/journal.pntd.0005568), and "mordecai_ae_albopictus" (the
        temperature suitability model for Ae. albopictus from
        https://doi.org/10.1371/journal.pntd.0005568).

    Returns
    -------
    epi_model : climepi.epimod.EpiModel
        An instance of the EpiModel class representing the example model.
    """
    example_details = _get_example_details(name)
    if "suitability_table_path" in example_details:
        suitability_table = xr.open_dataset(example_details["suitability_table_path"])
        epi_model = SuitabilityModel(suitability_table=suitability_table)
    elif (
        "temperature_range" in example_details
        and "precipitation_range" in example_details
    ):
        # Create a suitability table with suitability 1 in the relevant ranges and 0
        # outside them (with the range limits equidistant from two adjacent grid points
        # to ensure the correct ranges are enforced with nearest-neighbour
        # interpolation).
        temperature_range = example_details["temperature_range"]
        temperature_diff = temperature_range[1] - temperature_range[0]
        temperature_vals = temperature_range[0] + temperature_diff * np.arange(
            -0.005, 1.01, 0.01
        )
        precipitation_range = example_details["precipitation_range"]
        precipitation_diff = precipitation_range[1] - precipitation_range[0]
        precipitation_vals = precipitation_range[0] + precipitation_diff * np.arange(
            -0.005, 1.01, 0.01
        )
        suitability_vals = np.ones((len(temperature_vals), len(precipitation_vals)))
        suitability_vals[0, :] = 0
        suitability_vals[-1, :] = 0
        suitability_vals[:, 0] = 0
        suitability_vals[:, -1] = 0
        suitability_table = xr.Dataset(
            {"suitability": (["temperature", "precipitation"], suitability_vals)},
            coords={
                "temperature": temperature_vals,
                "precipitation": precipitation_vals,
            },
        )
        suitability_table["suitability"].attrs = {"long_name": "Suitability"}
        suitability_table["temperature"].attrs = {
            "long_name": "Temperature",
            "units": "°C",
        }
        suitability_table["precipitation"].attrs = {
            "long_name": "Precipitation",
            "units": "mm/day",
        }
        epi_model = SuitabilityModel(suitability_table=suitability_table)

    elif "temperature_range" in example_details:
        epi_model = SuitabilityModel(
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
        suitability_table["suitability"].attrs = {"long_name": "Suitability"}
        suitability_table["temperature"].attrs = {
            "long_name": "Temperature",
            "units": "°C",
        }
        epi_model = SuitabilityModel(suitability_table=suitability_table)
    else:
        raise ValueError(
            f"Example model '{name}' does not have a recognised format. "
            "Please check the documentation for the expected format of example models."
        )
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
