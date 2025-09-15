"""Module for creating/accessing example climate-sensitive epidemiological models."""

import pathlib
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from climepi.epimod._base_classes import SuitabilityModel

EXAMPLES: dict[str, dict[str, Any]] = {
    "mordecai_ae_aegypti_niche": {
        "temperature_range": [17.8, 34.6],
        "doc": "Posterior mean temperature range of suitability for virus transmission "
        "by Ae. aegypti from Mordecai et al., PLOS Negl Trop Dis, 2017 "
        "(https://doi.org/10.1371/journal.pntd.0005568).",
    },
    "mordecai_ae_albopictus_niche": {
        "temperature_range": [16.2, 31.6],
        "doc": "Posterior mean temperature range of suitability for virus transmission "
        "by Ae. albopictus from Mordecai et al., PLOS Negl Trop Dis, 2017 "
        "(https://doi.org/10.1371/journal.pntd.0005568).",
    },
    "ryan_ae_aegypti_niche": {
        "temperature_range": [21.3, 34.0],
        "doc": "Temperature range of 97.5% posterior probability of suitability for "
        "virus transmission by Ae. aegypti from Ryan et al., PLOS Negl Trop Dis, 2019 "
        "(https://doi.org/10.1371/journal.pntd.0007213).",
    },
    "ryan_ae_albopictus_niche": {
        "temperature_range": [19.9, 29.4],
        "doc": "Temperature range of 97.5% posterior probability of suitability for "
        "virus transmission by Ae. albopictus from Ryan et al., PLOS Negl Trop Dis, "
        "2019 (https://doi.org/10.1371/journal.pntd.0007213).",
    },
    "kaye_ae_aegypti_niche": {
        "suitability_table_path": str(pathlib.Path(__file__).parent)
        + "/_example_data/kaye_ae_aegypti_niche.nc",
        "doc": "Median temperature/rainfall suitability region for Ae. aegypti from "
        "Kaye et al., Lancet Planet Health, 2024 "
        "(https://doi.org/10.1016/S2542-5196(24)00238-9).",
    },
    "yang_ae_aegypti_niche": {
        "temperature_range": [13.6, 36.55],
        "doc": "Temperature range in which the offspring number of Ae. aegypti is at "
        "least one from Yang et al., Epidemiol Infect, 2009 "
        "(https://doi.org/10.1017/S0950268809002040).",
    },
    "mordecai_ae_aegypti_suitability": {
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
        "doc": "Relative (posterior mean) suitability of different temperatures for "
        "virus transmission by Ae. aegypti from Mordecai et al., PLOS Negl Trop Dis, "
        "2017 (https://doi.org/10.1371/journal.pntd.0005568).",
    },
    "mordecai_ae_albopictus_suitability": {
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
        "doc": "Relative (posterior mean) suitability of different temperatures for "
        "virus transmission by Ae. albopictus from Mordecai et al., PLOS Negl Trop "
        "Dis, 2017 (https://doi.org/10.1371/journal.pntd.0005568).",
    },
    "villena_an_stephensi_p_falciparum_niche": {
        "temperature_range": [15.3, 37.2],
        "doc": "Posterior median temperature range of suitability for P. falciparum "
        "malaria parasite transmission by An. stephensi from Villena et al., Ecology, "
        "2022 (https://doi.org/10.1002/ecy.3685).",
    },
    "villena_an_stephensi_p_vivax_niche": {
        "temperature_range": [15.7, 32.5],
        "doc": "Posterior median temperature range of suitability for P. vivax malaria "
        "parasite transmission by An. stephensi from Villena et al., Ecology, 2022 "
        "(https://doi.org/10.1002/ecy.3685).",
    },
    "villena_an_gambiae_p_falciparum_niche": {
        "temperature_range": [19.1, 30.1],
        "doc": "Posterior median temperature range of suitability for P. falciparum "
        "malaria parasite transmission by An. gambiae from Villena et al., Ecology, "
        "2022 (https://doi.org/10.1002/ecy.3685).",
    },
    "villena_an_gambiae_p_vivax_niche": {
        "temperature_range": [19.2, 31.7],
        "doc": "Posterior median temperature range of suitability for P. vivax malaria "
        "parasite transmission by An. gambiae from Villena et al., Ecology, 2022 "
        "(https://doi.org/10.1002/ecy.3685).",
    },
    "taylor_hlb_range_niche": {
        "temperature_range": [16, 33],
        "doc": "Approximate temperature range of suitability for HLB transmission by "
        "D. citri from Taylor et al., J Appl Ecol, 2019"
        "(https://doi.org/10.1111/1365-2664.13455).",
    },
    "parham_anopheles_niche": {
        "temperature_range": [12.1606, 40],
        "precipitation_range": [0.0001, 50],
        "doc": "Temperature and precipitation ranges of suitability for Anopheles "
        "mosquitoes from Parham et al., 2010 "
        "(https://doi.org/10.1007/978-1-4419-6064-1_13), as reconstructed by Kaye et "
        "al. (https://doi.org/10.1016/S2542-5196(24)00238-9).",
    },
}
EXAMPLE_NAMES = list(EXAMPLES.keys())


def get_example_model(name: str) -> SuitabilityModel:
    """
    Get an example climate-sensitive epidemiological model.

    Returns a :class:`~climepi.epimod.SuitabilityModel` object for the example model
    specified by the ``name`` argument.

    Parameters
    ----------
    name : str
        The name of the example model to return. A list of available example names can
        be accessed as ``climepi.epimod.EXAMPLE_NAMES``, and a description of each
        example can be accessed via the ``climepi.epimod.EXAMPLES`` dictionary (e.g.,
        ``climepi.epimod.EXAMPLES['mordecai_ae_aegypti_niche']['doc']``).

    Returns
    -------
    epi_model : climepi.epimod.SuitabilityModel
        The example model.
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
        precipitation_range = example_details["precipitation_range"]
        temperature_diff = temperature_range[1] - temperature_range[0]
        temperature_vals = temperature_range[0] + temperature_diff * np.arange(
            -0.005, 1.01, 0.01
        )
        precipitation_diff = precipitation_range[1] - precipitation_range[0]
        precipitation_vals = precipitation_range[0] + precipitation_diff * np.arange(
            -0.005, 1.01, 0.01
        )
        suitability_vals = np.ones(
            (len(temperature_vals), len(precipitation_vals)), dtype=bool
        )
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


def get_example_temperature_response_data(name: str) -> pd.DataFrame:
    """
    Get example temperature response data.

    Available datasets are 'mordecai_ae_aegypti' (data used to fit the Aedes aegypti
    model in Mordecai et al., PLoS Negl Trop Dis 2017,
    https://doi.org/10.1371/journal.pntd.0005568) and 'mordecai_ae_albopictus' (data
    used to fit the Aedes albopictus model in the same paper).

    Parameters
    ----------
    name : str
        The name of the example temperature response data to return. Available names
        are 'mordecai_ae_aegypti' and 'mordecai_ae_albopictus'.

    Returns
    -------
    pandas.DataFrame
        The temperature response data. Has columns 'trait_name', 'temperature' (for the
        temperature values), 'trait_value' (for the corresponding trait values), and
        'reference' (giving original source information for the data).
    """
    if name == "mordecai_ae_aegypti":
        return pd.read_csv(
            pathlib.Path(__file__).parent / "_example_data/mordecai_ae_aegypti_data.csv"
        )
    if name == "mordecai_ae_albopictus":
        return pd.read_csv(
            pathlib.Path(__file__).parent
            / "_example_data/mordecai_ae_albopictus_data.csv"
        )
    raise ValueError(
        f"Example response data '{name}' not found. Available examples are: "
        "'mordecai_ae_aegypti', 'mordecai_ae_albopictus'."
    )


def _get_example_details(name: str) -> dict[str, Any]:
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
