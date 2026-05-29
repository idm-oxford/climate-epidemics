# climepi

[![Version](https://img.shields.io/conda/vn/conda-forge/climepi.svg)](https://anaconda.org/conda-forge/climepi)
[![Platform](https://img.shields.io/conda/pn/conda-forge/climepi.svg)](https://anaconda.org/conda-forge/climepi)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14888949.svg)](https://doi.org/10.5281/zenodo.14888949)
[![License](https://img.shields.io/github/license/idm-oxford/climate-epidemics.svg)](https://github.com/idm-oxford/climate-epidemics/blob/main/LICENSE)
[![Tests](https://github.com/idm-oxford/climate-epidemics/actions/workflows/run_tests.yml/badge.svg)](https://github.com/idm-oxford/climate-epidemics/actions/workflows/run_tests.yml?branch=main)
[![Coverage](https://codecov.io/gh/idm-oxford/climate-epidemics/branch/main/graph/badge.svg)](https://codecov.io/gh/idm-oxford/climate-epidemics)
[![Documentation](https://readthedocs.org/projects/climate-epidemics/badge/?version=latest)](https://climate-epidemics.readthedocs.io/en/latest)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://www.mypy-lang.org/)

Python package and web app for combining climate projection datasets and epidemiological models characterising climate suitability for vector-borne disease.

Documentation: https://climate-epidemics.readthedocs.io/en/latest/

Web app: https://idm-oxford.github.io/climate-epidemics/

## Installation

climepi is available on [conda-forge](https://anaconda.org/conda-forge/climepi);
see the [installation docs](https://climate-epidemics.readthedocs.io/en/latest/getting-started/installation.html)
for details.

## Quickstart

```python
import climepi  # registers the .climepi xarray accessor
from climepi import climdata, epimod

# Load an example climate projection dataset (single CESM2 LENS ensemble member for
# years 2020 and 2100; triggers a ~10 MB download on first run)
ds_clim = climdata.get_example_dataset("lens2_2020_2100_monthly_one_realization")

# Define a simple suitability model: transmission possible at 15-30 °C
suitability_model = epimod.SuitabilityModel(temperature_range=(15, 30))

# Run the epidemiological model on the climate dataset
ds_epi = ds_clim.climepi.run_epi_model(suitability_model)

# Compute the number of suitable months per year
ds_months = ds_epi.climepi.yearly_portion_suitable()

# Plot a map of suitable months in 2100
ds_months.sel(time="2100").climepi.plot_map()
```

See the [Gallery](https://climate-epidemics.readthedocs.io/en/latest/gallery.html) for
more in-depth examples.

