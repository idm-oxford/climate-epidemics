---
title: 'climepi: A Python package for modeling climate suitability for vector-borne diseases'
tags:
  - Python
  - epidemiological modeling
  - infectious disease modeling
  - climate-sensitive infectious disease
  - vector-borne disease
authors:
  - name: William S. Hart
    orcid: 0000-0002-2504-6860
    affiliation: 1
  - name: Robin N. Thompson
    orcid: 0000-0001-8545-5212
    affiliation: 1
affiliations:
 - name: Mathematical Institute, University of Oxford, Oxford, OX2 6GG, UK
   index: 1
date: 29 September 2025
bibliography: paper.bib
---

# Summary

climepi (clim-epi) is a Python package for combining climate projections with models of
climate suitability for vector-borne diseases (VBDs). Utilizing the [xarray](
https://xarray.dev/) library [@hoyer_xarray_2017] for handling labeled multi-dimensional
arrays, climepi provides methods for: accessing climate projection data from a range of
sources; defining, parameterizing, and running models of climate suitability for
VBDs; and assessing the impacts of different sources of climate uncertainty
(uncertainty in emissions scenarios, structural uncertainty across climate models, and
natural climate fluctuations that occur alongside anthropogenic climate change). The
package also provides a front-end application that can be used to explore impacts of
climate change and its uncertainties using example climate datasets and climate-VBD
suitability models (a [web interface](https://idm-oxford.github.io/climate-epidemics/)
is available).

# Statement of need

Climate change is altering the dynamics of a range of infectious diseases, particularly
VBDs such as malaria and dengue [@ryan_global_2019]. Quantitative estimates of future
VBD risks are important for targeting vector and pathogen surveillance resources,
and for planning public health interventions. However, a recent review paper highlighted
that few operational software tools exist for assessing climate-sensitive disease risks
[@ryan_current_2023]. climepi contributes to addressing this gap by providing a flexible
and extensible Python package, as well as a user-friendly front-end application, which
can be used by climate-health researchers and other users (such as public health
professionals) to assess future climate suitability for VBDs and uncertainty therein.

# Main features

climepi is designed with a modular structure, comprising four components:

1. Climate data (`climdata`) subpackage: enables users to access climate projection data
   from different sources through a single interface. Rather than providing
   comprehensive access options for a large number of datasets, the focus is on
   facilitating straightforward access to data sources and climate variables
   (temperature and precipitation) that are particularly useful for analyzing the
   impacts of different types of climate uncertainty on future climate-VBD suitability.
   Supported data sources include the [Inter-Sectoral Impact Model Intercomparison
   Project](https://www.isimip.org/) (ISIMIP) [@lange_isimip3b_2021], which provides
   downscaled and bias-adjusted data from multiple [Coupled Model Intercomparison
   Project Phase 6](https://wcrp-cmip.org/cmip-phases/cmip6/) (CMIP6)
   [@eyring_overview_2016] climate models and emissions scenarios, and the [Community
   Earth System Model version 2 Large Ensemble](
   https://www.cesm.ucar.edu/community-projects/lens2) (LENS2) [@rodgers_ubiquity_2021],
   which provides 100 ensemble members for analyzing internal (natural) climate
   variability.

![Schematic illustrating typical workflows using climepi. (1) The `climdata` subpackage
can be used to access climate data from an external source, or to load a pre-defined
example climate dataset. (2) The `epimod` subpackage can be used to define a mechanistic
model describing climate suitability for VBD, infer temperature response curves of trait 
parameters using laboratory data (left plot), and determine the overall temperature
dependence of the suitability metric (right plot). Pre-defined epidemiological models
from the literature are also available as built-in examples. (3) The `climepi` accessor
for xarray datasets can be used to combine a climate dataset with an epidemiological
model of climate suitability for VBD to obtain suitability projections, to assess the contributions of different sources of climate uncertainty, and to visualize results. See
the package [documentation](
https://climate-epidemics.readthedocs.io/en/stable/gallery.html) for detailed usage
examples.](schematic.pdf)

2. Epidemiological model (`epimod`) subpackage: provides classes and methods for models
   of climate suitability for VBD, in which a suitability metric (e.g., the basic
   reproduction number, $R_0$) is defined as a function of temperature and/or
   precipitation. Methods are also provided for parameterizing mechanistic suitability
   models by fitting the temperature dependence of model parameters describing vector
   and pathogen traits to laboratory data [@mordecai_detecting_2017]. A selection of
   climate-VBD suitability models from the literature are also available as built-in
   examples
   [@kaye_impact_2024;@mordecai_detecting_2017;@parham_modelling_2010;@ryan_global_2019;@taylor_predicting_2019;@villena_temperature_2022;@yang_assessing_2009].
3. Accessor class for xarray datasets (`xarray.Dataset.climepi`): provides methods for
   combining climate data with epidemiological models and for assessing and visualizing
   the importance of different climate uncertainty sources
   [@hawkins_potential_2009;@hart_climate_2025], as well as other supporting functions.
4. Front-end application (`app`) subpackage: provides a method for running the front-end
   application locally.

# Example applications

In a recent study, we used climepi to assess the relative contributions of climate
scenario uncertainty and climate model uncertainty, as well as internal climate
variability, to uncertainty in future climate suitability for dengue virus transmission
in a range of locations that do not currently experience substantial outbreaks
[@hart_climate_2025]. The climepi [documentation](
https://climate-epidemics.readthedocs.io/en/stable/gallery.html) includes a detailed
usage example demonstrating how results from that study can be reproduced. Further
example pages show how climepi can be used to recreate results from two other studies
(that did not originally use climepi) [@mordecai_detecting_2017;@kaye_impact_2024]:
specifically, these examples involve parameterizing the temperature-dependent dengue
virus transmission suitability model developed by @mordecai_detecting_2017 and
recreating the analysis of the impact of internal climate variability on climate
suitability for *Aedes aegypti* (a vector of VBDs including dengue and yellow fever) by
@kaye_impact_2024.

# Acknowledgements

Thanks to members of the Infectious Disease Modelling group (Mathematical Institute,
University of Oxford) for useful discussions. This project was funded by Wellcome
(grant number 226057/Z/22/Z). The authors acknowledge the support of the JUNIPER
partnership (grant number MR/X018598/1).

# References
