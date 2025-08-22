API reference
=============

Accessor class for xarray datasets
----------------------------------

The ``climepi`` accessor class for ``xarray.Dataset`` objects can be used by chaining
the ``climepi`` attribute to a ``Dataset`` (e.g., ``ds.climepi.sel_geo("London")``; see
:ref:`getting-started/overview:functionality/climepi-accessor`).

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/

   climepi.ClimEpiDatasetAccessor

Methods
~~~~~~~

.. currentmodule:: xarray

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Dataset.climepi.ensemble_stats
   Dataset.climepi.monthly_average
   Dataset.climepi.plot_map
   Dataset.climepi.plot_time_series
   Dataset.climepi.plot_uncertainty_interval_decomposition
   Dataset.climepi.plot_variance_decomposition
   Dataset.climepi.run_epi_model
   Dataset.climepi.sel_geo
   Dataset.climepi.temporal_group_average
   Dataset.climepi.uncertainty_interval_decomposition
   Dataset.climepi.variance_decomposition
   Dataset.climepi.yearly_average
   Dataset.climepi.yearly_portion_suitable

Climate data subpackage
-----------------------

.. currentmodule:: climepi

Methods for downloading and accessing climate projection data are contained in the
``climdata`` subpackage (accessible via ``from climepi import climdata``; see
:ref:`getting-started/overview:functionality/climdata`).

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   climdata.get_climate_data
   climdata.get_climate_data_file_names
   climdata.get_example_dataset

Epidemiological model subpackage
--------------------------------

Classes and methods for climate-sensitive epidemiological models are contained in the
``epimod`` subpackage (accessible via ``from climepi import epimod``; see
:ref:`getting-started/overview:functionality/epimod`).

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/

   epimod.EpiModel
   epimod.SuitabilityModel
   epimod.ParameterizedSuitabilityModel

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   epimod.get_example_model
   epimod.get_example_temperature_response_data
   epimod.fit_temperature_response
   epimod.get_posterior_temperature_response

Front-end application subpackage
--------------------------------

A method to run the front-end application is contained in the ``app`` subpackage
(accessible via ``from climepi import app``; see
:ref:`getting-started/overview:Front-end application`).

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   app.run_app
