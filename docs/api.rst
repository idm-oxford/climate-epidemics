API reference
=============

Core functionality
------------------

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

   xarray.Dataset.climepi.ensemble_stats
   xarray.Dataset.climepi.estimate_ensemble_stats
   xarray.Dataset.climepi.monthly_average
   xarray.Dataset.climepi.months_suitable
   xarray.Dataset.climepi.plot_map
   xarray.Dataset.climepi.plot_time_series
   xarray.Dataset.climepi.plot_uncertainty_interval_decomposition
   xarray.Dataset.climepi.plot_variance_decomposition
   xarray.Dataset.climepi.run_epi_model
   xarray.Dataset.climepi.sel_geo
   xarray.Dataset.climepi.temporal_group_average
   xarray.Dataset.climepi.uncertainty_interval_decomposition
   xarray.Dataset.climepi.variance_decomposition
   xarray.Dataset.climepi.yearly_average

Climate data subpackage
-----------------------

.. currentmodule:: climepi

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   climepi.climdata.get_climate_data
   climepi.climdata.get_climate_data_file_names
   climepi.climdata.get_example_dataset

Epidemiological model subpackage
--------------------------------

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/

   climepi.epimod.EpiModel
   climepi.epimod.SuitabilityModel

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   climepi.epimod.get_example_model

Front-end application subpackage
--------------------------------

See :ref:`usage` for information on how to run the front-end application from the
command line.

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   climepi.app.run_app
