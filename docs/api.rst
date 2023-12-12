API
===

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

   xarray.Dataset.climepi.annual_mean
   xarray.Dataset.climepi.copy_bnds_from
   xarray.Dataset.climepi.copy_var_attrs_from
   xarray.Dataset.climepi.ensemble_mean
   xarray.Dataset.climepi.ensemble_mean_std_max_min
   xarray.Dataset.climepi.ensemble_percentiles
   xarray.Dataset.climepi.ensemble_stats
   xarray.Dataset.climepi.get_non_bnd_data_vars
   xarray.Dataset.climepi.plot_ensemble_ci_time_series
   xarray.Dataset.climepi.plot_map
   xarray.Dataset.climepi.plot_time_series
   xarray.Dataset.climepi.sel_data_var
   xarray.Dataset.climepi.sel_geopy

Climate data subpackage
-----------------------

.. currentmodule:: climepi

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/

   climepi.climdata.cesm.CESMDataGetter

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   climepi.climdata.cesm.create_example_dataset
   climepi.climdata.cesm.get_example_dataset
   climepi.climdata.cesm.get_cesm_data

Epidemiological model subpackage
--------------------------------

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/

   climepi.epimod.EpiModel
   climepi.epimod.EpiModDatasetAccessor
   climepi.epimod.ecolniche.EcolNicheModel

Methods
~~~~~~~

.. currentmodule:: xarray

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   xarray.Dataset.epimod.months_suitable
   xarray.Dataset.epimod.run_model

.. currentmodule:: climepi

.. autosummary::
   :toctree: generated/

   climepi.epimod.ecolniche.get_kaye_model

Front-end application subpackage
--------------------------------

If the climepi package is installed into the current python virtual environment, the
application can be initiated from the command line with ``python -m climepi.app``.

Methods
~~~~~~~
