API reference
=============

Core functionality
------------------

The ``climepi`` package provides an accessor class for ``xarray.Dataset`` objects,
which can be used by chaining the ``climepi`` attribute to a ``Dataset``. For example,
the :py:meth:`~xarray.Dataset.climepi.sel_geo` method can be used to select a named
location from a dataset containing data with latitude and longitude coordinates (named
"lat" and "lon", respectively) as follows:

.. code-block:: python

    import xarray as xr
    import climepi

    ds = xr.open_dataset("path/to/dataset.nc")
    ds.climepi.sel_geo("London")

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
   Dataset.climepi.estimate_ensemble_stats
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
``climdata`` subpackage:

.. code-block:: python

    from climepi import climdata

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   climdata.get_climate_data
   climdata.get_climate_data_file_names
   climdata.get_example_dataset

Epidemiological model subpackage
--------------------------------

Classes and methods for running climate-sensitive epidemiological models are contained
in the ``epimod`` subpackage:

.. code-block:: python

    from climepi import epimod

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/

   epimod.EpiModel
   epimod.SuitabilityModel

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   epimod.get_example_model

Front-end application subpackage
--------------------------------

A method to run the front-end application is contained in the ``app`` subpackage:

.. code-block:: python

    from climepi import app
    app.run_app()

See :ref:`usage:Front-end application` for information on how to run the application
from the command line.

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   app.run_app
