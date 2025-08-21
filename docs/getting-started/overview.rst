Overview
========

climepi is a Python package for combining ensemble climate projections with models of
climate suitability for vector-borne diseases to analyze the impacts of climate change
and its uncertainties on disease risks.

Core dependencies
-----------------

climepi relies heavily on the ``xarray`` package for working with labelled
multi-dimensional arrays and datasets. We highly recommend familiarizing yourself
with the `xarray documentation <https://docs.xarray.dev/en/stable/>`_ before using
climepi.

Other key dependencies include:

- ``dask``: used under the hood by ``xarray`` for parallel computing and handling large
  datasets (see https://docs.xarray.dev/en/stable/user-guide/dask.html).
- ``xcdat``: a package for climate data analysis with ``xarray``, providing (among other
  features) methods for temporal averaging and bounds handling that are wrapped by
  climepi.
- ``hvplot``: a high-level plotting API which is used within climepi's plotting methods.
- ``panel``: used to build the browser-based front-end application.

Conventions
-----------

climepi is designed with a modular architecture, allowing users to apply and extend its
functionality to a wide range of climate datasets and climate-sensitive epidemiological
models. To ensure smooth uses, its features assume that climate datasets are represented
as ``xarray.Dataset`` objects and follow certain naming conventions:

- Data variables describing temperature and precipitation values (where present) should
  be named "temperature" and "precipitation", respectively. Inbuilt example
  epidemiological models assume temperature values in °C and precipitation values in
  mm/day, respectively.
- The time dimension should be named "time".

- Longitude and latitude coordinates should be named "lon" and "lat", respectively. Note
  that some climepi methods return datasets with a dimension "location" indexing
  named locations.
- For ensemble datasets, multiple realizations of a single climate model under a single
  scenario should be indexed by a dimension named "realization". If multiple models
  and/or scenarios are included, they should be indexed by dimensions named "model"
  and "scenario", respectively.
- If temporal bounds are present, they should be included as a data variable named
  "time_bnds", with dimensions "time" and "bnds".

We welcome requests or contributions for relaxing these requirements (for example, using
``cf_xarray``) or converting other data structures (such as climate time series stored
as ``pandas.DataFrame`` objects) to compliant ``xarray.Dataset`` objects (see
:doc:`/development/contributing`).

Functionality
-------------

.. _`getting-started/overview:functionality/climdata`:

Climate data subpackage
~~~~~~~~~~~~~~~~~~~~~~~

The ``climdata`` subpackage contains methods for retrieving climate projection data.
It provides a simple interface for obtaining projections of key climate variables
for climate-VBD modeling (currently, temperature and precipitation) from various sources
(currently supported data sources include the CESM LENS2 and ISIMIP projects).

.. code-block:: python

    from climepi import climdata

.. _`getting-started/overview:functionality/epimod`:

Epidemiological model subpackage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classes and methods for climate-sensitive epidemiological models are contained in the
``epimod`` subpackage:

.. code-block:: python

    from climepi import epimod

.. _`getting-started/overview:functionality/climepi-accessor`:

Dataset accessor
~~~~~~~~~~~~~~~~

The ``climepi`` accessor class for ``xarray.Dataset`` objects provides methods for
running epidemiological models on climate datasets, and for analyzing and visualizing
the impact of climate uncertainty sources, as well as other utility methods.

The accessor class can be used by chaining the ``climepi`` attribute to a ``Dataset``.
For example, the :py:meth:`~xarray.Dataset.climepi.run_epi_model()` method can be used
to run an epidemiological model on a climate dataset as follows:

.. code-block:: python

    import climepi # noqa

    ds_epi = ds_clim.climepi.run_epi_model()

Front-end application
---------------------

climepi provides a browser-based front-end application, built using the ``Panel`` 
library.

A web application is available at https://idm-oxford.github.io/climate-epidemics/app.
A method to run the application locally is contained in the ``app`` subpackage:

.. code-block:: python

    from climepi import app
    app.run_app()


If the ``climepi`` package is installed within the current ``conda`` environment, the
application can also be run from the command line:

.. code-block:: console

    (<ENV_NAME>) $ python -m climepi.app

The application uses Dask to lazily operate on climate data and parallelize
computations. By default, the application will use the thread-based single-machine Dask
scheduler. To instead use the distributed Dask scheduler (which may be slower in simple
use cases but is more robust if running multiple instances of the application
simultaneously), first start a local Dask cluster by running

.. code-block:: console

    (<ENV_NAME>) $ python -m climepi.app.cluster

Then, from a separate terminal, initiate the application with the command

.. code-block:: console

    (<ENV_NAME>) $ python -m climepi.app --dask-distributed

We welcome requests to improve the application or provide more advanced configuration
options (see :doc:`/development/contributing`).