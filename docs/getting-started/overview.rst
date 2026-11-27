Overview
========

climepi is a Python package for combining ensemble climate projections with models of
climate suitability for vector-borne diseases (VBDs) to analyze the impacts of climate
change and its uncertainties on VBD risks. climepi is designed with a modular
architecture, allowing users to apply and extend its functionality to a wide range of
climate datasets and climate-sensitive epidemiological models.

Core dependencies
-----------------

climepi relies heavily on the `xarray <https://xarray.dev/>`_ package for working with
labelled multi-dimensional arrays and datasets. We highly recommend familiarizing
yourself with the `xarray documentation <https://docs.xarray.dev/en/stable/>`_ before
using climepi.

Other key dependencies include:

- `Dask <https://docs.dask.org/en/latest/>`_: used under the hood by xarray for
  parallel computing and handling large datasets (see
  https://docs.xarray.dev/en/stable/user-guide/dask.html).
- `xCDAT <https://xcdat.readthedocs.io/en/latest/>`_: a package for climate data
  analysis with xarray, providing (among other features) methods for temporal
  averaging and bounds handling that are wrapped by climepi.
- `pymc <https://www.pymc.io/en/stable/>`_: used for parameterizing climate-VBD
  suitability models by fitting temperature response curves of vector and pathogen
  traits to laboratory data.
- `hvPlot <https://hvplot.holoviz.org/en/docs/latest/>`_: a high-level plotting API
  which is used within climepi's plotting methods.
- `Panel <https://panel.holoviz.org/>`_: used to build the browser-based front-end
  application.

Conventions
-----------

climepi's features assume that climate datasets are represented as xarray
:py:class:`~xarray.Dataset` objects and follow certain naming conventions:

- Data variables describing temperature and precipitation values (where present) should
  be named 'temperature' and 'precipitation', respectively. Inbuilt example
  epidemiological models assume temperature values in °C and precipitation values in
  mm/day, respectively.
- The time dimension should be named 'time'.
- Longitude and latitude coordinates should be named 'lon' and 'lat', respectively. Note
  that some climepi methods return datasets with a dimension 'location' indexing
  named locations.
- For ensemble datasets, multiple realizations of a single climate model under a single
  scenario should be indexed by a dimension named 'realization' (typically provided
  as integer values starting from 0). If multiple models and/or scenarios are included,
  they should be indexed by dimensions named 'model' and 'scenario', respectively.
- If temporal bounds are present, they should be included as a data variable named
  'time_bnds', with dimensions 'time' and 'bnds'.

We welcome requests or contributions for relaxing these requirements (for example, using
`cf_xarray <https://cf-xarray.readthedocs.io/en/latest/>`_) or converting other data
structures (such as climate time series stored as `pandas <https://pandas.pydata.org/>`_
:py:class:`~pandas.DataFrame` objects) to compliant xarray :py:class:`~xarray.Dataset`
objects (see :doc:`/development/contributing`).

Functionality
-------------

.. _`getting-started/overview:functionality/climdata`:

Climate data subpackage
~~~~~~~~~~~~~~~~~~~~~~~

The ``climdata`` subpackage contains methods for retrieving climate projections and
historical reanalysis. It provides a single interface for obtaining key climate
variables used in climate-VBD models (currently, temperature and precipitation) from
various sources. Currently supported data sources are:

- `LENS2 <https://www.cesm.ucar.edu/community-projects/lens2>`_ (Community Earth System
  Model version 2 Large Ensemble), which provides 100 ensemble members for analyzing
  internal climate variability.
- `ISIMIP <https://www.isimip.org/>`_ (Inter-Sectoral Impact Model Intercomparison
  Project), which provides bias-adjusted CMIP6 and CMIP7 projections
  (``isimip`` and ``isimip4b``), historical GSWP3-ERA5 forcing (``isimip4a``),
  and paired observed and detrended GSWP3-W5E5 forcing for attribution
  (``isimip3a``).
- `ARISE-SAI <https://www.cesm.ucar.edu/community-projects/arise-sai/>`_ (Assessing
  Responses and Impacts of Solar intervention on the Earth system with Stratospheric
  Aerosol Injection), which supports analysis of the impacts of solar climate
  intervention.
- `GLENS <https://www.cesm.ucar.edu/community-projects/glens>`_ (Geoengineering Large
  Ensemble), which also supports analysis of the impacts of solar climate intervention.
- `ERA5 <https://doi.org/10.24381/cds.adbb2d47>`_ historical reanalysis on a global
  0.25° grid, available without an account.

For example, LENS2 data for London and Paris from 2030 to 2100 for the first two
ensemble members can be retrieved as follows:

.. code-block:: python

    from climepi import climdata

    ds_clim = climdata.get_climate_data(
        data_source="lens2",
        frequency="daily",
        subset={
            "years": list(range(2030, 2101)),
            "locations": ["London", "Paris"],
            "realizations": [0, 1],
        },
    )

The same interface retrieves historical ERA5 data for named locations or gridded
subsets from the `public Icechunk archive
<https://registry.opendata.aws/earthmover-era5/>`_. ERA5 retrieval requires
``icechunk`` and ``pcodec``, which are included in the Pixi environment.
Providing coordinates skips geocoding.

.. code-block:: python

    ds_era5 = climdata.get_climate_data(
        "era5",
        frequency="monthly",
        subset={
            "years": [2020],
            "locations": ["Oxford"],
            "lons": [-1.26],
            "lats": [51.75],
        },
    )

ISIMIP subsets are prepared by the repository before download by default.
Set ``subset_method="local"`` to skip the subsetting queue and download the
original global files. This can be practical for large regions or when the
queue is busy, but even a request for one year can transfer several GB per
model and scenario. Both methods reuse previously processed data.

See the documentation for
:py:meth:`climdata.get_climate_data() <climepi.climdata.get_climate_data()>` for details
of data sources and subsetting options. Example datasets are also provided (see the
documentation for
:py:meth:`climdata.get_example_dataset() <climepi.climdata.get_example_dataset()>`).

.. _`getting-started/overview:functionality/epimod`:

Epidemiological model subpackage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``epimod`` subpackage contains classes and methods for climate-sensitive
epidemiological models, particularly models describing climate suitability for VBDs.
For example, a simple model in which transmission is possible within a certain
temperature range (in the example below, in the range 15-30 degrees celsius) can be defined as follows:

.. code-block:: python

    from climepi import epimod

    suitability_model = epimod.SuitabilityModel(temperature_range=(15, 30))

Running the model on a climate dataset (either using :py:meth:`suitability_model.run()
<climepi.epimod.SuitabilityModel.run()>`, or via the ``climepi`` accessor as described
below) will then yield a dataset with a Boolean data variable 'suitability' indicating
whether or not each temperature value falls within the specified range).

Methods are also provided for inferring temperature responses of vector and pathogen
traits in order to construct suitability models in the
:py:class:`~climepi.epimod.ParameterizedSuitabilityModel` class. See :doc:`/gallery` for
detailed usage examples.

.. _`getting-started/overview:functionality/climepi-accessor`:

Accessor class for xarray datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``climepi`` accessor class for :py:class:`~xarray.Dataset` objects provides methods 
for running epidemiological models on climate datasets, and for analyzing and
visualizing the impact of climate uncertainty sources, as well as other utility methods.

The accessor class can be used by chaining the ``climepi`` attribute to a
:py:class:`~xarray.Dataset`. For example, the
:py:meth:`~xarray.Dataset.climepi.run_epi_model()` method can be used to run an
epidemiological model on a climate dataset as follows:

.. code-block:: python

    import climepi # noqa

    ds_epi = ds_clim.climepi.run_epi_model(suitability_model)

Acknowledgement
---------------

This package has been developed as part of a project funded by a Digital Technology
Development Award (Climate-Sensitive Infectious Disease Modelling) from
`Wellcome <https://wellcome.org/>`_ (grant number 226057/Z/22/Z).

.. image:: /_static/wellcome-logo-black.png
   :alt: Wellcome logo
   :scale: 40 %
   :align: left
