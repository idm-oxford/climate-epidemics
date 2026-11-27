Front-end application
---------------------

climepi provides a browser-based front-end application, built using the `Panel
<https://panel.holoviz.org/>`_ library. A web application is available at
https://idm-oxford.github.io/climate-epidemics/app. A method to run the application
locally is contained in the ``app`` subpackage:

.. code-block:: python

    from climepi import app

    app.run_app(enable_climate_download=True, enable_local_files=True)

The example datasets include ``era5_london_2020_daily`` for daily ERA5 temperature
and precipitation at one grid point in London in 2020.

If climepi package is installed within the current `conda
<https://docs.conda.io/projects/conda/en/stable/index.html>`_ environment, the
application can also be run from the command line:

.. code-block:: console

    (<ENV_NAME>) $ python -m climepi.app

In the local app, select **Retrieve online data** to request data from any source
supported by :func:`~climepi.climdata.get_climate_data`. Choose the frequency, years,
and a point, place name, or bounding box. You can also select scenarios, models, and
realizations where applicable. Use **Download NetCDF** to save the data, or **Use
dataset in app** to plot them and run a model. When a place name is searched for, the
app shows the address found so that you can check it is the intended place.
Some requests can take several minutes and GLENS may need to download full source
files before subsetting. If ISIMIP is still preparing a subset, the app links to the
ISIMIP job so that you can check its progress and retrieve the same selection later.

Select **Open local files** to open your own NetCDF files, such as data saved earlier
with **Download NetCDF**. **Choose files** and **Choose folder** show your computer's
file dialog, or you can enter the path of a file or folder instead. A folder contributes
the NetCDF files directly inside it, and several files are combined into one dataset by
their coordinates, for example one file per variable or period. The data need time,
latitude and longitude coordinates. The app reads the files where they are and does not
copy or change them. Retrieved and local data stay available under **Climate data
option** for the rest of the session. The hosted app offers the example datasets.

The application uses `Dask <https://docs.dask.org/en/latest/>`_ to lazily operate on
climate data and parallelize computations. By default, the application will use the
thread-based single-machine Dask scheduler. To instead use the distributed Dask
scheduler (which may be slower in simple use cases but is more robust if running
multiple instances of the application simultaneously), first start a local Dask cluster
by running

.. code-block:: console

    (<ENV_NAME>) $ python -m climepi.app.cluster

Then, from a separate terminal, initiate the application with the command

.. code-block:: console

    (<ENV_NAME>) $ python -m climepi.app --dask-distributed

We welcome requests to improve the application or provide more advanced configuration
options (see :doc:`/development/contributing`).
