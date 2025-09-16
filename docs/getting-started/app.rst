Front-end application
---------------------

climepi provides a browser-based front-end application, built using the `Panel
<https://panel.holoviz.org/>`_ library. A web application is available at
https://idm-oxford.github.io/climate-epidemics/app. A method to run the application
locally is contained in the ``app`` subpackage:

.. code-block:: python

    from climepi import app

    app.run_app()

If climepi package is installed within the current `conda
<https://docs.conda.io/projects/conda/en/stable/index.html>`_ environment, the
application can also be run from the command line:

.. code-block:: console

    (<ENV_NAME>) $ python -m climepi.app

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