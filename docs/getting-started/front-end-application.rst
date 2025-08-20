Front-end Application
=====================

This package provides a browser-based front-end application, built using the ``Panel``
library.

A web application is available at https://idm-oxford.github.io/climate-epidemics/app.
If the ``climepi`` package is installed within the current ``conda`` environment, the
application can also be run locally. To initiate the application from the command line,
run

.. code-block:: console

    (<ENV_NAME>) $ python -m climepi.app

The application uses ``Dask`` to lazily operate on climate data and parallelize
computations. By default, the application will use the thread-based single-machine Dask
scheduler. To instead use the distributed Dask scheduler (which may be slower in simple
use cases but is more robust if running multiple instances of the application
simultaneously), first start a local Dask cluster by running

.. code-block:: console

    (<ENV_NAME>) $ python -m climepi.app.cluster

Then, from a separate terminal, initiate the application with the command

.. code-block:: console

    (<ENV_NAME>) $ python -m climepi.app --dask-distributed

A method is also provided to run the application from within a Python script (see
:ref:`api:Front-end application subpackage`). If more advanced configuration options are
required, please contact the developers.