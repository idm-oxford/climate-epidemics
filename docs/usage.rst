Usage
=====

Installation
------------

To use climepi, download the source code from the github repository
(https://github.com/will-s-hart/climate-epidemics). The package and its dependencies can
then be installed via ``conda`` into a new virtual environment using the provided
environment.yml file: working in the repository root directory, run

.. code-block:: console

    climate-epidemics $ conda env create -f environment.yml

.. note::
    Using ``mamba`` instead of ``conda`` may substantially speed up the installation
    process.

The virtual environment can be activated using the following
command:   

.. code-block:: console
    
    climate-epidemics $ conda activate climepi

Front-end application
---------------------

This package provides a browser-based front-end application, built using the ``Panel``
library.

A web application is available at https://will-s-hart.github.io/climate-epidemics/app.
If the ``climepi`` package is installed within the current python virtual environment,
theapplication can also be run locally. To initiate the application from the command
line, run

.. code-block:: console

    (climepi) climate-epidemics $ python -m climepi.app

In this case, the application will use the default thread-based single-machine Dask
scheduler to run computations. To instead use the distributed Dask scheduler (which may
be slower in simple use cases but is more robust if running multiple instances of the
application simultaneously), first start a local Dask cluster by running

.. code-block:: console

    (climepi) climate-epidemics $ python -m climepi.app.cluster

Then, from a separate terminal, initiate the application with the command

.. code-block:: console

    (climepi) climate-epidemics $ python -m climepi.app --dask-distributed

A method is also provided to run the application from within a Python script (see
:ref:`api:Front-end application subpackage`).