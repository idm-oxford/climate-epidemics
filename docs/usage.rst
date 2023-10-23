Usage
=====

Installation
------------

The required dependencies are only available through conda (or mamba). To use
climepi, download the source code. Working in the repository root directory,
create a conda virtual environment with the required dependencies using the
provided environment.yml file:

.. code-block:: console

    climate-epidemics $ conda env create -f environment.yml

The package can then be installed within the virtual environment:   

.. code-block:: console
    
    (climepi) climate-epidemics $ pip install .