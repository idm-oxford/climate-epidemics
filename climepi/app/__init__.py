"""
Subpackage providing a browser-based front-end application that can be used to run and
visualize the output of climate-sensitive epidemiological models.

If the climepi package is installed into the current python virtual environment, the
application can be initiated from the command line with

.. code-block:: console
    
        python -m climepi.app

The application is built using the `Panel` library.
"""

from climepi.app._run_app import run_app  # noqa
