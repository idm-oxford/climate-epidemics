************
Contributing
************

.. note::
  Large portions of this document came from or are inspired by the `xCDAT Contributing
  Guide <https://xcdat.readthedocs.io/en/latest/contributing.html>`_.

Overview
--------

climepi is a community-driven open source project and we welcome everyone who would like
to contribute! Feel free to open a `GitHub Issue`_ for bug reports, bug fixes,
documentation improvements, enhancement suggestions, and other ideas.

Please note that climepi has a `Code of Conduct`_. By participating in the climepi
community, you agree to abide by its rules.

climepi is distributed under the `GNU General Public License v3.0`_.

Where to start?
---------------

If you are brand new to climepi or open-source development, we recommend going
through the `GitHub Issues`_ page to find issues that interest you.

Documentation updates
~~~~~~~~~~~~~~~~~~~~~

Contributing to the `documentation`_ is an excellent way to help climepi. Anything from
fixing typos to improving sentence structure or API docstrings/examples can go a long
way in making climepi more user-friendly.

Bug reports
~~~~~~~~~~~

Bug reports are an important part of making climepi more stable. Having a complete bug
report will allow others to reproduce the bug and provide insight into fixing.

Trying out the bug-producing code on the ``main`` branch is often a worthwhile exercise
to confirm that the bug still exists. It is also worth searching existing bug reports
and pull requests to see if the issue has already been reported and/or fixed.

Enhancement requests
~~~~~~~~~~~~~~~~~~~~

Enhancements are a great way to improve climepi's capabilities for the broader
scientific community.

If you are proposing an enhancement:

* Explain in detail how it would work
* Make sure it fits in the domain of geospatial climate science
* Keep the scope as narrow as possible, to make it easier to implement
* Generally reusable among the community (not model or data-specific)
* Remember that this is a open-source project, and that contributions are welcome :)

All other inquiries
~~~~~~~~~~~~~~~~~~~~

We welcome comments, questions, or feedback!

Version control, Git, and GitHub
--------------------------------

The code is hosted on `GitHub`_. To contribute you will need to sign up for a
`free GitHub account`_. We use `Git`_ for version control to allow many people to work
together on the project.

Some great resources for learning Git:

* the `GitHub help pages`_
* the `NumPy's documentation`_
* Matthew Brett's `Pydagogue`_

Getting started with Git
~~~~~~~~~~~~~~~~~~~~~~~~

`GitHub has instructions for setting up Git`_ including installing git,
setting up your SSH key, and configuring git.  All these steps need to be completed
before you can work seamlessly between your local repository and GitHub.

Set up the Repo
~~~~~~~~~~~~~~~

.. note::

    The following instructions assume you want to learn how to interact with GitHub via
    the git command-line utility, but contributors who are new to git may find it easier
    to use other tools instead such as `GitHub Desktop`_.

Once you have Git setup, you will need to fork the main repository then clone your fork:
   - Head over to the `GitHub`_ repository page, click the fork button, and click
     "Create a new fork".
   - Then clone your fork by running ``git clone https://github.com/<GITHUB-USERNAME>/climate-epidemics.git``

.. _GitHub has instructions for setting up Git: https://help.github.com/set-up-git-redirect
.. _templates: https://github.com/idm-oxford/climate-epidemics/issues/new/choose
.. _documentation: https://climate-epidemics.readthedocs.io/en/latest/
.. _GitHub Issues: https://github.com/idm-oxford/climate-epidemics/issues
.. _GitHub Issue: https://github.com/idm-oxford/climate-epidemics/issues
.. _GitHub Issues: https://github.com/idm-oxford/climate-epidemics/issues
.. _Code of Conduct: https://github.com/idm-oxford/climate-epidemics/blob/main/CODE_OF_CONDUCT.rst
.. _GitHub: https://www.github.com/idm-oxford/climate-epidemics
.. _free GitHub account: https://github.com/signup/free
.. _Git: http://git-scm.com/
.. _GitHub help pages: https://help.github.com/
.. _NumPy's documentation: https://numpy.org/doc/stable/dev/index.html
.. _Pydagogue: https://matthew-brett.github.io/pydagogue/
.. _GitHub Desktop: https://desktop.github.com/
.. _GNU General Public License v3.0: https://github.com/idm-oxford/climate-epidemics/blob/main/LICENSE

Creating a development environment
----------------------------------

Before starting any development, you'll need to create an isolated climepi development
environment. We recommend using the `pixi`_ package manager; from the
``climate-epidemics`` root directory, run:

  .. code-block:: bash

      >>> pixi install -e default

.. _pixi: https://pixi.sh/latest/

Contributing to the code base
-----------------------------

Pull request (PR)
~~~~~~~~~~~~~~~~~

Here's a simple checklist for PRs:

- **Properly comment and document your code.** API docstrings are formatted using the
  `NumPy style guide`_
- **Test that the documentation builds correctly** by typing ``pixi run docs`` in the 
  root of the ``climate-epidemics`` directory. This is not strictly necessary, but this
  may be easier than waiting for CI to catch a mistake.
- **Test your code**.

  - Write new tests if needed.
  - Test the code using `Pytest`_ (type ``pixi run test`` in the root directory to run
    all tests).

- **Properly format your code** and verify that it passes the formatting guidelines set
  by `Ruff`_ (type ``pixi run lint`` in the root directory to run the linter).

- **Push your code** and `create a PR on GitHub`_.
- **Use a helpful title for your pull request** by summarizing the main contributions
  rather than using the latest commit message. If the PR addresses a `GitHub Issue`_,
  please `reference it`_.

.. _code-formatting:

Code formatting
~~~~~~~~~~~~~~~

climepi uses `Ruff`_ for standardized code formatting, linting, and ordering of imports.

.. _pull request: https://github.com/idm-oxford/climate-epidemics/compare
.. _create a PR on GitHub: https://help.github.com/en/articles/creating-a-pull-request
.. _reference it: https://help.github.com/en/articles/autolinked-references-and-urls
.. _NumPy style guide: https://numpydoc.readthedocs.io/en/latest/format.html
.. _Pytest: http://doc.pytest.org/en/latest/
.. _Ruff: https://docs.astral.sh/ruff/

Testing with continuous integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The climepi `build workflow`_ runs the test suite automatically via the
`GitHub Actions`_ continuous integration service, once your pull request is submitted.

A pull-request will be considered for merging when you have an all 'green' build. If any
tests are failing, then you will get a red 'X', where you can click through to see the
individual failed tests. This is an example of a green build.

.. note::

   Each time you push to your PR branch, a new run of the tests will be
   triggered on the CI. If they haven't already finished, tests for any older
   commits on the same branch will be automatically cancelled.

.. _build workflow: https://github.com/idm-oxford/climate-epidemics/actions/workflows/run_tests.yml
.. _GitHub Actions: https://docs.github.com/en/free-pro-team@latest/actions

Writing tests
~~~~~~~~~~~~~

All tests should go into the ``tests`` subdirectory of the specific package.
This folder contains many current examples of tests, and we suggest looking to these for
inspiration.

The ``xarray.testing`` module has many special ``assert`` functions that
make it easier to make statements about whether DataArray or Dataset objects are
equivalent. The easiest way to verify that your code is correct is to
explicitly construct the result you expect, then compare the actual result to
the expected correct result::

    def test_constructor_from_0d():
        expected = Dataset({None: ([], 0)})[None]
        actual = DataArray(0)
        assert_identical(expected, actual)