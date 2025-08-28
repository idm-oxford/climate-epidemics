"""Sphinx configuration."""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import inspect
import os
import sys

import sphinx_autosummary_accessors

sys.path.insert(0, os.path.abspath(".."))
import climepi  # noqa: E402

# -- Project information -----------------------------------------------------

project = "climepi"
copyright = "2023, William S Hart"
author = "William S Hart"

# The full version, including alpha/beta/rc tags
release = climepi.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "jupyter_sphinx",
    "nbsphinx",
    "sphinx_autosummary_accessors",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# pydata_sphinx_theme configurations
html_logo = "_static/climepi-logo.svg"
html_title = "climepi documentation"
html_context = {
    "github_user": "idm-oxford",
    "github_repo": "climate-epidemics",
    "github_version": "main",
    "doc_path": "docs",
}
html_theme_options = {
    "github_url": "https://github.com/idm-oxford/climate-epidemics",
    "use_edit_page_button": True,
}
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Extension configuration -------------------------------------------------

# autodoc
autodoc_typehints = "none"

# autosectionlabel
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# autosummary
autosummary_generate = True

# napoleon
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

# nbsphinx
nbsphinx_requirejs_path = ""
nbsphinx_widgets_path = ""


# based on xarray doc/conf.py
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(climepi.__file__))

    if "+" in climepi.__version__:
        return f"https://github.com/idm-oxford/climate-epidemics/blob/main/climepi/{fn}{linespec}"
    else:
        return (
            f"https://github.com/idm-oxford/climate-epidemics/blob/"
            f"v{climepi.__version__}/climepi/{fn}{linespec}"
        )
