# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../nussl/'))

import jupytext

# -- Project information -----------------------------------------------------

project = 'nussl'
copyright = '2020, Ethan Manilow, Prem Seetharaman'
author = 'Ethan Manilow, Prem Seetharaman'

# The full version, including alpha/beta/rc tags
release = '1.0.0'

# Mock the dependencies
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'matplotlib.pyplot',
    'vamp',
    'ffmpy',
    'norbert',
    'zarr',
    'numcodecs',
    'librosa',
    'jams',
]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'm2r2', 
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'autodocsumm',
    'nbsphinx',
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', 
    '**.ipynb_checkpoints', '*.ipynb',
    'stage_docs.py',
    'create_and_execute_notebook.py']


nbsphinx_custom_formats = {
    '.py': lambda s: jupytext.reads(s, '.py'),
}
nbsphinx_allow_errors = True
highlight_language = 'none'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_context = {
    'css_files': [
        '_static/theme_overrides.css',  # override wide tables in RTD theme
        ],
     }