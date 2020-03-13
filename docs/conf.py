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
sys.path.insert(0, os.path.abspath('../nussl/'))


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

with open('../requirements.txt', 'r') as f:
    MOCK_MODULES = f.readlines()
    MOCK_MODULES = [_mock.rstrip() for _mock in MOCK_MODULES]
MOCK_MODULES += [
    'matplotlib.pyplot', 
    'torch.utils', 
    'torch.utils.data', 
    'torch.nn',
    'torch.utils.checkpoint'
]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'm2r', 
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'autoapi.extension',
]

autoapi_type = 'python'
autoapi_dirs = ['../nussl']
exclude_folders = [
    'composite', 
    'factorization', 
    'primitive', 
    'spatial'
]
autoapi_ignore = [f'*/separation/{x}/*' for x in exclude_folders]
autoapi_add_toctree_entry = False
autoapi_template_dir = '_templates/'
autoapi_options = [
    'members', 'undoc-members', 'private-members', 
    'show-inheritance', 'special-members']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']