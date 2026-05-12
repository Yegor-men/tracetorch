# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'traceTorch'
copyright = '2026, Yegor Menovchshikov'
author = 'Yegor Menovchshikov'
release = 'v0.19.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

napoleon_use_ivar = True
autodoc_mock_imports = [
    'torch',
    'numpy',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.colors',
    'scipy',
    'scipy.stats',
]

import importlib.util

if importlib.util.find_spec('myst_parser') is not None:
    extensions.append('myst_parser')

if importlib.util.find_spec('sphinxcontrib.googleanalytics') is not None:
    extensions.append('sphinxcontrib.googleanalytics')
    googleanalytics_id = "G-4B6TFLZ5PC"
    googleanalytics_enabled = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

has_rtd_theme = importlib.util.find_spec('sphinx_rtd_theme') is not None
html_theme = 'sphinx_rtd_theme' if has_rtd_theme else 'alabaster'
html_static_path = ['_static']
html_theme_options = {}
if has_rtd_theme:
    html_theme_options = {
        'collapse_navigation': True,
        'navigation_depth': 2,
        'sticky_navigation': True,
    }

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
