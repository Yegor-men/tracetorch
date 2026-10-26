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
    'nbsphinx',
    'nbsphinx_link',
]

# Notebook outputs are recorded during authoring, not during the documentation build.
nbsphinx_execute = 'never'
nbsphinx_allow_errors = False
nbsphinx_codecell_lexer = 'python3'


def configure_linked_notebooks(app):
    # Set this after source registration; nbsphinx-link's default lambda cannot be cached.
    app.config.nbsphinx_custom_formats['.nblink'] = ('nbformat.reads', {'as_version': 4})


def setup(app):
    app.connect('builder-inited', configure_linked_notebooks)
    app.connect('html-page-context', hide_notebook_link_source)


def hide_notebook_link_source(app, pagename, templatename, context, doctree):
    # A .nblink contains routing metadata; the lesson links to its actual notebook.
    if context.get('sourcename', '').endswith('.nblink.txt'):
        context['show_source'] = False

napoleon_use_ivar = True
autodoc_mock_imports = ['torch']

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
html_baseurl = 'https://yegor-men.github.io/tracetorch/'
html_theme_options = {}
if has_rtd_theme:
    html_theme_options = {
        'collapse_navigation': True,
        'navigation_depth': 2,
        'sticky_navigation': True,
    }

import os
import sys
import shutil

# The documentation dependency bundles Pandoc for systems without a separate install.
if shutil.which('pandoc') is None:
    import pypandoc

    os.environ['PATH'] = os.path.dirname(pypandoc.get_pandoc_path()) + os.pathsep + os.environ.get('PATH', '')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
