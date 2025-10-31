# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TiRank'
copyright = '2025, LenisLin'
author = 'LenisLin'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'recommonmark',
    'sphinx.ext.autodoc',       # Core: pulls documentation from docstrings
    'sphinx.ext.autosummary',   # Core: creates summary tables
    'sphinx.ext.napoleon',      # Enables Sphinx to read Google-style docstrings
    'sphinx.ext.viewcode',      # Adds a "[source]" link next to your functions
    'nbsphinx'
]

# Tell autosummary to auto-generate stub files
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/TiRank_white.png'

autodoc_mock_imports = [
    "torch", "torchvision", "torchaudio",
    "cupy", "cudnn", "pytorch_lightning",
    "timm", "scanpy", "anndata", "igraph", "leidenalg",
]

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))