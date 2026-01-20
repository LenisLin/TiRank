.. TiRank documentation master file

===========================================================
TiRank prioritizes phenotypic niches in tumor microenvironment for clinical biomarker discovery
===========================================================

.. image:: _static/TiRank_white.png
   :width: 50%
   :align: center
   :alt: TiRank Logo

.. image:: https://img.shields.io/pypi/v/tirank?style=flat-square
  :target: https://pypi.org/project/TiRank/
  :alt: PyPI
.. image:: https://img.shields.io/conda/vn/bioconda/tirank?style=flat-square
  :target: https://anaconda.org/bioconda/tirank
  :alt: Bioconda
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.18275554.svg
  :target: https://doi.org/10.5281/zenodo.18275554
  :alt: Zenodo DOI
.. image:: https://img.shields.io/github/license/LenisLin/TiRank?style=flat-square
  :target: https://github.com/LenisLin/TiRank/blob/main/LICENSE
  :alt: License
.. image:: https://readthedocs.org/projects/tirank/badge/?version=latest&style=flat-square
  :target: https://tirank.readthedocs.io/en/latest/
  :alt: Documentation Status

TiRank is a toolkit designed to integrate and analyze RNA-seq and single-cell RNA-seq (scRNA-seq) data.
By combining spatial transcriptomics or scRNA-seq data with bulk RNA sequencing data, TiRank enables
phenotype-associated region/cluster discovery. The toolkit supports survival analysis (Cox),
classification, and regression.

.. image:: _static/FigS1.png
   :width: 100%
   :align: center
   :alt: TiRank Workflow

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   snakemake_workflow
   model_input

.. toctree::
   :maxdepth: 2
   :caption: CLI Tutorials

   tutorial_st_survival
   tutorial_sc_classification

.. toctree::
   :maxdepth: 2
   :caption: GUI Tutorial

   tutorial_web

.. toctree::
   :maxdepth: 1
   :caption: Features Document

   result_interpretation
   hyperparameters

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api

----

Citing & Support
================
For support, please use the GitHub Issues page: https://github.com/LenisLin/TiRank/issues

License
=======
TiRank is distributed under the MIT License.
