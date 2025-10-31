.. TiRank documentation master file

==================================================
TiRank: Phenotype-driven Niche Prioritization
==================================================

.. image:: _static/TiRank_white.png
   :width: 50%
   :align: center
   :alt: TiRank Logo

|

.. image:: https://img.shields.io/pypi/v/tirank?style=flat-square
   :target: https://pypi.org/project/TiRank/
   :alt: PyPI
.. image:: https://img.shields.io/github/license/LenisLin/TiRank?style=flat-square
   :target: https://github.com/LenisLin/TiRank/blob/main/LICENSE
   :alt: License
.. image:: https://readthedocs.org/projects/tirank/badge/?version=latest&style=flat-square
   :target: https://tirank.readthedocs.io/en/latest/
   :alt: Documentation Status

TiRank is a cutting-edge toolkit designed to integrate and analyze RNA-seq and single-cell RNA-seq (scRNA-seq) data. 
By seamlessly combining spatial transcriptomics or scRNA-seq data with bulk RNA sequencing data, 
TiRank enables researchers to identify phenotype-associated regions or clusters. 
The toolkit supports various analysis modes, including survival analysis (Cox), classification, and regression, providing a comprehensive solution for transcriptomic data analysis.

.. image:: _static/Fig1.png
   :width: 100%
   :align: center
   :alt: TiRank Workflow

----

Features
========

* **🔗 Seamless Data Integration**:
    Combines bulk RNA-seq with single-cell or spatial transcriptomics data.

* **🔄 Versatile Analysis Modes**:
    Includes survival analysis (Cox), classification, and regression.

* **📈 Advanced Visualization**:
    Offers tools for generating UMAP plots, spatial maps, and other visualizations.

* **⚙️ Customizable Hyperparameters**:
    Provides flexibility to fine-tune settings for optimized results.

.. image:: _static/FigS1.png
   :width: 100%
   :align: center
   :alt: TiRank Workflow

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation

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
   :caption: API Reference

   api
   
----

Citing & Support
================

For support, please visit the `TiRank GitHub Issues page <https://github.com/LenisLin/TiRank/issues>`.

License
=======

TiRank is distributed under the MIT License.