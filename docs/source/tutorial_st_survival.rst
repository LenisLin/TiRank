.. _tutorial_st_survival:

====================================================================
Tutorial 1: Spatial Transcriptomics (ST) + Cox Survival Analysis
====================================================================

This tutorial demonstrates a complete workflow for integrating Spatial Transcriptomics (ST) data with bulk RNA-seq data to perform a "Cox" survival analysis.

The example uses Colorectal Cancer (CRC) data to identify spatial spots associated with patient prognosis.

Prerequisites: Download the Model
---------------------------------
Before running this script, you must download the pre-trained feature extraction model.

1. **Download:** Get ``ctranspath.pth`` from `Zenodo <https://zenodo.org/records/18275554>`_.
2. **Setup:** Create a folder named ``data/pretrainModel/`` in your project root and place the file there.

.. code-block:: text

    TiRank/
    └── data/
        └── pretrainModel/
            └── ctranspath.pth

Full Python Script
------------------
This file is available in your repository at ``Example/ST-Cox-CRC.py``.

.. literalinclude:: ../../../Example/ST-Cox-CRC.py
   :language: python
   :linenos: