=========================================================
Tutorial 1: Spatial Transcriptomics (ST) + Cox Survival Analysis
=========================================================

This tutorial demonstrates TiRank survival analysis (Cox mode) using spatial transcriptomics as inference data
and bulk survival labels as supervision.

Example resources
-----------------

Example datasets are hosted on Zenodo:

* https://zenodo.org/records/18275554

Recommended placement::

   TiRank/data/ExampleData/CRC_ST_Prog/
   ├── GSE39582_clinical_os.csv
   ├── GSE39582_exp_os.csv
   └── SN048_A121573_Rep1/        (ST folder)

Run the example script (Python)
-------------------------------

From the repository root::

   python Example/ST-Cox-CRC.py

Notes
-----

* If your local data paths differ, edit the ``dataPath`` / ``savePath`` variables at the top of the example script.
* For a fully automated run with environment management, see :doc:`snakemake_workflow`.

Example script (for reference)
------------------------------

.. literalinclude:: ../../Example/ST-Cox-CRC.py
   :language: python
   :linenos:
