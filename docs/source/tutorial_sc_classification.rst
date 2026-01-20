=================================================
Tutorial 2: scRNA-seq + Classification Analysis
=================================================

This tutorial demonstrates a typical TiRank classification workflow using scRNA-seq as inference data and
bulk phenotype labels for supervision.

Example resources
-----------------

Example datasets are hosted on Zenodo:

* https://zenodo.org/records/18275554

Recommended placement::

   TiRank/data/ExampleData/SKCM_SC_Res/
   ├── GSE120575.h5ad
   ├── Liu2019_exp.csv
   └── Liu2019_meta.csv

Run the example script (Python)
-------------------------------

From the repository root::

   python Example/SC-Response-SKCM.py

Notes
-----

* If your local data paths differ, edit the ``dataPath`` / ``savePath`` variables at the top of the example script.
* For a fully automated run with environment management, see :doc:`snakemake_workflow`.

Example script (for reference)
------------------------------

.. literalinclude:: ../../Example/SC-Response-SKCM.py
   :language: python
   :linenos:
