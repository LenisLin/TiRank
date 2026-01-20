==================
Snakemake Workflow
==================

TiRank provides a Snakemake workflow under the repository's ``workflow/`` directory to support
reproducible end-to-end execution with Snakemake-managed conda environments.

Why we recommend a fixed Snakemake version
------------------------------------------

Snakemake behavior and dependency resolution can change across versions. For consistent and
reproducible execution, we recommend installing a fixed Snakemake release (currently ``7.32.4``).

.. code-block:: bash

   conda create -n tirank_smk -c conda-forge -c bioconda python=3.9 snakemake=7.32.4
   conda activate tirank_smk

Running the workflow with conda environment management
------------------------------------------------------

Run from the TiRank repository root:

.. code-block:: bash

   snakemake --snakefile workflow/Snakefile \
             --configfile workflow/config/config.yaml \
             --use-conda --cores 8

Notes:
- ``--use-conda`` enables Snakemake to automatically create and manage rule environments.
- Configuration is provided via the repository's workflow config file. Adjust it as needed before running.

Repository layout and Snakemake deployment guidance
---------------------------------------------------

Snakemake recommends separating workflow code (``workflow/``) from configuration (often a top-level
``config/`` directory) for distribution and reproducibility.

TiRank is distributed as a software repository that includes a namespaced workflow. To keep workflow
assets clearly scoped, TiRank stores workflow configuration under ``workflow/config/`` while preserving
the key separation of concerns (workflow code vs. configuration vs. environments).

If you prefer a standalone Snakemake workflow repository, you can treat:
- ``workflow/`` as the Snakemake workflow folder, and
- ``workflow/config/`` as the configuration folder for that workflow.
