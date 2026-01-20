============
Installation
============

TiRank supports multiple installation methods. We recommend using a dedicated conda environment.

Method 1: Bioconda Installation (Recommended)
---------------------------------------------

This is the simplest method. It installs the released TiRank package from Bioconda.

1. Clone the TiRank repository (recommended to access example scripts and the Snakemake workflow):

.. code-block:: bash

   git clone https://github.com/LenisLin/TiRank.git
   cd TiRank

2. Create a clean Python environment:

.. code-block:: bash

   conda create -n tirank python=3.9
   conda activate tirank

3. Install TiRank (and required graph dependencies):

.. code-block:: bash

   conda install -c bioconda -c conda-forge tirank leidenalg python-igraph

.. note::
   TiRank has been tested on Ubuntu 22.04 with Python 3.9.

Method 2: Docker
----------------

1. Install Docker.

2. Pull the TiRank Docker image:

.. code-block:: bash

   docker pull lenislin/tirank_v1:latest

3. Run the Docker container:

.. code-block:: bash

   docker run -p 8050:8050 lenislin/tirank_v1:latest /bin/bash

4. Verify TiRank inside the container:

.. code-block:: bash

   conda activate TiRank
   python -c "import tirank; print(tirank.__version__)"

5. (Optional) Mount a local directory for persistent storage:

.. code-block:: bash

   docker run -it -v /path/to/local/data:/container/data lenislin/tirank_v1:latest /bin/bash

Method 3: Interactive Web Tool (GUI)
------------------------------------

This method runs the TiRank web application locally.

1. Set up the web server directory:

.. code-block:: bash

   cd TiRank/Web
   mkdir -p data

2. Download and extract the archived example datasets and pretrained model files from Zenodo:

   - Zenodo record (ExampleData.zip, ctranspath.pth, GUI tutorial video):
     https://doi.org/10.5281/zenodo.18275554

3. Ensure the following directory structure under ``Web/``:

.. code-block:: text

   Web/
   ├── assets/
   ├── components/
   ├── img/
   ├── layout/
   ├── data/
   │   ├── pretrainModel/
   │   │   └── ctranspath.pth
   │   ├── ExampleData/
   │   │   ├── CRC_ST_Prog/
   │   │   └── SKCM_SC_Res/
   ├── tiRankWeb/
   └── app.py

4. Run the web application:

.. code-block:: bash

   python app.py

5. Open a browser at: ``http://localhost:8050``

Method 4: Snakemake Workflow (Reproducible Execution)
-----------------------------------------------------

For end-to-end workflow execution with automated environment provisioning, we recommend running the
provided Snakemake workflow with integrated conda environment management.

1. Install a fixed Snakemake version:

.. code-block:: bash

   conda create -n tirank_smk -c conda-forge -c bioconda python=3.9 snakemake=7.32.4
   conda activate tirank_smk

2. From the TiRank repository root, run:

.. code-block:: bash

   snakemake --snakefile workflow/Snakefile \
             --configfile workflow/config/config.yaml \
             --use-conda --cores 8

For details (including recommended folder layout and configuration), see:
:doc:`snakemake_workflow`
