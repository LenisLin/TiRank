============
Installation
============

TiRank supports multiple installation methods. It is recommended to create a dedicated conda environment to ensure compatibility.

Method 1: Bioconda Installation (Recommended)
---------------------------------------------
This is the easiest method. You can install the stable version of TiRank directly from the Bioconda channel.

1. Clone the TiRank repository (for access to example scripts):

   .. code-block:: bash

      git clone https://github.com/LenisLin/TiRank.git
      cd TiRank

2. Create a clean python environment:

   .. code-block:: bash

      conda create -n tirank python=3.9
      conda activate tirank

3. Install TiRank:

   .. code-block:: bash

      conda install -c bioconda tirank

.. note::

   The TiRank framework has been tested on ``Ubuntu 22.04`` with ``Python 3.9``, using ``NVIDIA Driver 12.4`` and ``RTX 3090 GPUs``.

Method 2: Docker
----------------

1. **Install Docker**

   Ensure Docker is installed on your system.

2. **Pull the TiRank Docker Image**

   .. code-block:: bash

      docker pull lenislin/tirank_v1:latest

3. **Run the Docker Container**

   .. code-block:: bash

      docker run -p 8050:8050 lenislin/tirank_v1:latest /bin/bash

4. **Verify Container Execution**

   After running the above command, you should be inside the container's terminal. Verify the setup by activating the environment and checking the TiRank version:

   .. code-block:: bash

      conda activate TiRank
      
      python -c "import tirank; print(tirank.__version__)"

5. **Persistent Data Storage** (Optional)

   To mount a local directory to retain data inside the container:

   .. code-block:: bash

      docker run -it -v /path/to/local/data:/container/data lenislin/tirank_v1:latest /bin/bash

6. **Stop and Remove the Docker Container**

   Use the following commands to manage containers:

   .. code-block:: bash

      docker stop <container_id>
      docker rm <container_id>


Method 3: Interactive Web Tool (GUI)
------------------------------------

This method is for running the TiRank web application locally.

1. **Set Up the Web Server**

   Navigate to the ``Web`` directory and create a ``data`` folder:

   .. code-block:: bash

      cd TiRank/Web
      mkdir data

   Download the required datasets and pretrained models into the newly created ``data`` directory:

   - `Example Data and Pretrained Models <https://zenodo.org/records/18275554>`_

2. **Set Up Directory Structure**

   Ensure the following directory structure is maintained within the ``Web`` folder:

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

3. **Run the Web Application**

   Execute the following command from within the ``Web`` directory:

   .. code-block:: bash

      python app.py

4. **Access the Web Interface**

   Open a web browser and navigate to ``http://localhost:8050`` to access the TiRank GUI.
