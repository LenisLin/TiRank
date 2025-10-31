============
Installation
============

TiRank supports multiple installation methods. It is recommended to create a dedicated conda environment to ensure compatibility.

Method 1: Environment Setup and Installation
------------------------------------------
This is the recommended method. You can set up the full TiRank environment directly using the provided environment file.

1. Clone the TiRank repository:

   .. code-block:: bash

      git clone https://github.com/LenisLin/TiRank.git
      cd TiRank

2. Create the conda environment from the file:

   .. code-block:: bash

      conda env create -f environment.yml

3. Activate the new environment:

   .. code-block:: bash

      conda activate Tirank

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
      
      python -c "import TiRank; print(TiRank.__version__)"

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

   - `Pretrained Models <https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view>`_
   - `Example Data <https://drive.google.com/drive/folders/1CsvNsDOm3GY8slit9Hl29DdpwnOc29bE>`_

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