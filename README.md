# TiRank

<img src="./docs/source/_static/TiRank_white.png" alt="TiRank Logo" width=50% />

| | |
| :--- | :--- |
| **Full Documentation** | 📖 [**https://tirank.readthedocs.io**](https://tirank.readthedocs.io) |
| **PyPI** | [![PyPI](https://img.shields.io/pypi/v/tirank?style=flat-square)](https://pypi.org/project/TiRank/) |
| **License** | [![License](https://img.shields.io/github/license/LenisLin/TiRank?style=flat-square)](https://github.com/LenisLin/TiRank/blob/main/LICENSE) |
| **Build Status** | [![Documentation Status](https://readthedocs.org/projects/tirank/badge/?version=latest&style=flat-square)](https://tirank.readthedocs.io/en/latest/) |

TiRank is a cutting-edge toolkit designed to integrate and analyze RNA-seq and single-cell RNA-seq (scRNA-seq) data. By seamlessly combining spatial transcriptomics or scRNA-seq data with bulk RNA sequencing data, TiRank enables researchers to identify phenotype-associated regions or clusters. The toolkit supports various analysis modes, including survival analysis (Cox), classification, and regression, providing a comprehensive solution for transcriptomic data analysis.

![TiRank Workflow](./docs/source/_static/Fig1.png)

---

## TiRank Features

- **🔗 Seamless Data Integration**:
    Combines bulk RNA-seq with single-cell or spatial transcriptomics data.

- **🔄 Versatile Analysis Modes**:
    Includes survival analysis (Cox), classification, and regression.

- **📈 Advanced Visualization**:
    Offers tools for generating UMAP plots, spatial maps, and other visualizations.

- **⚙️ Customizable Hyperparameters**:
    Provides flexibility to fine-tune settings for optimized results.

---

## Table of Contents

- 📖 [Full Documentation](#full-documentation)
- 🛠️ [Installation](#installation)
- 📚 [Quickstart Tutorials](#quickstart-tutorials)
- 🧑‍💻 [Support](#support)
- 📜 [License](#license)

---

## Full Documentation

For detailed guides on using TiRank, please see our full documentation site:

### ➡️ [**https://tirank.readthedocs.io**](https://tirank.readthedocs.io)

This includes:

  * **[Installation](https://tirank.readthedocs.io/en/latest/model_input.html)**
  * **[CLI Tutorial](https://tirank.readthedocs.io/en/latest/result_interpretation.html)**
  * **[Web Tutorial](https://tirank.readthedocs.io/en/latest/hyperparameters.html)**
  * **[Features Document](https://tirank.readthedocs.io/en/latest/api.html)**
  * **[API Reference](https://tirank.readthedocs.io/en/latest/api.html)**

---

## Installation

TiRank supports multiple installation methods. It is recommended to create a dedicated conda environment to ensure compatibility.

### Method 1: Environment Setup and Installation

You can set up the full TiRank environment directly with the following commands:

```bash
cd TiRank
conda env create -f ./installation/environment.yml
````

Then, activate the environment and run TiRank:

```bash
conda activate Tirank
```

-----

`Note`: The TiRank framework has been tested on `Ubuntu 22.04` with `Python 3.9`, using `NVIDIA Driver 12.4` and `RTX 3090 GPUs`.

### Method 2: Docker

1.  **Install Docker**:

      - Ensure Docker is installed on your system.

2.  **Pull the TiRank Docker Image**:

    ```bash
    docker pull lenislin/tirank_v1:latest
    ```

3.  **Run the Docker Container**:

    ```bash
    docker run -p 8050:8050 lenislin/tirank_v1:latest /bin/bash
    ```

### Method 3: Interactive Web Tool

For instructions on running the TiRank-GUI, please see the **[GUI Tutorial](https://tirank.readthedocs.io/en/latest/tutorial_web.html)** in our documentation.

-----

## Quickstart Tutorials

### Examples:

1.  **Integrating scRNA-seq Data for Melanoma Response Analysis**

      - [Example Script](./docs/Example/SC-Response-SKCM.py)
      - ➡️ **See the full [scRNA-seq Tutorial](https://tirank.readthedocs.io/en/latest/tutorial_sc_classification.html)**

2.  **Combining Spatial Transcriptomics and Bulk Data for Phenotype Detection**

      - [Example Script](./docs/Example/ST-Cox-CRC.py)
      - ➡️ **See the full [ST-Cox Tutorial](https://tirank.readthedocs.io/en/latest/tutorial_st_survival.html)**

3.  **Comprehensive Downstream Analysis**

      - [Example Workflow](./docs/Example/Downstream/CRC)

-----


## Support

For assistance, please visit the [TiRank GitHub Issues page](https://github.com/LenisLin/TiRank/issues).

-----

## License

TiRank is distributed under the MIT License.