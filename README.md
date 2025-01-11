
# TiRank

TiRank is a comprehensive tool for integrating and analyzing RNA-seq and scRNA-seq data. It enables researchers to identify phenotype-associated spots by integrating spatial transcriptomics or single-cell RNA sequencing data with bulk RNA sequencing data. TiRank supports various analysis modes, including survival analysis (Cox), classification, and regression.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Method 1: pip Installation](#method-1-pip-installation)
  - [Method 2: Docker](#method-2-docker)
  - [Method 3: Interactive Web Tool](#method-3-interactive-web-tool)
- [Result Interpretation](#result-interpretation)
- [Hyperparameters](#hyperparameters)
  - [Feature Selection Hyperparameters](#feature-selection-hyperparameters)
  - [Model Training Hyperparameters](#model-training-hyperparameters)
- [Support](#support)
- [License](#license)

---

## Quickstart tutorials

- [1. Example for integrate single-cell RNA-seq data of melanoma and response information.](Example/SC-Response-SKCM.py)
- [2. Example for integrate spatial transcriptomics (ST) and bulk transcriptomics data to identify phenotype-associated spots and determine significant clusters.](Example/ST-Cox-CRC.py)
- [3. A comprehensive downstream analysis of the spatial transcriptome: We present a series of downstream analyses based on TiRank results, demonstrating that TiRank can facilitate the identification and characterization of spatial structures associated with specific clinical phenotypes in spatial transcriptome studies.](Downstream/CRC)

---

## Model input:

- Spatial transcriptome data or single-cell data that we want to characterize.
- Bulk transcriptomics data: Expression profiles and pre-processed clinical information files. The format of the pre-processed clinical information file should align with our sample data.

---
  
## Features

- **Integration of Bulk and Single-cell Data**: Seamlessly integrates bulk RNA-seq data with single-cell or spatial transcriptomics data.
- **Multiple Analysis Modes**: Supports Cox survival analysis, classification, and regression modes.
- **Visualization Tools**: Provides functions for visualizing results, including UMAP plots and spatial maps.
- **Customizable Hyperparameters**: Offers flexibility in tuning hyperparameters to optimize results.

---

## Installation

TiRank can be installed using one of the following methods. We recommend creating a new conda environment for TiRank to ensure compatibility and isolation from other Python packages.

### Prerequisites

- **Anaconda or Miniconda**: For managing Python environments.
- **Python 3.9**: TiRank requires Python version 3.9.

### Method 1: pip Installation

1. **Set up a new conda environment**:

   ```bash
   conda create -n TiRank python=3.9 -y
   conda activate TiRank
   ```
2. **Clone the TiRank repository from GitHub**:
   ```bash
   git clone git@github.com:LenisLin/TiRank.git
   ```
3. **Install TiRank via pip**:
   ```bash
   pip install TiRank
   ```
4. **Install required dependencies**:

TiRank depends on the `timm==0.5.4` package from [TransPath](https://github.com/Xiyue-Wang/TransPath). Follow these steps to install it:
   - Install the package:
     ```bash
     pip install ./TiRank/timm-0.5.4.tar # Replace with your actual path
     ```
   - [Reference Link](https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view?pli=1).

5. **Prepare Example Data**:

   - Download the example data from [Google Drive](https://drive.google.com/drive/folders/1CsvNsDOm3GY8slit9Hl29DdpwnOc29bE)

6. **(Optional, for Spatial Transcriptomics)**: Download the pre-trained [CTransPath](https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view) model weights.

#### Method 2: Docker
_(Instructions to be provided)_

#### Method 3: Interactive Web tool
1. **Install TiRank**: Follow the installation steps described in *Method 1*

2. **Activate the Web Server**:
   - Navigate to the Web directory:
   ```bash
   cd TiRank/Web
   ```
   - Set up data directories:
      - Create a `data` directory:
   ```bash
   mkdir data
   ```

      - Inside the `data` directory, create an `ExampleData` folder and download the sample data:
      ```bash
      cd data
      mkdir ExampleData
      ```

      Download the sample data from [Google Drive](https://drive.google.com/drive/folders/1CsvNsDOm3GY8slit9Hl29DdpwnOc29bE) into the `ExampleData` directory.

      - Return to the `Web` directory:
      ```bash
      cd ../
      ```

   - Verify the directory structure:
   ```bash
   Web/
   ├── assets/
   ├── components/
   ├── img/
   ├── layout/
   ├── data/
   │ ├── ExampleData
   │ │ ├── CRC_ST_Prog/
   │ │ └── SKCM_SC_Res/
   ├── tiRankWeb/
   └── app.py
   ```

   - Run the web application:
   ```bash
   python app.py
   ```

**Note**: If you encounter any issues with image loading, ensure that you are running the program from the `Web` directory.

For more tutorials on using the web interface, please refer to the "Tutorials" section within the web application.

Please choose the installation method that best suits your setup. If you encounter any issues, feel free to open an issue on the [TiRank GitHub Issues page](https://github.com/LenisLin/TiRank/issues).

---

## Result Interpretation

After running TiRank, you can find the results in the `savePath/3_Analysis/` directory. The key output file is `spot_predict_score.csv`, where the `Rank_Label` column represents the TiRank prediction results.

- **For `Cox` mode**:
  - `Rank+` spots are associated with **worse survival**.
  - `Rank-` spots are associated with **better survival**.

- **For `Classification` mode**:
  - `Rank+` spots are associated with the phenotype of the group encoded as `1`.
  - `Rank-` spots are associated with the phenotype of the group encoded as `0`.

- **For `Regression` mode**:
  - `Rank+` spots are associated with **high phenotype label scores**.
  - `Rank-` spots are associated with **low phenotype label scores**.
  - **For example**, if the input is the IC50 values of different cell lines, `Rank+` spots are associated with **drug resistance**, and `Rank-` spots are associated with **drug sensitivity**.

---

## Hyperparameters

TiRank provides several hyperparameters that can be adjusted to optimize the analysis. The first three hyperparameters are crucial for feature selection in bulk transcriptomics, while the latter three are used for training the multilayer perceptron network. TiRank automatically selects suitable combinations for the training hyperparameters within a predefined range.

### Feature Selection Hyperparameters

- **`top_var_genes`**:

  - **Description**: The number of top variable genes to select from the bulk RNA-seq data.
  - **Default**: `2000`
  - **Recommendation**: If you find that the number of filtered genes is low, consider increasing `top_var_genes`.

- **`p_value_threshold`**:

  - **Description**: The p-value threshold for selecting genes significantly associated with the phenotype.
  - **Default**: `0.05`
  - **Recommendation**: If too few genes are selected, consider increasing `p_value_threshold`.

- **`top_gene_pairs`**:

  - **Description**: The number of top gene pairs to select based on variability.
  - **Default**: `2000`

### Model Training Hyperparameters

- **`alphas`**:

  - **Description**: Weights of different components in the total loss computation.
  - **Details**: Adjusts the influence of each loss component during training.

- **`n_epochs`**:

  - **Description**: The number of training epochs.
  - **Recommendation**: Increase if the model has not converged.

- **`lr`** (Learning Rate):

  - **Description**: Controls the step size during parameter updates.
  - **Recommendation**: A lower `lr` leads to slower but more stable convergence. A higher `lr` may speed up convergence but can cause the model to overshoot optimal solutions.

---
### Reference
