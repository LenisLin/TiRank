
# TiRank
TiRank is a comprehensive tool for integrating and analyzing RNA-seq and scRNA-seq data. This document provides detailed instructions on how to install TiRank in your environment.

### Installation Instructions for TiRank

TiRank can be installed through multiple methods. We recommend creating a new conda environment specifically for TiRank for optimal compatibility and isolation from other Python packages.

#### Method 1: Online pip Installation
1. Set up a new conda environment:
   ```bash
   conda create -n TiRank python=3.9.7 -y
   conda activate TiRank
   ```
2. Navigate to the TiRank directory:
   ```bash
   cd ./TiRank
   ```
3. Install TiRank via pip:
   ```bash
   pip install TiRank
   ```
4. Additionally, install the `timm==0.5.4` package from [TransPath](https://github.com/Xiyue-Wang/TransPath). This is a required dependency for TiRank. Follow these steps:
   - Download the modified `timm==0.5.4` package from [this link](https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view?pli=1).
   - Install it using pip with the path to the downloaded package:
     ```bash
     pip install /YOUR/LOCATION/TO/PACKAGE/timm-0.5.4.tar # Replace with your actual path
     ```
5. Download the pre-trained [CTransPath](https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view) model weights.

#### Method 2: Local conda Installation
1. Clone the TiRank repository:
   ```bash
   git clone git@github.com:LenisLin/TiRank.git
   ```
2. Modify the `TiRank.yml` environment file. Replace the "prefix" at the bottom of this file with your path to the conda environment files.
3. Create the environment from the `TiRank.yml` file:
   ```bash
   conda env create -f TiRank.yml
   ```

#### Method 3: Docker Installation (Highly Recommended)
_(Instructions to be provided)_

---

Please choose the installation method that best suits your setup. If you encounter any issues, feel free to open an issue on the [TiRank GitHub page](https://github.com/LenisLin/TiRank).
