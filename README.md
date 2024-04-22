
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


### Usage

#### Gene Pair extractor
```
from TiRank.main import *

GenePairSelection(scst_exp_path, bulk_exp_path, bulk_cli_path, datatype, mode, savePath, lognormalize = True, top_var_genes=2000, top_gene_pairs=1000, p_value_threshold=0.05)
```
**Input:**

* ```scst_exp_path```: If your datatype is SC, you should provide ```CSV``` files path which rows represent genes and column represent cells. The first row should be cells id and first column should be gene symbol.
  
  If your datatype is ST, you should provide 10x spaceranger output folder path.
* ```bulk_exp_path```: CSV files which rows represent genes and column represent samples. The first row should be samples id and first column should be gene symbol.
* ```bulk_cli_path```: CSV files with 2 column if your phenotype label is continuous or binary. Or CSV files with 3 column if your phenotype label is survival with second column represent survival time and third column represent survival status. No need to set column names. The first column should be samples id same as the bulk_exp file. Specifically, if your phenotype label is binary you need to convert them to 01 form.
* ```datatype```: ```SC``` represent scRNA-seq data. ```ST``` represent spatial transcriptomics data.
* ```mode```: ```Cox``` represent your phenotype label is survival, ```Classification``` represent your phenotype label is binary, ```Regression``` represent your phenotype label is continuous.
* ```savePath```: The path to save model.
* ```lognormalize```: Whether to perform lognormalize.
* ```top_var_genes, top_gene_pairs, p_value_threshold```: See datails in **Hyperparameter in TiRank** part

#### Model training and prediction
```
TiRank(savePath, datatype, mode, device="cuda")
```
**Input:**

* ```savePath, datatype, mode```:Same as the input in GenePairSelection function
* ```device```: Whether use ```cuda``` or ```cpu``` to train model

#### Result interpretation
After successfully running the above two steps, you can find the file named spot_predict_score.csv in the path savePath/3_Analysis/ , where the Rank_Label column represents the TiRank prediction result.

For ```Cox``` mode, Rank+ cells are associated with worse survival, and Rank- cells are associated with good survival.

For ```Classification``` mode, Rank+ cells are associated with phenotype of the group encoded as 1, and Rank- cells are associated with phenotype of the group encoded as 0.

For ```Regression``` mode, Rank+ cells are associated with high phenotype label scores, and Rank- cells are associated with low phenotype label scores. For example, if input is the IC50 of different cell lines, Rank+ cells associated with drug resistance and Rank- cells associated with drug sensitivity.



### Hyperparameter in TiRank
In TiRank, six key hyperparameters influence the results. The first three are crucial for feature selection in bulk transcriptomics, while the latter three are used for training the multilayer perceptron network. TiRank autonomously chooses suitable combinations for these latter three parameters within a predefined range (Detailed in our article Methods-Tuning of Hyperparameters). However, due to the variability across different bulk transcriptomics datasets, we cannot preset the first three hyperparameters. We give the default setting and clarify the function of each parameter to help users get a optimal results.

* ```top_var_genes```:Considering the high dropout rates in single-cell or spatial transcriptome datasets, the initial feature selection step is to select highly variable features, top_var_genes. Default setting for top_var_genes is 2000. If users find the number of filtered genes is low, you could increase the top_var_genes.

* ```p_value_threshold```:p_value_threshold indicates the significance between each gene and phenotype(Detailed in our article Methods-scRank workflow design-Step1). A lower p_value_threshold indicates a stronger ability of gene to distinguish different phenotypes in bulk transcriptomics. Default setting for p_value_threshold is 0.05. Depending on the number of filtered genes, users may need to adjust this threshold. If users find the number of filtered genes is low, you could increase the p_value_threshold.

* ```top_gene_pairs```:top_gene_pairs is used to selected highly variable gene pairs in bulk transcriptomics that more effectively differentiate phenotypes. Default setting for top_gene_pairs is 2000.

* ```alphas```:alphas determine the weight of different components in total loss computation. (Detailed in our article Methods-scRank workflow design-Step2)

* ```n_epochs```:n_epochs is the number of training epochs in TiRank.

* ```lr```:The learning rate (lr) controls the step size of model training during each iteration of parameter updates. A lower learning rate corresponds to more gradual updates, resulting in slower convergence over each epoch. Conversely, a higher learning rate might cause the model to oscillate around the optimal solution, potentially preventing the attainment of the best results.

### TiRank Web
In order to use TiRank's web pages, you need to go to the Web folder first.
```bash
cd ./Web
```
Everything you do next should be done in this directory.
Next you need to do the following steps:
1. Create the data folder
```bash
mkdir data
```
2. Create an ExampleData folder inside the data floder and download the sample data from https://drive.google.com/drive/folders/1CsvNsDOm3GY8slit9Hl29DdpwnOc29bE
```bash
cd data
```
```bash
mkdir ExampleData
```
```bash
cd ../
```
You need to make sure your file directory structure is as follows:
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
3. You can now run your web application.
```bash
python app.py
```
More tutorials on the Web can be found in the "Tutorials" section of the web page.

