# Usage

## Gene Pair extractor
```
from TiRank.main import *

GenePairSelection(scst_exp_path, bulk_exp_path, bulk_cli_path, datatype, mode, savePath, lognormalize = True, top_var_genes=2000, top_gene_pairs=1000, p_value_threshold=0.05)
```
#### Input:

* ```scst_exp_path```: If your datatype is SC, you should provide ```CSV``` files path which rows represent genes and column represent cells. The first row should be cells id and first column should be gene symbol.
  
  If your datatype is ST, you should provide 10x spaceranger output folder path.
* ```bulk_exp_path```: CSV files which rows represent genes and column represent samples. The first row should be samples id and first column should be gene symbol.
* ```bulk_cli_path```: CSV files with 2 column if your phenotype label is continuous or binary. Or CSV files with 3 column if your phenotype label is survival with first column represent survival time and second column represent survival status. No need to set column names. The first column should be samples id same as the bulk_exp file.
* ```datatype```: ```SC``` represent scRNA-seq data. ```ST``` represent spatial transcriptomics data.
* ```mode```: ```Cox``` represent your phenotype label is survival, ```Classification``` represent your phenotype label is binary, ```Regression``` represent your phenotype label is continuous.
* ```savePath```: The path to save model.
* ```lognormalize```: Whether to perform lognormalize.
* ```top_var_genes, top_gene_pairs, p_value_threshold```: See datails in **Hyperparameter in TiRank** part

## Model training and prediction
```
TiRank(savePath, datatype, mode, device="cuda")
```
Input:

* ```savePath, datatype, mode```:Same as the input in GenePairSelection function
* ```device```: Whether use ```cuda``` or ```cpu``` to train model


# Hyperparameter in TiRank
In TiRank, six key hyperparameters influence the results. The first three are crucial for feature selection in bulk transcriptomics, while the latter three are used for training the multilayer perceptron network. TiRank autonomously chooses suitable combinations for these latter three parameters within a predefined range (Detailed in our article Methods-Tuning of Hyperparameters). However, due to the variability across different bulk transcriptomics datasets, we cannot preset the first three hyperparameters. We give the default setting and clarify the function of each parameter to help users get a optimal results.
## top_var_genes
Considering the high dropout rates in single-cell or spatial transcriptome datasets, the initial feature selection step is to select highly variable features, top_var_genes. Default setting for top_var_genes is 2000. If users find the number of filtered genes is low, you could increase the top_var_genes.
## p_value_threshold
p_value_threshold indicates the significance between each gene and phenotype(Detailed in our article Methods-scRank workflow design-Step1). A lower p_value_threshold indicates a stronger ability of gene to distinguish different phenotypes in bulk transcriptomics. Default setting for p_value_threshold is 0.05. Depending on the number of filtered genes, users may need to adjust this threshold. If users find the number of filtered genes is low, you could increase the p_value_threshold.
## top_gene_pairs
top_gene_pairs is used to selected highly variable gene pairs in bulk transcriptomics that more effectively differentiate phenotypes. Default setting for top_gene_pairs is 2000.
## alphas
alphas determine the weight of different components in total loss computation. (Detailed in our article Methods-scRank workflow design-Step2)
## n_epochs
n_epochs is the number of training epochs in TiRank.
## lr
The learning rate (lr) controls the step size of model training during each iteration of parameter updates. A lower learning rate corresponds to more gradual updates, resulting in slower convergence over each epoch. Conversely, a higher learning rate might cause the model to oscillate around the optimal solution, potentially preventing the attainment of the best results.
