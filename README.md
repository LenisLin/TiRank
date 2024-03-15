# Usage

## Load data
```
# 1.1 selecting a path to save the results
savePath = "./web_test"
savePath_1 = os.path.join(savePath, "1_loaddata")
if not os.path.exists(savePath_1):
    os.makedirs(savePath_1, exist_ok=True)

dataPath = "/home/lenislin/Experiment/data/scRankv2/data/ExampleData/SKCM_SC_Res/"

# 1.2 load clinical data
path_to_bulk_cli = os.path.os.path.join(dataPath, "Liu2019_meta.csv")
bulkClinical = load_bulk_clinical(path_to_bulk_cli)
view_dataframe(bulkClinical) 

# 1.3 load bulk expression profile
path_to_bulk_exp = os.path.join(dataPath, "Liu2019_exp.csv")
bulkExp = load_bulk_exp(path_to_bulk_exp)
bulkExp = normalize_data(bulkExp)
view_dataframe(bulkExp)  ## if user try to view the data

# 1.4 Check name
check_bulk(savePath, bulkExp, bulkClinical)

# 1.5 load SC data
path_to_sc_floder = (os.path.join(dataPath,"GSE120575.h5ad"))
scAnndata = load_sc_data(path_to_sc_floder, savePath)
st_exp_df = transfer_exp_profile(scAnndata)
view_dataframe(st_exp_df)  ## if user try to view the data
```
## Preprocessing
```
# 2.1 selecting a path to save the results
savePath = "./web_test"
savePath_1 = os.path.join(savePath, "1_loaddata")
savePath_2 = os.path.join(savePath, "2_preprocessing")

if not os.path.exists(savePath_2):
    os.makedirs(savePath_2, exist_ok=True)

# 2.2 load data
f = open(os.path.join(savePath_1, "anndata.pkl"), "rb")
scAnndata = pickle.load(f)
f.close()

# 2.3 Preprocessing on sc/st data
infer_mode = "Cell"  ## optional parameter

scAnndata = FilteringAnndata(
    scAnndata,
    max_count=35000,
    min_count=5000,
    MT_propor=10,
    min_cell=10,
    imgPath=savePath_2,
)  ## optional parameters: max_count, min_count, MT_propor, min_cell
scAnndata = Normalization(scAnndata)
scAnndata = Logtransformation(scAnndata)
scAnndata = Clustering(scAnndata, infer_mode=infer_mode, savePath=savePath)
compute_similarity(savePath=savePath, ann_data=scAnndata)

with open(os.path.join(savePath_2, "scAnndata.pkl"), "wb") as f:
    pickle.dump(scAnndata, f)
f.close()

# 2.4 clinical column selection and bulk data split
mode = "Bionomial"

bulkClinical = view_clinical_variables(savePath)
choose_clinical_variable(
    savePath,
    bulkClinical=bulkClinical,
    mode=mode,
    var_1="Response"
)

# data split
generate_val(
    savePath=savePath, validation_proportion=0.15, mode=mode
)  ## optinal parameter: validation_proportion

# sampling
perform_sampling_on_RNAseq(savePath=savePath,mode="SMOTE", threshold=0.5)

# 2.5 Genepair Transformation
GPextractor = GenePairExtractor(
    savePath=savePath,
    analysis_mode=mode,
    top_var_genes=2000,
    top_gene_pairs=1000,
    p_value_threshold=0.05,
    padj_value_threshold=None,
    max_cutoff=0.8,
    min_cutoff=-0.8,
)  ## optinal parameter: top_var_genes, top_gene_pairs, padj_value_threshold, padj_value_threshold

GPextractor.load_data()
GPextractor.run_extraction()
GPextractor.save_data()

```
## Model Train and Prediction
```
savePath = "./web_test"
savePath_1 = os.path.join(savePath, "1_loaddata")
savePath_2 = os.path.join(savePath, "2_preprocessing")
savePath_3 = os.path.join(savePath, "3_Analysis")

if not os.path.exists(savePath_3):
    os.makedirs(savePath_3, exist_ok=True)


# 3.1.1 Dataloader
mode = "Bionomial"
infer_mode = "Cell"
device = "cuda" if torch.cuda.is_available() else "cpu"

PackData(savePath, mode=mode, infer_mode=infer_mode, batch_size=1024)

# 3.1.2 Training
encoder_type = "MLP"  ## Optional parameter

# Model parameter
initial_model_para(
    savePath=savePath,
    nhead=2,
    nhid1=96,
    nhid2=8,
    n_output=32,
    nlayers=3,
    n_pred=2,
    dropout=0.5,
    mode=mode,
    encoder_type=encoder_type,
    infer_mode=infer_mode,
)

tune_hyperparameters(
    ## Parameters Path
    savePath=savePath,
    device=device,
    n_trials=10,
)  ## optional parameters: n_trials

# 3.1.3 Inference
Predict(savePath=savePath, mode=mode, do_reject=True, tolerance=0.05, reject_mode="GMM")

# 3.1.4 Visualization
plot_score_distribution(savePath)  # Display the prob score distribution
```

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
