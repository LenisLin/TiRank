# Example for integrate single-cell RNA-seq data of melanoma and response information
import warnings
warnings.filterwarnings("ignore")

import torch
import pickle
import os

from tirank.Model import setup_seed, initial_model_para
from tirank.LoadData import *
from tirank.SCSTpreprocess import *
from tirank.GPextractor import GenePairExtractor
from tirank.Dataloader import generate_val, PackData
from tirank.TrainPre import tune_hyperparameters, Predict
from tirank.Visualization import plot_score_distribution, DEG_analysis, DEG_volcano, Pathway_Enrichment
from tirank.Visualization import plot_score_umap, plot_label_distribution_among_conditions

setup_seed(619)

## 1. Load data
# 1.1 selecting a path to save the results
savePath = "./Example/SKCM_SC_Res_experiment"
savePath_1 = os.path.join(savePath, "1_loaddata")
if not os.path.exists(savePath_1):
    os.makedirs(savePath_1, exist_ok=True)

dataPath = "./data/ExampleData/SKCM_SC_Res"

# 1.2 load clinical data
path_to_bulk_cli = os.path.join(dataPath, "Liu2019_meta.csv")
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

## 2. Preprocessing
# 2.1 selecting a path to save the results
savePath = "./SC_Respones_SKCM"
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
    min_count=3,
    MT_propor=10,
    min_cell=1,
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
mode = "Classification"

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
    max_cutoff=0.8,
    min_cutoff=-0.8,
)  ## optinal parameter: top_var_genes, top_gene_pairs, padj_value_threshold, padj_value_threshold

GPextractor.load_data()
GPextractor.run_extraction()
GPextractor.save_data()

## 3. Analysis
# 3.1 tirank
savePath = "./SC_Respones_SKCM"
savePath_1 = os.path.join(savePath, "1_loaddata")
savePath_2 = os.path.join(savePath, "2_preprocessing")
savePath_3 = os.path.join(savePath, "3_Analysis")

if not os.path.exists(savePath_3):
    os.makedirs(savePath_3, exist_ok=True)


# 3.1.1 Dataloader
mode = "Classification"
infer_mode = "SC"
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
    n_trials=5,
)  ## optional parameters: n_trials

# 3.1.3 Inference
Predict(savePath=savePath, mode=mode, do_reject=True, tolerance=0.05, reject_mode="GMM")

# 3.1.4 Visualization
plot_score_distribution(savePath)  # Display the prob score distribution
plot_score_umap(savePath,infer_mode)
plot_label_distribution_among_conditions(savePath,group="leiden_clusters")

# 3.2 DEGs and Pathway enrichment
fc_threshold = 2
Pvalue_threshold = 0.05
do_p_adjust = False

DEG_analysis(
    savePath,
    fc_threshold=fc_threshold,
    Pvalue_threshold=Pvalue_threshold,
    do_p_adjust=do_p_adjust,
)
DEG_volcano(
    savePath,
    fc_threshold=fc_threshold,
    Pvalue_threshold=Pvalue_threshold,
    do_p_adjust=do_p_adjust,
)

# database = ["KEGG_2021_Human","MSigDB_Hallmark_2020","GO_Biological_Process_2023"]
# refer to https://maayanlab.cloud/Enrichr/#libraries

Pathway_Enrichment(savePath, database=["GO_Biological_Process_2023"])

