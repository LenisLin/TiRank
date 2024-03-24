# main
import warnings
warnings.filterwarnings("ignore")

import torch
import pickle
import os

import sys
sys.path.append("/home/lenislin/Experiment/projects/TiRankv2/github/TiRank")

from TiRank.Model import setup_seed, initial_model_para
from TiRank.LoadData import *
from TiRank.SCSTpreprocess import *
from TiRank.Imageprocessing import GetPathoClass
from TiRank.GPextractor import GenePairExtractor
from TiRank.Dataloader import view_clinical_variables, choose_clinical_variable, generate_val, PackData
from TiRank.TrainPre import tune_hyperparameters, Predict
from TiRank.Visualization import plot_score_distribution, DEG_analysis, DEG_volcano, Pathway_Enrichment
from TiRank.Visualization import plot_score_umap, plot_label_distribution_among_conditions

setup_seed(619)

## 1. Load data
# 1.1 selecting a path to save the results
savePath = "./ST_Survival_CRC"
savePath_1 = os.path.join(savePath, "1_loaddata")
if not os.path.exists(savePath_1):
    os.makedirs(savePath_1, exist_ok=True)

# 1.2 load clinical data
dataPath = "./ExampleData/ST_Prog_CRC/"
path_to_bulk_cli = os.path.join(dataPath, "GSE39582_clinical_os.csv")
bulkClinical = load_bulk_clinical(path_to_bulk_cli)
view_dataframe(bulkClinical)  ## if user try to view the data

# 1.3 load bulk expression profile
path_to_bulk_exp = os.path.join(dataPath, "GSE39582_exp_os.csv")
bulkExp = load_bulk_exp(path_to_bulk_exp)
view_dataframe(bulkExp)  ## if user try to view the data

# 1.4 Check name
check_bulk(savePath, bulkExp, bulkClinical)

# 1.5 load ST data
path_to_st_floder = (os.path.join(dataPath,"SN048_A121573_Rep1"))
scAnndata = load_st_data(path_to_st_floder, savePath)
st_exp_df = transfer_exp_profile(scAnndata)
view_dataframe(st_exp_df)  ## if user try to view the data

## 2. Preprocessing
# 2.1 selecting a path to save the results
savePath = "./ST_Survival_CRC"
savePath_1 = os.path.join(savePath, "1_loaddata")
savePath_2 = os.path.join(savePath, "2_preprocessing")

if not os.path.exists(savePath_2):
    os.makedirs(savePath_2, exist_ok=True)

# 2.2 load data
f = open(os.path.join(savePath_1, "anndata.pkl"), "rb")
scAnndata = pickle.load(f)
f.close()

# 2.3 Preprocessing on sc/st data
infer_mode = "Spot"  ## optional parameter

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
compute_similarity(savePath=savePath, ann_data=scAnndata, calculate_distance=False)

pretrain_path = "/home/lenislin/Experiment/projects/TiRankv2/TiRank/pretrainModel/ctranspath.pth"  ## put this file in the package

n_patho_cluster = 6  ## optional variable

scAnndata = GetPathoClass(
    adata=scAnndata,
    pretrain_path=pretrain_path,
    n_clusters=n_patho_cluster,
    image_save_path=os.path.join(savePath_2, "patho_label.png"),
)  ## advanced parameters: n_components, n_clusters

with open(os.path.join(savePath_2, "scAnndata.pkl"), "wb") as f:
    pickle.dump(scAnndata, f)
f.close()

# 2.4 clinical column selection and bulk data split
mode = "Cox"

bulkClinical = view_clinical_variables(savePath)
choose_clinical_variable(
    savePath,
    bulkClinical=bulkClinical,
    mode=mode,
    var_1="Overall_time",
    var_2="Overall_event",
)

# data split
generate_val(
    savePath=savePath, validation_proportion=0.15, mode=mode
)  ## optinal parameter: validation_proportion

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

## 3. Analysis
# 3.1 TiRank
savePath = "./ST_Survival_CRC"
savePath_1 = os.path.join(savePath, "1_loaddata")
savePath_2 = os.path.join(savePath, "2_preprocessing")
savePath_3 = os.path.join(savePath, "3_Analysis")

if not os.path.exists(savePath_3):
    os.makedirs(savePath_3, exist_ok=True)

# 3.1.1 Dataloader
mode = "Cox"  ## how to let this variable continuous in the analysis ?
infer_mode = "Spot"  ## optional parameter
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
    n_pred=1,
    n_patho = 6,
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
do_p_adjust = True

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

# ## Sub-region visualization
# savePath_selecting = os.path.join(savePath_3, "selecting")
# if not os.path.exists(savePath_selecting):
#     os.makedirs(savePath_selecting, exist_ok=True)

# upload_metafile(savePath,"/home/lenislin/Experiment/data/scRankv2/data/ExampleData/CRC_ST_Prog/SN048_A121573_Rep1/meta.csv")


