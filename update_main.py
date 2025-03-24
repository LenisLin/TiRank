# For ablation study
## Objectives: (1) REO module (2) Bulk loss (CrossEntropy) (3) SC loss (Cosine loss)
import warnings
warnings.filterwarnings("ignore")

import os
import torch
import pickle
import pandas as pd
from anndata import AnnData
import logging

from TiRank.Model import setup_seed, initial_model_para
from TiRank.LoadData import *
from TiRank.SCSTpreprocess import *
from TiRank.GPextractor import GenePairExtractor
from TiRank.Dataloader import generate_val, PackData
from TiRank.TrainPre import tune_hyperparameters

from update_func import *

# Set random seed for reproducibility
setup_seed(619)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Set paths for loading data and saving results
savePath_base_ = "./"
dataPath = "/mnt/data/songjinsheng/a_graduate/dataset/cellline_benchmark/data"

# Load dataset correspondence
dataPairfile = pd.read_csv("./cellline_data.csv")

# Basic settings
infer_mode = "SC"  # Optional parameter
mode = "Classification"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Number of repeats
repeatTimes = 2
idx_ = 6

for i in range(repeatTimes):
    ## Set save path
    savePath_base = os.path.join(savePath_base_,"intera_"+str(i))
    if not os.path.exists(savePath_base):
        os.makedirs(savePath_base,exist_ok=True)

    ## 0. Path to save each data
    datasetName_ = dataPairfile.iloc[idx_,2] ## dataset name
    savePath = os.path.join(savePath_base,str(datasetName_))
    if not os.path.exists(savePath_base):
        os.makedirs(savePath_base,exist_ok=True)

    print(f"Processing dataset {datasetName_}")

    ## 1.1 Set path to single cell and bulk files
    scRNApath_ = os.path.join(dataPath,"single",str(dataPairfile.iloc[idx_,1]))
    bulkRNApath_ = os.path.join(dataPath,"bulk",str(dataPairfile.iloc[idx_,0]))

    scRNA_exp = scRNApath_+"_exp.csv"
    scRNA_meta = scRNApath_+"_meta.csv"

    bulkRNA_exp = bulkRNApath_+"_exp.csv"
    bulkRNA_meta = bulkRNApath_+"_meta.csv"

    ## 1.2 Load data
    savePath_1 = os.path.join(savePath, "1_loaddata")
    savePath_2 = os.path.join(savePath, "2_preprocessing")
    savePath_3 = os.path.join(savePath, "3_Analysis")

    if not os.path.exists(savePath_1):
        os.makedirs(savePath_1, exist_ok=True)

    if not os.path.exists(savePath_2):
        os.makedirs(savePath_2, exist_ok=True)
    
    if not os.path.exists(savePath_3):
        os.makedirs(savePath_3, exist_ok=True)

    ### Bulk
    bulkClinical = load_bulk_clinical(bulkRNA_meta)
    view_dataframe(bulkClinical)

    bulkExp = load_bulk_exp(bulkRNA_exp)
    bulkExp = normalize_data(bulkExp)
    view_dataframe(bulkExp)  ## if user try to view the data

    check_bulk(savePath, bulkExp, bulkClinical)

    ### Single-cell
    ## 1.3 Preprocessing SC data
    sc_exp = pd.read_csv(scRNA_exp,index_col=0)
    sc_meta = pd.read_csv(scRNA_meta)
    scAnndata = AnnData(sc_exp.T)
    scAnndata.obs = sc_meta
    del sc_exp,sc_meta

    scAnndata = Normalization(scAnndata)
    scAnndata = Logtransformation(scAnndata)
    scAnndata = Clustering(scAnndata, infer_mode=infer_mode, savePath=savePath)
    compute_similarity(savePath=savePath, ann_data=scAnndata)

    with open(os.path.join(savePath_2, "scAnndata.pkl"), "wb") as f:
        pickle.dump(scAnndata, f)
    f.close()

    ## 1.4 clinical column selection and bulk data split
    generate_val(
        savePath=savePath, validation_proportion=0.15, mode=mode
    )  ## optinal parameter: validation_proportion

    perform_sampling_on_RNAseq(savePath=savePath,mode="SMOTE", threshold=0.5)     ## sampling

    ## 1.5 Save informative genes and genepair matrix
    GenePairExtractor.run_extraction = run_extraction_
    GenePairExtractor.save_data = save_data_

    GPextractor = GenePairExtractor(
        savePath=savePath,
        analysis_mode=mode,
        top_var_genes=1500,
        top_gene_pairs=200,
        p_value_threshold=0.05,
        max_cutoff=0.8,
        min_cutoff=-0.8,
    )  ## optinal parameter: top_var_genes, top_gene_pairs, padj_value_threshold, padj_value_threshold

    GPextractor.load_data()
    GPextractor.run_extraction()
    GPextractor.save_data()

    ## 2.1 TiRank
    PackData(savePath, mode=mode, infer_mode=infer_mode, batch_size=1024)

    ### MLP
    initial_model_para(
        savePath=savePath,
        nhead=2,
        nhid1=32,
        nhid2=8,
        n_output=16,
        nlayers=3,
        n_pred=2,
        dropout=0.2,
        mode=mode,
        encoder_type="MLP",
        infer_mode=infer_mode,
    )

    tune_hyperparameters(
        ## Parameters Path
        savePath=savePath,
        device=device,
        n_trials=10,
    )  ## optional parameters: n_trials

    Predict_(savePath=savePath, mode=mode, do_reject=False, tolerance=0.05, reject_mode="GMM",suffix="MLP_1")

    plot_score_distribution_(savePath,suffix="MLP_1")