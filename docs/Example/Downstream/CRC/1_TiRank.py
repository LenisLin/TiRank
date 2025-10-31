# TiRank Analysis Pipeline Example
# This script demonstrates how to use the TiRank library to integrate spatial transcriptomics (ST)
# and bulk transcriptomics data to identify phenotype-associated spots and determine significant clusters.

# Import necessary libraries and modules
import warnings
warnings.filterwarnings("ignore")

import torch
import pickle
import os

from TiRank.Model import setup_seed, initial_model_para
from TiRank.LoadData import *
from TiRank.SCSTpreprocess import *
from TiRank.Imageprocessing import GetPathoClass
from TiRank.GPextractor import GenePairExtractor
from TiRank.Dataloader import generate_val, PackData
from TiRank.TrainPre import tune_hyperparameters, Predict, Pcluster
from TiRank.Visualization import plot_score_distribution, plot_score_umap, plot_label_distribution_among_conditions,plot_STmap

# --------------------------------------------
# 0. Initial and Load Bulk Data
# --------------------------------------------

allResultPath = "/home/lenislin/Experiment/projectsResult/TiRank/bulk2st/CRC/results"
allDataPath = "/mnt/raid5/ProjectData/TiRank/ST/CRC/slices"
slicesName = os.listdir(allDataPath)

infer_mode = "ST"  # Optional parameter
mode = "Cox" # Define the analysis mode (e.g., "Cox" for survival analysis)
pretrain_path = "/home/lenislin/Experiment/projects/TiRankv2/bulk2st/ctranspath.pth" # Path to the pre-trained image processing model (ensure this file is in the package)
n_patho_cluster = 7  # Number of pathological clusters to identify

device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
encoder_type = "MLP"  # Set the encoder type for the model (e.g., "MLP" for multi-layer perceptron)

## 1.2 Load clinical data
bulkDataPath = "/home/lenislin/Experiment/projects/TiRankv2/ExampleData/ST_Prog_CRC"  # Directory containing your data

path_to_bulk_cli = os.path.join(bulkDataPath, "GSE39582_clinical_os.csv")
bulkClinical = load_bulk_clinical(path_to_bulk_cli)

## 1.3 Load bulk expression profile
path_to_bulk_exp = os.path.join(bulkDataPath, "GSE39582_exp_os.csv")
bulkExp = load_bulk_exp(path_to_bulk_exp)

# --------------------------------------------
# 1. TiRank Iterative
# --------------------------------------------

for sliceID in slicesName:
    # Set random seed for reproducibility
    setup_seed(619)

    ## 1.1 Select a path to save the results
    savePath = os.path.join(allResultPath,sliceID) # directory for saving results
    savePath_1 = os.path.join(savePath, "1_loaddata")
    if not os.path.exists(savePath_1):
        os.makedirs(savePath_1, exist_ok=True)

    ## 1.4 Check consistency between bulk expression and clinical data
    # check_bulk(savePath, bulkExp, bulkClinical)

    ## 1.5 Load spatial transcriptomics (ST) data
    path_to_st_folder = os.path.join(allDataPath, sliceID)
    # scAnndata = load_st_data(path_to_st_folder, savePath)
    # st_exp_df = transfer_exp_profile(scAnndata)
    # view_dataframe(st_exp_df)  # Optional: view the ST expression DataFrame

    # --------------------------------------------
    # 2. Preprocessing
    # --------------------------------------------

    ## 2.1 Select a path to save preprocessing results
    savePath_2 = os.path.join(savePath, "2_preprocessing")
    if not os.path.exists(savePath_2):
        os.makedirs(savePath_2, exist_ok=True)

    ## 2.2 Load the saved AnnData object from step 1
    # with open(os.path.join(savePath_1, "anndata.pkl"), "rb") as f:
    #     scAnndata = pickle.load(f)

    # ## 2.3 Preprocess the ST data
    # # Filtering the data based on counts and mitochondrial gene proportion
    # scAnndata = FilteringAnndata(
    #     scAnndata,
    #     max_count=50000,    # Maximum total counts per cell
    #     min_count=2000,     # Minimum total counts per cell
    #     MT_propor=25,       # Maximum percentage of mitochondrial genes
    #     min_cell=10,        # Minimum number of cells expressing the gene
    #     imgPath=savePath_2  # Path to save images/results
    # )
    # # Normalize the data
    # scAnndata = Normalization(scAnndata)
    # scAnndata = Logtransformation(scAnndata)
    # scAnndata = Clustering(scAnndata, infer_mode=infer_mode, savePath=savePath)

    # # Compute similarity matrix (optional distance calculation)
    # compute_similarity(
    #     savePath=savePath,
    #     ann_data=scAnndata,
    #     calculate_distance=False  # Set to True if distance calculation is needed
    # )

    # # Perform image processing to get pathological classifications
    # scAnndata = GetPathoClass(
    #     adata=scAnndata,
    #     pretrain_path=pretrain_path,
    #     n_clusters=n_patho_cluster,
    #     image_save_path=os.path.join(savePath_2, "patho_label.png")
    #     # Advanced parameters: n_components (PCA components), n_clusters
    # )

    # # Save the processed AnnData object
    # with open(os.path.join(savePath_2, "scAnndata.pkl"), "wb") as f:
    #     pickle.dump(scAnndata, f)

    # ## 2.4 Clinical data preparation and splitting bulk data
    # # Split data into training and validation sets
    # generate_val(
    #     savePath=savePath,
    #     validation_proportion=0.15,  # Optional parameter: proportion of data for validation
    #     mode=mode
    # )

    # ## 2.5 Gene pair transformation
    # # Initialize the GenePairExtractor with parameters
    # GPextractor = GenePairExtractor(
    #     savePath=savePath,
    #     analysis_mode=mode,
    #     top_var_genes=2000,       # Optional: number of top variable genes to select
    #     top_gene_pairs=1000,      # Optional: number of top gene pairs to select
    #     p_value_threshold=0.05,   # Optional: p-value threshold for gene pair selection
    #     max_cutoff=0.8,           # Optional: upper cutoff for correlation coefficient
    #     min_cutoff=-0.8           # Optional: lower cutoff for correlation coefficient
    # )

    # # Load data for gene pair extraction
    # GPextractor.load_data()

    # # Run the gene pair extraction process
    # GPextractor.run_extraction()

    # # Save the extracted gene pairs
    # GPextractor.save_data()

    # # --------------------------------------------
    # # 3. Analysis
    # # --------------------------------------------

    # ## 3.1 TiRank Analysis
    # # Define paths for saving analysis results
    savePath_3 = os.path.join(savePath, "3_Analysis")
    # if not os.path.exists(savePath_3):
    #     os.makedirs(savePath_3, exist_ok=True)

    # ### 3.1.1 Data Loading and Preparation
    # # Pack the data into DataLoader objects for training and validation
    # PackData(
    #     savePath=savePath,
    #     mode=mode,
    #     infer_mode=infer_mode,
    #     batch_size=1024   # Optional parameter: batch size for DataLoader
    # )

    # ### 3.1.2 Model Training
    # # Initialize model parameters
    # initial_model_para(
    #     savePath=savePath,
    #     nhead=2,           # Optional: number of heads in multi-head attention (if using Transformer)
    #     nhid1=96,          # Optional: hidden layer size 1
    #     nhid2=8,           # Optional: hidden layer size 2
    #     n_output=32,       # Optional: output size
    #     nlayers=3,         # Optional: number of layers
    #     n_pred=1,          # Optional: number of predictions (e.g., 1 for regression)
    #     dropout=0.5,       # Optional: dropout rate
    #     mode=mode,
    #     encoder_type=encoder_type,
    #     infer_mode=infer_mode
    # )

    # # Tune hyperparameters using Optuna or other optimization libraries
    # tune_hyperparameters(
    #     savePath=savePath,
    #     device=device,
    #     n_trials=10    # Optional parameter: number of hyperparameter tuning trials
    # )

    # ### 3.1.3 Model Inference
    # # Predict phenotype-associated spots and perform rejection (uncertainty estimation)
    # Predict(
    #     savePath=savePath,
    #     mode=mode,
    #     do_reject=True,        # Optional: whether to perform rejection
    #     tolerance=0.05,        # Optional: tolerance level for rejection
    #     reject_mode="GMM"      # Optional: rejection mode (e.g., "GMM" for Gaussian Mixture Model)
    # )

    # ### 3.1.4 Visualization
    # # Plot the distribution of prediction scores
    # plot_score_distribution(savePath)  # Displays the probability score distribution

    # # Plot UMAP embedding colored by prediction scores
    # plot_score_umap(savePath, infer_mode)

    # Load Pathological annotation
    patho_anno = pd.read_csv(os.path.join(path_to_st_folder,f"Pathologist_Annotations_{sliceID}.csv"))
    pred_df = pd.read_csv(os.path.join(savePath_3, "spot_predict_score.csv"), index_col=0)

    patho_anno.set_index('Barcode', inplace=True)
    pred_df["Patho_anno"] = pred_df.index.map(patho_anno['Pathologist Annotations'])
    pred_df.to_csv(os.path.join(savePath_3, "spot_predict_score.csv")) ## save

    # Pclustering
    Pcluster(savePath=savePath, clusterColName="Patho_anno", perm_n=2001)

    # Plot
    plot_label_distribution_among_conditions(savePath, group="Patho_anno")
    plot_STmap(savePath=savePath, group="Patho_anno")
