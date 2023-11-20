# main
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import pickle

import numpy as np
import pandas as pd
import scanpy as sc

import torch
from torch.utils.data import DataLoader

sys.path.append("../scRank")

from Loss import *
from Model import *
from GPextractor import *
from Dataloader import *
from SCSTpreprocess import *
from TrainPre import *
from Visualization import *

# Dictionary Path
dataPath = "/mnt/data/lyx/scRankv2/data/"
savePath = "./tempfiles/"

if not (os.path.exists(savePath)):
    os.makedirs(savePath)

# load clinical data
bulkClinical = pd.read_table(os.path.join(dataPath,
                                          "RNAseq_treatment/Cellline_IC50/GDSC_Cisplatin_meta.csv"), sep=",")
bulkClinical.head()

bulkClinical.columns = ["ID", "Variable"]
bulkClinical.index = bulkClinical["ID"]
del bulkClinical["ID"]

# load bulk expression profile
bulkExp = pd.read_csv(os.path.join(
    dataPath, "RNAseq_treatment/Cellline_IC50/GDSC_Cisplatin_exp.csv"), index_col=0)

bulkExp.shape
bulkExp.iloc[0:5, 0:5]

# sampling
# bulkExp, bulkClinical = perform_sampling_on_RNAseq(bulkExp = bulkExp, bulkClinical = bulkClinical, mode="SMOTE", threshold=0.5)

# load RNA-seq and scRNA-seq expression profile
scPath = "/mnt/data/lyx/scRankv2/data/scRNAseq/Cellline/"
scExp = pd.read_csv(os.path.join(scPath, "GSE117872_Primary_exp.csv"), index_col=0)
scClinical = pd.read_csv(os.path.join(
    scPath, "GSE117872_Primary_meta.csv"), index_col=0)

scExp_ = scExp.T
scExp_.index = scClinical.index
scAnndata = sc.AnnData(X=scExp_, obs=scClinical)
del scExp, scClinical, scExp_

# scAnndata.write_h5ad(filename=os.path.join(savePath,"GSE117872_Primary.h5ad"))
scAnndata = sc.read_h5ad(os.path.join(savePath, "GSE117872_Primary.h5ad"))

# Preprocessing scRNA-seq data
# scAnndata_magic = perform_magic_preprocessing(scAnndata,require_normalization=True)
similarity_df = calculate_cells_similarity(
    input_data=scAnndata, require_normalization=False)
with open(os.path.join(savePath, 'similarity_df.pkl'), 'wb') as f:
    pickle.dump(similarity_df, f)
f.close()

f = open(os.path.join(savePath, 'similarity_df.pkl'), 'rb')
similarity_df = pickle.load(f)
f.close()

# Get gene-pairs matrix
scExp = pd.DataFrame(scAnndata.X.T)  # 这里的接口很难受，得再调一下
scExp.index = scAnndata.var_names
scExp.column = scAnndata.obs.index

GPextractor = GenePairExtractor(
    bulk_expression=bulkExp,
    clinical_data=bulkClinical,
    single_cell_expression=scExp,
    analysis_mode="Regression",
    top_var_genes=2000,
    top_gene_pairs=500,
    padj_value_threshold=0.05,
    max_cutoff=0.8,
    min_cutoff=-0.8
)

bulk_gene_pairs_mat, single_cell_gene_pairs_mat = GPextractor.run_extraction()
bulk_gene_pairs_mat = pd.DataFrame(bulk_gene_pairs_mat.T)
single_cell_gene_pairs_mat = pd.DataFrame(single_cell_gene_pairs_mat.T)

with open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'wb') as f:
    pickle.dump(bulk_gene_pairs_mat, f)
f.close()

with open(os.path.join(savePath, 'single_cell_gene_pairs_mat.pkl'), 'wb') as f:
    pickle.dump(single_cell_gene_pairs_mat, f)
f.close()

f = open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'rb')
bulk_gene_pairs_mat = pickle.load(f)
f.close()

f = open(os.path.join(savePath, 'single_cell_gene_pairs_mat.pkl'), 'rb')
single_cell_gene_pairs_mat = pickle.load(f)
f.close()

# expand bulk data
# expand_times = (single_cell_gene_pairs_mat.shape[0] / bulk_gene_pairs_mat.shape[0] )/ (2048/256)
# expand_times = int(expand_times)+1

# Define your train_loader and test_loader here
mode = "Regression"

train_dataset_Bulk, val_dataset_Bulk = generate_val(bulk_gene_pairs_mat, bulkClinical, mode = mode, need_val = True, validation_proportion = 0.2)

train_dataset_SC = SCDataset(single_cell_gene_pairs_mat)

train_loader_Bulk = DataLoader(
    train_dataset_Bulk, batch_size=1024, shuffle=False)

val_loader_Bulk = DataLoader(
    val_dataset_Bulk, batch_size=1024, shuffle=False)

train_loader_SC = DataLoader(train_dataset_SC, batch_size=1024, shuffle=True)

# training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define your model here
encoder_type = "MLP"
mode = "Regression"

# model = TransformerEncoderModel(n_features = bulk_gene_pairs_mat.shape[1], nhead = 2, nhid = 32, nlayers = 2, n_output = 8, dropout=0.5)
# model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
#                nhid2=8, n_output=32, nlayers=3, dropout=0.5, encoder_type="MLP")
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, n_pred=1, dropout=0.5, mode=mode, encoder_type=encoder_type)

model = model.to(device)

# Hyperparameters for the losses
# alphas = [1, 1, 1, 1]

# Assign the mode of analysis
infer_mode = "Cell"
if infer_mode == "Cell":
    adj_A = torch.from_numpy(similarity_df.values)

# Training
best_params = tune_hyperparameters(
    model=model,
    train_loader_Bulk=train_loader_Bulk,
    val_loader_Bulk=val_loader_Bulk,
    train_loader_SC=train_loader_SC,
    adj_A=adj_A,
    device=device,
    pheno=mode,
    infer_mode=infer_mode,
    n_trials=20
)

print("Best hyperparameters:", best_params)


# Predict
mode = "Regression"
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, n_pred=1, dropout=0.5, mode=mode, encoder_type=encoder_type)
model.load_state_dict(torch.load(os.path.join("./checkpoints/","model_trial_4_val_loss_0.1893.pt")))
model = model.to("cpu")

bulk_PredDF, sc_PredDF = Predict(model, 
bulk_GPmat=bulk_gene_pairs_mat, sc_GPmat=single_cell_gene_pairs_mat,
                    mode="Regression", 
                    bulk_rownames = bulkClinical.index.tolist(),
                    sc_rownames=scAnndata.obs.index.tolist(), 
                    do_reject=True)

scAnndata = categorize(scAnndata, sc_PredDF, do_cluster=False, mode = mode)

# Test
pred_prob_sc = sc_PredDF["Pred_score"]
pred_prob_bulk = bulk_PredDF["Pred_score"]

# Print the entire code for user's own data
true_labels_sc = scAnndata.obs["response"]
predicted_scores_sc = scAnndata.obs["Rank_Score"]
predicted_label_sc = scAnndata.obs["Rank_Label"]
true_labels_bulk = bulkClinical["Variable"]

# Convert to DataFrame
sc_data = pd.DataFrame({'True Label': true_labels_sc, 'Predicted Score': predicted_scores_sc, 'Predicted Score with reject': predicted_label_sc})
bulk_data = pd.DataFrame({'True Label': true_labels_bulk, 'Predicted Score': pred_prob_bulk})

# Create figure and subplots
fig, ax = plt.subplots(1, 5, figsize=(30, 7))

# Plot 1
create_boxplot(sc_data, 'Boxplot of Predicted Scores by True Labels (SC)', ax[0])

# Plot 2
sc_data_rejected = sc_data[sc_data['Predicted Score with reject'] != 0]
create_boxplot(sc_data_rejected, 'Boxplot of Predicted Scores (with reject) by True Labels (SC)', ax[1], score_column='Predicted Score with reject')

# Plot 3
create_density_plot(pred_prob_bulk, 'Predict Bulk', ax[2], 'Density Plot of True and Predicted Scores (Bulk)')
create_density_plot(bulk_data['True Label'], 'True Bulk', ax[2], 'Density Plot of True and Predicted Scores (Bulk)')

# Plot 4
create_hist_plot(predicted_scores_sc, ax[3], 'Distribution of Predicted Scores (SC)')

# Plot 5
create_comparison_density_plot(pred_prob_bulk, 'Bulk', pred_prob_sc, 'Single Cell', ax[4], 'Density Plot of Predicted Scores (Bulk vs SC)')

plt.tight_layout()
plt.savefig("test.png")
plt.show()