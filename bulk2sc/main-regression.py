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
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
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
    n_trials=10
)

print("Best hyperparameters:", best_params)


# Predict

mode = "Regression"
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, n_pred=1, dropout=0.5, mode=mode, encoder_type=encoder_type)
model.load_state_dict(torch.load(os.path.join("./checkpoints/","model_trial_2_val_loss_0.2831.pt")))
model = model.to("cpu")

sc_PredDF = Predict(model, bulk_GPmat=bulk_gene_pairs_mat, sc_GPmat=single_cell_gene_pairs_mat,
                    mode="Regression", sc_rownames=scAnndata.obs.index.tolist(), do_reject=True)

scAnndata = categorize(scAnndata, sc_PredDF, do_cluster=False, mode = mode)

# Test
Exp_sc = single_cell_gene_pairs_mat
Exp_Tensor_sc = torch.from_numpy(np.array(Exp_sc))
Exp_Tensor_sc = torch.tensor(Exp_Tensor_sc, dtype=torch.float32)

embeddings_sc, prob_scores_sc, _ = model(Exp_Tensor_sc)
pred_prob_sc = prob_scores_sc.detach().numpy().reshape(-1, 1)

Exp_bulk = bulk_gene_pairs_mat
Exp_Tensor_bulk = torch.from_numpy(np.array(Exp_bulk))
Exp_Tensor_bulk = torch.tensor(Exp_Tensor_bulk, dtype=torch.float32)

embeddings_bulk, prob_bulkores_bulk, _ = model(Exp_Tensor_bulk)
pred_prob_bulk = prob_bulkores_bulk.detach().numpy().reshape(-1, 1)

# Print the entire code for user's own data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Replace with your actual data
true_labels_sc = scAnndata.obs["response"]
predicted_scores_sc = scAnndata.obs["Rank_Score"]
predicted_label_sc = scAnndata.obs["Rank_Label"]

true_labels_bulk = bulkClinical["Variable"]
predicted_scores_bulk  = pred_prob_bulk

# Convert to DataFrame for ease of plotting
sc_data = pd.DataFrame({'True Label': true_labels_sc, 'Predicted Score': predicted_scores_sc, 'Predicted Score with reject': predicted_label_sc})
bulk_data = pd.DataFrame({'True Label': true_labels_bulk})
bulk_data['Predicted Score'] = predicted_scores_bulk

# Create a figure with 1 row and 4 columns of subplots
fig, ax = plt.subplots(1, 5, figsize=(30, 7))

# Plot 1: Boxplot for single-cell data
sns.boxplot(x='True Label', y='Predicted Score', data=sc_data, ax=ax[0])
ax[0].set_title('Boxplot of Predicted Scores by True Labels (SC)')

# Statistical Test for Plot 1
group0 = sc_data[sc_data['True Label'] == 0]['Predicted Score']
group1 = sc_data[sc_data['True Label'] == 1]['Predicted Score']

stat, p_value = mannwhitneyu(group0, group1)
ax[0].text(0.5, 0.95, f'p = {p_value:.2e}', ha='center', va='center', transform=ax[0].transAxes)

# Plot 2: Boxplot for single-cell data
sns.boxplot(x='True Label', y='Predicted Score with reject', data=sc_data[sc_data['Predicted Score with reject'] != 0], ax=ax[1])
ax[1].set_title('Boxplot of Predicted Scores (with reject) by True Labels (SC)')

# Statistical Test for Plot 2
group0 = sc_data[sc_data['True Label'] == 0]['Predicted Score with reject']
group1 = sc_data[sc_data['True Label'] == 1]['Predicted Score with reject']

group0 = group0[group0 != 0]
group1 = group1[group1 != 0]

stat, p_value = mannwhitneyu(group0, group1)
ax[1].text(0.5, 0.95, f'p = {p_value:.2e}', ha='center', va='center', transform=ax[1].transAxes)

# Plot 3: Distribution of predicted and true scores for bulk data
sns.kdeplot(predicted_scores_bulk, shade=True, linewidth=3, label='Predict Bulk', ax=ax[2])
sns.kdeplot(bulk_data['True Label'], shade=True, linewidth=3, label='True Bulk', ax=ax[2])
ax[2].set_title('Density Plot of True and Predicted Scores (Bulk)')
ax[2].legend()

# Plot 4: Distribution of predicted scores for single-cell data
sns.histplot(predicted_scores_sc, bins=20, kde=True, ax=ax[3])
ax[3].set_title('Distribution of Predicted Scores (SC)')

# Plot 5: Density plot for predicted scores of bulk and single-cell data
sns.kdeplot(predicted_scores_bulk, shade=True, linewidth=3, label='Bulk', ax=ax[4])
sns.kdeplot(predicted_scores_sc, shade=True, linewidth=3, label='Single Cell', ax=ax[4])
ax[4].set_title('Density Plot of Predicted Scores (Bulk vs SC)')
ax[4].legend()

plt.tight_layout()
plt.savefig("test.png")
plt.show()