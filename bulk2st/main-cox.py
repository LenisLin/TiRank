# main
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import pickle

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append("../scRank/")

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
                                          "RNAseq_prog/CRC/clinical/GSE39582_clinical.csv"), sep=",")
bulkClinical.head()

bulkClinical.columns = ["ID", "Time", "Event"]
bulkClinical.index = bulkClinical["ID"]
del bulkClinical["ID"]

# load bulk expression profile
bulkExp = pd.read_csv(os.path.join(
    dataPath, "RNAseq_prog/CRC/exp/GSE39582_exp.csv"), index_col=0)

# sampling
# bulkExp, bulkClinical = perform_sampling_on_RNAseq(bulkExp = bulkExp, bulkClinical = bulkClinical, mode="SMOTE", threshold=0.5)

# load RNA-seq and scRNA-seq expression profile
stPath = "/mnt/data/lyx/scRankv2/data/ST/CRC/"
slices = os.listdir(stPath)
scAnndata = sc.read_visium(os.path.join(stPath, slices[0]))

# Preprocessing scRNA-seq data
# scAnndata_magic = perform_magic_preprocessing(scAnndata,require_normalization=True)
similarity_df, distance_df = compute_spots_similarity(
    input_data=scAnndata, perform_normalization=True)

with open(os.path.join(savePath, 'similarity_df.pkl'), 'wb') as f:
    pickle.dump(similarity_df, f)
f.close()

with open(os.path.join(savePath, 'distance_df.pkl'), 'wb') as f:
    pickle.dump(distance_df, f)
f.close()

f = open(os.path.join(savePath, 'similarity_df.pkl'), 'rb')
similarity_df = pickle.load(f)
f.close()

f = open(os.path.join(savePath, 'distance_df.pkl'), 'rb')
distance_df = pickle.load(f)
f.close()

# scAnndata.write_h5ad(filename=os.path.join(savePath,slices[0]+".h5ad"))
scAnndata = sc.read_h5ad(filename=os.path.join(savePath, slices[0] + ".h5ad"))

# Get gene-pairs matrix
stExp = pd.DataFrame(scAnndata.X.toarray().T)  # 这里的接口很难受，得再调一下
stExp.index = scAnndata.var_names
stExp.column = scAnndata.obs.index

GPextractor = GenePairExtractor(
    bulk_expression=bulkExp,
    clinical_data=bulkClinical,
    single_cell_expression=stExp,
    analysis_mode="Cox",
    top_var_genes=1000,
    top_gene_pairs=1000,
    padj_value_threshold=0.05,
    max_cutoff=0.8,
    min_cutoff=0.1
)

bulk_gene_pairs_mat, st_gene_pairs_mat = GPextractor.run_extraction()
bulk_gene_pairs_mat = pd.DataFrame(bulk_gene_pairs_mat.T)
st_gene_pairs_mat = pd.DataFrame(st_gene_pairs_mat.T)

with open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'wb') as f:
    pickle.dump(bulk_gene_pairs_mat, f)
f.close()

with open(os.path.join(savePath, 'st_gene_pairs_mat.pkl'), 'wb') as f:
    pickle.dump(st_gene_pairs_mat, f)
f.close()

f = open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'rb')
bulk_gene_pairs_mat = pickle.load(f)
f.close()

f = open(os.path.join(savePath, 'st_gene_pairs_mat.pkl'), 'rb')
st_gene_pairs_mat = pickle.load(f)
f.close()

# Define your train_loader and test_loader here
train_dataset_Bulk = BulkDataset(
    bulk_gene_pairs_mat, bulkClinical, mode="Cox")

train_dataset_ST = STDataset(st_gene_pairs_mat)

train_loader_Bulk = DataLoader(
    train_dataset_Bulk, batch_size=1024, shuffle=False)

# Use a larger batch size for X_b since it has more samples
train_loader_ST = DataLoader(train_dataset_ST, batch_size=1024, shuffle=True)

# training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define your model here
encoder_type = "MLP"
mode = "Cox"

# model = TransformerEncoderModel(n_features = bulk_gene_pairs_mat.shape[1], nhead = 2, nhid = 32, nlayers = 2, n_output = 8, dropout=0.5)
# model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
#                nhid2=8, n_output=32, nlayers=3, dropout=0.5, encoder_type="MLP")
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, n_pred=1, dropout=0.5, mode=mode, encoder_type=encoder_type)

model = model.to(device)

# Hyperparameters for the losses
alphas = [2, 4, 4, 1]

# Assign the mode of analysis
infer_mode = "Spot"
if infer_mode == "Cell":
    adj_A = torch.from_numpy(similarity_df.values)
elif infer_mode == "Spot":
    adj_A = torch.from_numpy(similarity_df.values)
    adj_B = torch.from_numpy(distance_df.values)


# Training
optimizer = Adam(model.parameters(), lr=0.003)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

n_epochs = 100

for epoch in range(n_epochs):
    train_loss = Train_one_epoch(
        model=model,
        dataloader_A=train_loader_Bulk, dataloader_B=train_loader_ST,
        pheno="Cox", infer_mode=infer_mode,
        adj_A=adj_A, adj_B=adj_B,
        optimizer=optimizer, alphas=alphas, device=device)

    # Step the scheduler
    scheduler.step()

    # print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}, Validation Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]}")
    print(
        f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}, LR: {scheduler.get_last_lr()[0]}")

# save model
torch.save(model.state_dict(), "model.pt")

mode = "Cox"
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, n_pred=1, dropout=0.5, mode=mode, encoder_type=encoder_type)
model.load_state_dict(torch.load("./model.pt"))
model = model.to("cpu")

sc_PredDF = Predict(model, bulk_GPmat=bulk_gene_pairs_mat, sc_GPmat=st_gene_pairs_mat,
                    mode=mode, sc_rownames=scAnndata.obs.index.tolist(), do_reject=True, tolerance=0.1)


# DownStream analysis
scAnndata = categorize(scAnndata=scAnndata,
                       sc_PredDF=sc_PredDF, do_cluster=True)

scAnndata.write_h5ad(filename=os.path.join(
    savePath, slices[0] + "_downstream.h5ad"))

# Test
Exp_sc = st_gene_pairs_mat
Exp_Tensor_sc = torch.from_numpy(np.array(Exp_sc))
Exp_Tensor_sc = torch.tensor(Exp_Tensor_sc, dtype=torch.float32)

st_embeddings, risk_scores_sc = model(Exp_Tensor_sc)
risk_scores_sc = risk_scores_sc.detach().numpy().reshape(-1, 1)
st_embeddings = st_embeddings.detach().numpy()

Exp_bulk = bulk_gene_pairs_mat
Exp_Tensor_bulk = torch.from_numpy(np.array(Exp_bulk))
Exp_Tensor_bulk = torch.tensor(Exp_Tensor_bulk, dtype=torch.float32)

bulk_embeddings, risk_scores_bulk = model(Exp_Tensor_bulk)
risk_scores_bulk = risk_scores_bulk.detach().numpy().reshape(-1, 1)

from lifelines import CoxPHFitter

# assuming df is your DataFrame and it has columns 'Time', 'Event' and 'RiskScore'

# Using Cox Proportional Hazards model
bulkClinical["RiskScore"] = risk_scores_bulk
cph = CoxPHFitter()
cph.fit(bulkClinical, duration_col='Time', event_col='Event')

# print the coefficients (log hazard ratios)
print(cph.summary)

# score distribution
# Assuming df is your dataframe, and 'Group' column is true labels and 'PredClass' column is predicted labels
# display the prob score distribution
sns.distplot(risk_scores_bulk, hist=False, kde=True, kde_kws={
             'shade': True, 'linewidth': 3}, label='Bulk')
sns.distplot(risk_scores_sc, hist=False, kde=True, kde_kws={
             'shade': True, 'linewidth': 3}, label='Spot')

plt.title('Density Plot')
plt.xlabel('Values')
plt.ylabel('Density')

# Move the legend box to upper left
plt.legend(title='Sample Type', loc='upper left')
plt.savefig('Spot survival pred risk - both.png')

plt.show()
plt.close()
