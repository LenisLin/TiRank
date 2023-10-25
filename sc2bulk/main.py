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
                                          "RNAseq_prog/CRC/clinical/GSE39582_clinical.csv"), sep=",")
bulkClinical.head()

bulkClinical.columns = ["ID", "Time", "Event"]
bulkClinical.index = bulkClinical["ID"]
del bulkClinical["ID"]

# load bulk expression profile
bulkExp = pd.read_csv(os.path.join(
    dataPath, "RNAseq_prog/CRC/exp/GSE39582_exp.csv"), index_col=0)

# load RNA-seq and scRNA-seq expression profile
scPath = "/mnt/data/lyx/scRankv2/data/scRNAseq/CRC/"
# scExp = pd.read_csv(os.path.join(scPath,"GSE144735_exp.csv"))
# scClinical = pd.read_csv(os.path.join(scPath,"GSE144735_anno.csv"))

# scExp_ = scExp.T
# scExp_.index = scClinical.index
# scAnndata = sc.AnnData(X=scExp_,obs=scClinical)
# del scExp,scClinical,scExp_

# scAnndata.write_h5ad(filename=os.path.join(savePath,"GSE144735.h5ad"))
scAnndata = sc.read_h5ad(os.path.join(savePath, "GSE144735.h5ad"))

# Preprocessing scRNA-seq data


# Get gene-pairs matrix
scExp = pd.DataFrame(scAnndata.X.T)  # 这里的接口很难受，得再调一下
scExp.index = scAnndata.var_names
scExp.column = scAnndata.obs.index

GPextractor = GenePairExtractor(
    bulk_expression=bulkExp,
    clinical_data=bulkClinical,
    single_cell_expression=scExp,
    analysis_mode="Cox",
    top_var_genes=500,
    top_gene_pairs=1000,
    p_value_threshold=0.05,
    max_cutoff=0.8,
    min_cutoff=0.1
)

bulk_gene_pairs_mat, single_cell_gene_pairs_mat = GPextractor.run_extraction()
bulk_gene_pairs_mat = pd.DataFrame(bulk_gene_pairs_mat.T)

single_cell_gene_pairs_mat = pd.DataFrame(single_cell_gene_pairs_mat.T)
single_cell_gene_pairs_mat = calculate_populations_meanRank(
    single_cell_gene_pairs_mat, scAnndata.obs["Cell_subtype"])

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

# Define your train_loader and test_loader here
train_dataset_Bulk = BulkDataset(
    bulk_gene_pairs_mat, bulkClinical, mode="Cox", expand_times=1)

train_dataset_SC = SCDataset(single_cell_gene_pairs_mat)

train_loader_Bulk = DataLoader(
    train_dataset_Bulk, batch_size=256, shuffle=False)

# Use a larger batch size for X_b since it has more samples
train_loader_SC = DataLoader(train_dataset_SC, batch_size=2048, shuffle=False)

# training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define your model here
# model = TransformerEncoderModel(n_features = bulk_gene_pairs_mat.shape[1], nhead = 2, nhid = 32, nlayers = 2, n_output = 8, dropout=0.5)
# model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
#                nhid2=8, n_output=32, nlayers=3, dropout=0.5, encoder_type="MLP")
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, dropout=0.5, encoder_type="DenseNet")

model = model.to(device)

optimizer = Adam(model.parameters(), lr=0.001)

# Define learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

# Hyperparameters for the losses
alphas = [2, 1, 0.1]

# Assign the mode of analysis
infer_mode = "Subpopulation"

n_epochs = 100

for epoch in range(n_epochs):
    train_loss = Train_one_epoch(
        model=model,
        dataloader_A=train_loader_Bulk, dataloader_B=train_loader_SC,
        pheno="Cox", infer_mode=infer_mode,
        adj_A=None,
        optimizer=optimizer, alphas=alphas, device=device)  # infer_type = "SC",

    # Step the scheduler
    scheduler.step()

    # print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}, Validation Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]}")
    print(
        f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}, LR: {scheduler.get_last_lr()[0]}")

torch.save(model.state_dict(), "model.pt")

# save model
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, dropout=0.5, encoder_type="DenseNet")
model.load_state_dict(torch.load("./model.pt"))
model = model.to("cpu")

Predict(model, bulk_gene_pairs_mat,
        bulk_gene_pairs_mat.index.tolist(), "./GSE39582_riskScore.csv")
Predict(model, single_cell_gene_pairs_mat,
        single_cell_gene_pairs_mat.index.tolist(), "./GSE144735_riskScore.csv")