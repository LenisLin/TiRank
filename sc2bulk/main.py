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

sys.path.append("./scRank")

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
                                          "bulkClinical/Clinical_GSE39582.txt"), sep="\t")
bulkClinical.head()

bulkClinical.columns = ["ID", "Time", "Event"]
bulkClinical.index = bulkClinical["ID"]
del bulkClinical["ID"]

# load bulk expression profile
bulkExp = pd.read_csv(os.path.join(dataPath, "bulkExp/GSE39582_exp.csv"))

# load RNA-seq and scRNA-seq expression profile
scPath = "/mnt/data/lyx/scRankv2/data/sc/"
# scExp = pd.read_csv(os.path.join(scPath,"GSE144735_exp.csv"))
# scClinical = pd.read_csv(os.path.join(scPath,"GSE144735_anno.csv"))

# scExp_ = scExp.T
# scExp_.index = scClinical.index
# scAnndata = sc.AnnData(X=scExp_,obs=scClinical)

# scAnndata.write_h5ad(filename=os.path.join(scPath,"GSE144735.h5ad"))
scAnndata = sc.read_h5ad(os.path.join(scPath, "GSE144735.h5ad"))

# Preprocessing scRNA-seq data
# scAnndata_magic = perform_magic_preprocessing(scAnndata,require_normalization=True)
similarity_df = calculate_cells_similarity(
    input_data=scAnndata, require_normalization=True)
with open(os.path.join(savePath, 'similarity_df.pkl'), 'wb') as f:
    pickle.dump(similarity_df, f)
f.close()

f = open(os.path.join(savePath, 'similarity_df.pkl'),'rb')
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

with open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'wb') as f:
    pickle.dump(bulk_gene_pairs_mat, f)
f.close()

with open(os.path.join(savePath, 'single_cell_gene_pairs_mat.pkl'), 'wb') as f:
    pickle.dump(single_cell_gene_pairs_mat, f)
f.close()

f = open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'),'rb')
bulkGPM = pickle.load(f)
f.close()

f = open(os.path.join(savePath, 'single_cell_gene_pairs_mat.pkl'),'rb')
scGPM = pickle.load(f)
f.close()


# Define your train_loader and test_loader here
train_dataset_Bulk = BulkDataset(bulkGPM, bulkClinical["Time"], bulkClinical["Event"])
# test_dataset_Bulk = BulkDataset(df_Xa_test, df_t_test, df_e_test)

train_dataset_SC = SCDataset(scGPM)
# test_dataset_SC = SCDataset(df_Xb_test)

train_loader_Bulk = DataLoader(train_dataset_Bulk, batch_size=96, shuffle=True)
# test_loader_Bulk = DataLoader(test_dataset_Bulk, batch_size=16, shuffle=False)

# Use a larger batch size for X_b since it has more samples
train_loader_SC = DataLoader(train_dataset_SC, batch_size=256, shuffle=True)
# test_loader_SC = DataLoader(test_dataset_SC, batch_size=256, shuffle=False)

# training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define your model here
# model = TransformerEncoderModel(n_features = bulkGPM.shape[1], nhead = 2, nhid = 32, nlayers = 2, n_output = 8, dropout=0.5)
model = scRank(n_features=bulkGPM.shape[1], nhead = 2, nhid1=96, nhid2 = 8, n_output=32, nlayers = 3, dropout=0.5, encoder_type = "MLP")
model = model.to(device)

optimizer = Adam(model.parameters(), lr=0.001)

# Define learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Hyperparameters for the losses
alphas = [1, 1, 1]

# Assign the mode of analysis
mode = "SC"
if mode == "SC":
    adj_A = torch.from_numpy(similarity_df.values)

n_epochs = 50

for epoch in range(n_epochs):
    train_loss = Train_one_epoch(
        model = model, 
        dataloader_A = train_loader_Bulk, dataloader_B = train_loader_SC, 
        adj_A = adj_A, adj_B = None, 
        optimizer = optimizer, alphas = alphas, mode = "SC", device = device)
    # val_loss = evaluate(model, test_loader_Bulk, test_loader_SC) # You need to modify evaluate function to use two dataloaders

    # Step the scheduler
    scheduler.step()

    # print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}, Validation Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]}")
    print(
        f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}, LR: {scheduler.get_last_lr()[0]}")

torch.save(model.state_dict(), "model.pt")

# save model
model = MLP(n_features=bulkGPM.shape[1], nhid=96, n_output=32, dropout=0.5)
model.load_state_dict(torch.load("./model.pt"))
model = model.to("cpu")

Predict(model, bulkGPM, "./GSE38832riskScore.csv")
Predict(model, scGPM, "./GSE144735riskScore.csv")
