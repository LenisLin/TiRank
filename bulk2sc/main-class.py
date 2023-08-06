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
                                          "RNAseq_treatment/Cellline/GDSC_Gefitinib_meta.csv"), sep=",")
bulkClinical.head()

bulkClinical.columns = ["ID", "Group"]
bulkClinical.index = bulkClinical["ID"]
del bulkClinical["ID"]

# load bulk expression profile
bulkExp = pd.read_csv(os.path.join(
    dataPath, "RNAseq_treatment/Cellline/GDSC_Gefitinib_exp.csv"), index_col=0)

bulkExp.shape
bulkExp.iloc[0:5, 0:5]

# subset = False
# if subset:
#     n_samples = bulkClinical.shape[0] # get number of rows
#     n_samples_to_pick = int(n_samples * 0.5) # get 50% of the number of rows

#     random_indices = np.random.choice(n_samples, size=n_samples_to_pick, replace=False)
#     idx_0 = np.where(bulkClinical["Group"] == 0)[0]

#     idx = np.unique(np.concatenate((random_indices,idx_0),axis = 0))

#     bulkExp = bulkExp.iloc[:,idx]
#     bulkClinical = bulkClinical.iloc[idx,:]

# load RNA-seq and scRNA-seq expression profile
scPath = "/mnt/data/lyx/scRankv2/data/scRNAseq/Cellline/"
scExp = pd.read_csv(os.path.join(scPath, "GSE112274_exp.csv"), index_col=0)
scClinical = pd.read_csv(os.path.join(
    scPath, "GSE112274_meta.csv"), index_col=0)

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
    analysis_mode="Bionomial",
    top_var_genes=500,
    top_gene_pairs=1000,
    padj_value_threshold=0.05,
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
mode = "Bionomial"

train_dataset_Bulk = BulkDataset(
    bulk_gene_pairs_mat, bulkClinical, mode=mode)

train_dataset_SC = SCDataset(single_cell_gene_pairs_mat)

train_loader_Bulk = DataLoader(
    train_dataset_Bulk, batch_size=1024, shuffle=False)

# Use a larger batch size for X_b since it has more samples
train_loader_SC = DataLoader(train_dataset_SC, batch_size=1024, shuffle=True)

# training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define your model here
encoder_type = "DenseNet"
mode = "Bionomial"

# model = TransformerEncoderModel(n_features = bulk_gene_pairs_mat.shape[1], nhead = 2, nhid = 32, nlayers = 2, n_output = 8, dropout=0.5)
# model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
#                nhid2=8, n_output=32, nlayers=3, dropout=0.5, encoder_type="MLP")
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, n_pred=2, dropout=0.5, mode=mode, encoder_type=encoder_type)

model = model.to(device)

# Hyperparameters for the losses
alphas = [1, 1, 1]

# Assign the mode of analysis
infer_mode = "Cell"
if infer_mode == "Cell":
    adj_A = torch.from_numpy(similarity_df.values)


# Training
optimizer = Adam(model.parameters(), lr=0.003)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

n_epochs = 300

for epoch in range(n_epochs):
    train_loss = Train_one_epoch(
        model=model,
        dataloader_A=train_loader_Bulk, dataloader_B=train_loader_SC,
        pheno="Bionomial", infer_mode=infer_mode,
        adj_A=adj_A,
        optimizer=optimizer, alphas=alphas, device=device)

    # Step the scheduler
    scheduler.step()

    # print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}, Validation Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]}")
    print(
        f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}, LR: {scheduler.get_last_lr()[0]}")

# save model
torch.save(model.state_dict(), "model.pt")

mode = "Bionomial"
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, n_pred=2, dropout=0.5, mode=mode, encoder_type=encoder_type)
model.load_state_dict(torch.load("./model.pt"))
model = model.to("cpu")

sc_PredDF = Predict(model, bulk_GPmat=bulk_gene_pairs_mat, sc_GPmat=single_cell_gene_pairs_mat,
                    mode="Bionomial", sc_rownames=scAnndata.obs.index.tolist(), do_reject=True, tolerance=0.05)

# Test
Exp_sc = single_cell_gene_pairs_mat
Exp_Tensor_sc = torch.from_numpy(np.array(Exp_sc))
Exp_Tensor_sc = torch.tensor(Exp_Tensor_sc, dtype=torch.float32)

embeddings, prob_scores_sc = model(Exp_Tensor_sc)
pred_label_sc = torch.max(
    prob_scores_sc, dim=1).indices.detach().numpy().reshape(-1, 1)
pred_prob_sc = torch.nn.functional.softmax(
    prob_scores_sc)[:, 1].detach().numpy().reshape(-1, 1)

Exp_bulk = bulk_gene_pairs_mat
Exp_Tensor_bulk = torch.from_numpy(np.array(Exp_bulk))
Exp_Tensor_bulk = torch.tensor(Exp_Tensor_bulk, dtype=torch.float32)

embeddings, prob_bulkores_bulk = model(Exp_Tensor_bulk)
pred_label_bulk = torch.max(
    prob_bulkores_bulk, dim=1).indices.detach().numpy().reshape(-1, 1)
pred_prob_bulk = torch.nn.functional.softmax(
    prob_bulkores_bulk)[:, 1].detach().numpy().reshape(-1, 1)

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Assuming df is your dataframe, and 'Group' column is true labels and 'PredClass' column is predicted labels
true_labels_sc = scAnndata.obs["response"]
predicted_labels_sc = pred_label_sc

mask = (sc_PredDF.iloc[:, 0] == 0)
true_labels_sc = true_labels_sc[mask]
predicted_labels_sc = predicted_labels_sc[mask]

true_labels_bulk = bulkClinical["Group"]
predicted_labels_bulk = pred_label_bulk

# Compute confusion matrix
if True:
    # create a figure with 1 row and 2 columns of subplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))

    # First heatmap
    cm = confusion_matrix(true_labels_sc, predicted_labels_sc)
    accuracy = accuracy_score(true_labels_sc, predicted_labels_sc)
    precision = precision_score(true_labels_sc, predicted_labels_sc)
    recall = recall_score(true_labels_sc, predicted_labels_sc)

    print(f'Single cell Accuracy: {accuracy}')
    print(f'Single cell Precision: {precision}')
    print(f'Single cell Recall: {recall}')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                0, 1], yticklabels=[0, 1], ax=ax[0])  # add ax=ax[0]

    ax[0].set_title('Confusion Matrix (Single Cell)')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')

    # Second heatmap
    cm = confusion_matrix(true_labels_bulk, predicted_labels_bulk)
    accuracy = accuracy_score(true_labels_bulk, predicted_labels_bulk)
    precision = precision_score(true_labels_bulk, predicted_labels_bulk)
    recall = recall_score(true_labels_bulk, predicted_labels_bulk)

    print(f'Bulk Accuracy: {accuracy}')
    print(f'Bulk Precision: {precision}')
    print(f'Bulk Recall: {recall}')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                0, 1], yticklabels=[0, 1], ax=ax[1])  # add ax=ax[1]

    ax[1].set_title('Confusion Matrix (Bulk)')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Actual')

    # Save the entire figure as a .png
    fig.savefig('cell treatment (reject) - both.png')

    plt.show()
    plt.close()

# display the prob score distribution
sns.distplot(pred_prob_bulk, hist=False, kde=True, kde_kws={
             'shade': True, 'linewidth': 3}, label='Bulk')
sns.distplot(pred_prob_sc, hist=False, kde=True, kde_kws={
             'shade': True, 'linewidth': 3}, label='Single Cell')

plt.title('Density Plot')
plt.xlabel('Values')
plt.ylabel('Density')

# Move the legend box to upper left
plt.legend(title='Sample Type', loc='upper left')
plt.savefig('cell treatment pred prob - both.png')

plt.show()
plt.close()
