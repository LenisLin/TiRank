# main
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import scanpy as sc

import torch
from torch.utils.data import DataLoader

sys.path.append("../../scRank")

from Loss import *
from Model import *
from GPextractor import *
from Dataloader import *
from SCSTpreprocess import *
from TrainPre import *
from Visualization import *

# Dictionary Path
dataPath = "/mnt/data/lyx/scRankv2/data/RNAseq_treatment/Melanoma/"
savePath = "./tempfiles/"

if not (os.path.exists(savePath)):
    os.makedirs(savePath)

# load clinical data
bulkClinical = pd.read_table(os.path.join(dataPath,"PUCH2021_meta.csv"), sep=",", index_col=0)
bulkClinical.columns = ["Group","OS_status","OS_time"]
bulkClinical['Group'] = bulkClinical['Group'].apply(lambda x: 0 if x in ['PR', 'CR', 'CRPR'] else 1)
bulkClinical = pd.DataFrame(bulkClinical['Group']) 

bulkClinical.head()

# load bulk expression profile
bulkExp = pd.read_csv(os.path.join(dataPath, "PUCH2021_exp.csv"), index_col=0)

bulkExp.shape
bulkExp.iloc[0:5, 0:5]

# load validation profile
for dataId in ["Hugo2016","VanAllen2015","Riaz2017"]: # "Gide2019","Liu2019"
    val_bulkClinical = pd.read_table(os.path.join(dataPath,dataId+"_meta.csv"), sep=",", index_col=0)
    val_bulkClinical.columns = ["Group","OS_status","OS_time"]
    val_bulkClinical['Group'] = val_bulkClinical['Group'].apply(lambda x: 0 if x in ['PR', 'CR', 'CRPR'] else 1)
    val_bulkClinical = pd.DataFrame(val_bulkClinical['Group']) 
    val_bulkExp = pd.read_csv(os.path.join(dataPath, dataId+"_exp.csv"), index_col=0)
    # test_bulkExp.iloc[0:5, 0:5]

    # merge two datasets
    bulkExp, bulkClinical = merge_datasets(bulkClinical_1 = bulkClinical,bulkClinical_2 = val_bulkClinical,bulkExp_1 = bulkExp,bulkExp_2 = val_bulkExp)

# sampling
bulkExp, bulkClinical = perform_sampling_on_RNAseq(bulkExp = bulkExp, bulkClinical = bulkClinical, mode="SMOTE", threshold=0.5)

# load RNA-seq and scRNA-seq expression profile
scPath = "/mnt/data/lyx/scRankv2/data/scRNAseq/SKCM/"
scAnndata = sc.read_h5ad(os.path.join(scPath, "GSE120575.h5ad"))

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
scExp = pd.DataFrame(scAnndata.X.T)
scExp.index = scAnndata.var_names
scExp.column = scAnndata.obs.index

GPextractor = GenePairExtractor(
    bulk_expression=bulkExp,
    clinical_data=bulkClinical,
    single_cell_expression=scExp,
    analysis_mode="Bionomial",
    top_var_genes=2000,
    top_gene_pairs=1000,
    # padj_value_threshold=0.05,
    p_value_threshold = 0.05,
    max_cutoff=0.8,
    min_cutoff=-0.8
)

bulk_gene_pairs_mat, single_cell_gene_pairs_mat = GPextractor.run_extraction()
bulk_gene_pairs_mat = pd.DataFrame(bulk_gene_pairs_mat.T)
single_cell_gene_pairs_mat = pd.DataFrame(single_cell_gene_pairs_mat.T)

# val_bulkExp_gene_pairs_mat = transform_test_exp(train_exp = bulk_gene_pairs_mat,test_exp = val_bulkExp) ## validation

with open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'wb') as f:
    pickle.dump(bulk_gene_pairs_mat, f)
f.close()

with open(os.path.join(savePath, 'bulkClinical.pkl'), 'wb') as f:
    pickle.dump(bulkClinical, f)
f.close()

with open(os.path.join(savePath, 'single_cell_gene_pairs_mat.pkl'), 'wb') as f:
    pickle.dump(single_cell_gene_pairs_mat, f)
f.close()

f = open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'rb')
bulk_gene_pairs_mat = pickle.load(f)
f.close()

f = open(os.path.join(savePath, 'bulkClinical.pkl'), 'rb')
bulkClinical = pickle.load(f)
f.close()

f = open(os.path.join(savePath, 'single_cell_gene_pairs_mat.pkl'), 'rb')
single_cell_gene_pairs_mat = pickle.load(f)
f.close()

# expand bulk data
# expand_times = (single_cell_gene_pairs_mat.shape[0] / bulk_gene_pairs_mat.shape[0] )/ (2048/256)
# expand_times = int(expand_times)+1

# Define your train_loader and val_loader here
mode = "Bionomial"

# train_dataset_Bulk = BulkDataset(
#     bulk_gene_pairs_mat, bulkClinical, mode=mode)

# val_dataset_Bulk = BulkDataset(
#     val_bulkExp_gene_pairs_mat, val_bulkClinical, mode=mode)

train_dataset_Bulk, val_dataset_Bulk = generate_val(bulk_gene_pairs_mat, bulkClinical, mode = mode, need_val = True, validation_proportion = 0.2)

train_dataset_SC = SCDataset(single_cell_gene_pairs_mat)

train_loader_Bulk = DataLoader(
    train_dataset_Bulk, batch_size=1024, shuffle=False)

val_loader_Bulk = DataLoader(
    val_dataset_Bulk, batch_size=1024, shuffle=False)

# Use a larger batch size for X_b since it has more samples
train_loader_SC = DataLoader(train_dataset_SC, batch_size=1024, shuffle=True)

# training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define your model here
encoder_type = "MLP"
mode = "Bionomial"

# model = TransformerEncoderModel(n_features = bulk_gene_pairs_mat.shape[1], nhead = 2, nhid = 32, nlayers = 2, n_output = 8, dropout=0.5)
# model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
#                nhid2=8, n_output=32, nlayers=3, dropout=0.5, encoder_type="MLP")
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, n_pred=2, dropout=0.5, mode=mode, encoder_type=encoder_type)

model = model.to(device)

# Hyperparameters for the losses
# alphas = [1, 1, 1, 1]

# Assign the mode of analysis
infer_mode = "Cell"
if infer_mode == "Cell":
    adj_A = torch.from_numpy(similarity_df.values)

# Training
# optimizer = Adam(model.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

# n_epochs = 200

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

# for epoch in range(n_epochs):
#     train_loss = Train_one_epoch(
#         model=model,
#         dataloader_A=train_loader_Bulk, dataloader_B=train_loader_SC,
#         pheno="Bionomial", infer_mode=infer_mode,
#         adj_A=adj_A,
#         optimizer=optimizer, alphas=alphas, device=device)

#     # Step the scheduler
#     scheduler.step()

#     # print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}, Validation Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]}")
#     print(
#         f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss}, LR: {scheduler.get_last_lr()[0]}")

# save model
# torch.save(model.state_dict(), os.path.join(savePath,"model.pt"))

mode = "Bionomial"
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, n_pred=2, dropout=0.5, mode=mode, encoder_type=encoder_type)
model.load_state_dict(torch.load(os.path.join("./checkpoints/","model_trial_1_val_loss_0.5055.pt")))
model = model.to("cpu")

sc_PredDF = Predict(model, bulk_GPmat=bulk_gene_pairs_mat, sc_GPmat=single_cell_gene_pairs_mat,
                    mode="Bionomial", sc_rownames=scAnndata.obs.index.tolist(), do_reject=True, tolerance=0.1, reject_mode = "Strict")

scAnndata = categorize(scAnndata, sc_PredDF, do_cluster=False)
sc_pred_df = scAnndata.obs
sc_pred_df.to_csv(os.path.join(savePath,"sc_predict_score.csv"))

# Test
## scRNA
Exp_sc = single_cell_gene_pairs_mat
Exp_Tensor_sc = torch.from_numpy(np.array(Exp_sc))
Exp_Tensor_sc = torch.tensor(Exp_Tensor_sc, dtype=torch.float32)

embeddings, prob_scores_sc, _ = model(Exp_Tensor_sc)
pred_label_sc = torch.max(
    prob_scores_sc, dim=1).indices.detach().numpy().reshape(-1, 1)
pred_prob_sc = torch.nn.functional.softmax(
    prob_scores_sc)[:, 1].detach().numpy().reshape(-1, 1)

## bulk RNA
Exp_bulk = bulk_gene_pairs_mat
Exp_Tensor_bulk = torch.from_numpy(np.array(Exp_bulk))
Exp_Tensor_bulk = torch.tensor(Exp_Tensor_bulk, dtype=torch.float32)

embeddings, prob_bulkores_bulk, _ = model(Exp_Tensor_bulk)
pred_label_bulk = torch.max(
    prob_bulkores_bulk, dim=1).indices.detach().numpy().reshape(-1, 1)
pred_prob_bulk = torch.nn.functional.softmax(
    prob_bulkores_bulk)[:, 1].detach().numpy().reshape(-1, 1)


# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

scAnndata.write(os.path.join(savePath,"scAnndata.h5ad"))

## Evaluate on other data
test_set = ["Gide2019","Hugo2016","Liu2019","PUCH2021","Riaz2017","VanAllen2015"]
for dataId in test_set:
    test_bulkClinical = pd.read_table(os.path.join(dataPath,dataId+"_meta.csv"), sep=",", index_col=0)
    test_bulkClinical.columns = ["Group","OS_status","OS_time"]
    test_bulkClinical['Group'] = test_bulkClinical['Group'].apply(lambda x: 0 if x in ['PR', 'CR', 'CRPR'] else 1)

    # load bulk expression profile
    test_bulkExp = pd.read_csv(os.path.join(dataPath, dataId+"_exp.csv"), index_col=0)
    # test_bulkExp.iloc[0:5, 0:5]

    test_bulkExp_gene_pairs_mat = transform_test_exp(train_exp = bulk_gene_pairs_mat,test_exp = test_bulkExp)
    test_Exp_Tensor_bulk = torch.from_numpy(np.array(test_bulkExp_gene_pairs_mat))
    test_Exp_Tensor_bulk = torch.tensor(test_Exp_Tensor_bulk, dtype=torch.float32)

    test_embeddings, test_prob_bulkores_bulk, _ = model(test_Exp_Tensor_bulk)
    test_pred_label = torch.max(test_prob_bulkores_bulk, dim=1).indices.detach().numpy()
    test_bulkClinical["scRank_Label"] = [x for x in test_pred_label]
    test_bulkClinical.to_csv(os.path.join(savePath,"bulk_pred_score",dataId+"_predict_score.csv"))

    true_labels_bulk = test_bulkClinical['Group']
    predicted_labels_bulk = test_bulkClinical["scRank_Label"]

    cm = confusion_matrix(true_labels_bulk, predicted_labels_bulk)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(savePath, "bulk_confusion", f'Pred on bulk: {dataId}.png') )
    plt.show()
    plt.close()