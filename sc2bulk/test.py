# main
import warnings
warnings.filterwarnings("ignore")

import torch
import scanpy as sc
import anndata as ad
import pandas as pd
import pickle
import os
import sys

sys.path.append("../scRank")
from TrainPre import *
from Visualization import *
from SCSTpreprocess import *
from Dataloader import *
from GPextractor import *
from Model import *
from Loss import *
from torch.utils.data import DataLoader

setup_seed(619)

# Dictionary Path
dataPath = "/home/lenislin/Experiment/data/scRankv2/data/"
savePath_ = "./tempfiles/"

if not (os.path.exists(savePath_)):
    os.makedirs(savePath_)

# load clinical data
bulkClinical = pd.read_table(os.path.join(dataPath,"RNAseq_prog/CRC/clinical/GSE39582_clinical.csv"), sep=",")
bulkClinical.head()

bulkClinical.columns = ["ID", "Time", "Event"]
bulkClinical.index = bulkClinical["ID"]
del bulkClinical["ID"]

# load bulk expression profile
bulkExp = pd.read_csv(os.path.join(dataPath, "RNAseq_prog/CRC/exp/GSE39582_exp.csv"), index_col=0)
# remain validation set
bulkExp_train, bulkClinical_train = bulkExp, bulkClinical
# bulkExp_train, bulkExp_val, bulkClinical_train, bulkClinical_val = generate_val(bulkExp, bulkClinical, validation_proportion=0.15,mode="Cox")

# sampling
# bulkExp, bulkClinical = perform_sampling_on_RNAseq(bulkExp = bulkExp, bulkClinical = bulkClinical, mode="SMOTE", threshold=0.5)

# load RNA-seq and scRNA-seq expression profile
scPath = os.path.join(dataPath,"scRNAseq","CRC")
datasets = "GSE144735"

# Preprocessing SC data
# sc_exp = pd.read_csv(os.path.join("/home/lenislin/Experiment/data/scRankv2/data/scRNAseq/CRC/GSE144735_exp.csv"))
# sc_meta = pd.read_csv(os.path.join("/home/lenislin/Experiment/data/scRankv2/data/scRNAseq/CRC/GSE144735_anno.csv"))
# scAnndata = ad.AnnData(sc_exp.T)
# scAnndata.obs = sc_meta
# del sc_exp,sc_meta

savePath = os.path.join(savePath_,datasets)
if not (os.path.exists(savePath)):
    os.makedirs(savePath)

# scAnndata.write(os.path.join(savePath,'anndata.h5ad'), compression="gzip")
scAnndata = sc.read_h5ad(os.path.join(savePath, "anndata.h5ad"))

# scAnndata_magic = perform_magic_preprocessing(scAnndata,require_normalization=True)

## Calculate similarity matrix
similarity_df = calculate_cells_similarity(input_data=scAnndata, require_normalization=False)

# Get gene-pairs matrix
scExp = pd.DataFrame(scAnndata.X.T)
scExp.index = scAnndata.var_names
scExp.column = scAnndata.obs.index

GPextractor = GenePairExtractor(
    bulk_expression=bulkExp_train,
    clinical_data=bulkClinical_train,
    single_cell_expression=scExp,
    analysis_mode="Cox",
    top_var_genes=2000,
    top_gene_pairs=100,
    p_value_threshold=0.05,
    padj_value_threshold=None,
    max_cutoff=0.8,
    min_cutoff=-0.8
    )

bulk_gene_pairs_mat, sc_gene_pairs_mat = GPextractor.run_extraction()
bulk_gene_pairs_mat = pd.DataFrame(bulk_gene_pairs_mat.T)
sc_gene_pairs_mat = pd.DataFrame(sc_gene_pairs_mat.T)

## Save temp files
with open(os.path.join(savePath, 'similarity_df.pkl'), 'wb') as f:
    pickle.dump(similarity_df, f) ## similarity matrix
f.close()

with open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'wb') as f:
    pickle.dump(bulk_gene_pairs_mat, f) ## training bulk gene pair matrix
f.close()

with open(os.path.join(savePath, 'sc_gene_pairs_mat.pkl'), 'wb') as f:
    pickle.dump(sc_gene_pairs_mat, f) ## sc gene pair matrix
f.close()


# ## Load temp files
f = open(os.path.join(savePath, 'similarity_df.pkl'), 'rb')
similarity_df = pickle.load(f)
f.close()

# gene pair matrix
f = open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'rb')
bulk_gene_pairs_mat = pickle.load(f)
f.close()

f = open(os.path.join(savePath, 'sc_gene_pairs_mat.pkl'), 'rb')
sc_gene_pairs_mat = pickle.load(f)
f.close()

## Define your train_loader and test_loader here
mode="Cox"
train_dataset_Bulk = BulkDataset(bulk_gene_pairs_mat, bulkClinical, mode=mode)
train_dataset_SC = SCDataset(sc_gene_pairs_mat)

train_loader_Bulk = DataLoader(train_dataset_Bulk, batch_size=1024, shuffle=False)
train_loader_SC = DataLoader(train_dataset_SC, batch_size=1024, shuffle=True)

## Training setting
device = "cuda" if torch.cuda.is_available() else "cpu"
infer_mode = "Cell"
encoder_type = "MLP"

if infer_mode == "Cell":
    adj_A = torch.from_numpy(similarity_df.values)


## Hyper-parameter searching
model_save_path = os.path.join(savePath,"checkpoints")

n_features=bulk_gene_pairs_mat.shape[1]
nhead=2
nhid1=96
nhid2=8
n_output=32
nlayers=3
n_pred=1
dropout=0.5

model = scRank(n_features=n_features, nhead=nhead, nhid1=nhid1, nhid2=nhid2, n_output=n_output,
                   nlayers=nlayers, n_pred=n_pred, dropout=dropout, mode=mode, encoder_type=encoder_type)
model = model.to(device)

lr = 1e-4
n_epochs = 1000
alphas = [1,1,0.1,1]

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

optimizer = Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
train_loss_dcit = dict()

for epoch in range(n_epochs):
    # Train
    model.train()
    train_loss_dcit_epoch = Train_one_epoch(
        model=model,
        dataloader_A=train_loader_Bulk, dataloader_B=train_loader_SC,
        mode=mode, infer_mode=infer_mode,
        adj_A=adj_A,adj_B = None, pre_patho_labels = None,
        optimizer=optimizer, alphas=alphas, device=device)

    train_loss_dcit["Epoch_"+str(epoch+1)] = train_loss_dcit_epoch

    scheduler.step()

model_filename = os.path.join(model_save_path,f"iter2_model_lr_{lr}_epochs_{n_epochs}.pt")
torch.save(model.state_dict(), os.path.join(model_filename))

model1 = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
            nhid2=8, n_output=32, nlayers=3, n_pred=1, n_patho=6,dropout=0.5, mode=mode, encoder_type=encoder_type)
model1.load_state_dict(torch.load("/home/lenislin/Experiment/projects/scRankv2/sc2bulk/tempfiles/GSE144735/checkpoints/iter1_model_lr_0.0001_epochs_1000.pt"))
model1 = model1.to("cpu")

model2 = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
            nhid2=8, n_output=32, nlayers=3, n_pred=1, n_patho=6,dropout=0.5, mode=mode, encoder_type=encoder_type)
model2.load_state_dict(torch.load("/home/lenislin/Experiment/projects/scRankv2/sc2bulk/tempfiles/GSE144735/checkpoints/iter2_model_lr_0.0001_epochs_1000.pt"))
model2 = model2.to("cpu")

bulk_gene_pairs_mat_all = transform_test_exp(train_exp = bulk_gene_pairs_mat,test_exp = bulkExp)
bulk_PredDF, sc_PredDF1 = Predict(model1,
                                bulk_GPmat=bulk_gene_pairs_mat_all, sc_GPmat=sc_gene_pairs_mat,
                                mode="Cox",
                                bulk_rownames=bulkClinical.index.tolist(), sc_rownames=scAnndata.obs.index.tolist(),
                                do_reject=True, tolerance=0.05, reject_mode="GMM")

bulk_PredDF, sc_PredDF2 = Predict(model2,
                                bulk_GPmat=bulk_gene_pairs_mat_all, sc_GPmat=sc_gene_pairs_mat,
                                mode="Cox",
                                bulk_rownames=bulkClinical.index.tolist(), sc_rownames=scAnndata.obs.index.tolist(),
                                do_reject=True, tolerance=0.05, reject_mode="GMM")

scAnndata.obs["Reject_1"] = sc_PredDF1["Reject"]
scAnndata.obs["Pred_score_1"] = sc_PredDF1["Pred_score"]
scAnndata.obs["Reject_2"] = sc_PredDF2["Reject"]
scAnndata.obs["Pred_score_2"] = sc_PredDF2["Pred_score"]

## plot the umap of score distribution
import scanpy as sc
import matplotlib.pyplot as plt
sc.pp.normalize_total(scAnndata, target_sum=1e4)
sc.pp.log1p(scAnndata)
sc.pp.highly_variable_genes(scAnndata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.tl.pca(scAnndata, svd_solver='arpack')
sc.pp.neighbors(scAnndata, n_neighbors=10)
sc.tl.umap(scAnndata)

sc.pl.umap(scAnndata, color=['seurat_clusters', 'Pred_score_1', 'Pred_score_2'])
plt.savefig("test.pdf")

scAnndata = categorize(scAnndata=scAnndata,sc_PredDF=sc_PredDF, do_cluster=True)
sc_pred_df = scAnndata.obs
sc_pred_df.to_csv(os.path.join(savePath, "spot_predict_score.csv"))

pred_prob_sc = sc_PredDF["Pred_score"]  # scRNA
pred_prob_bulk = bulk_PredDF["Pred_score"]  # Bulk RNA

# Display the prob score distribution
plot_prob_distribution(pred_prob_bulk, pred_prob_sc, os.path.join(
    savePath, 'scRank Pred Score Distribution.png'))

scAnndata.write_h5ad(filename=os.path.join(
    savePath, "downstream_anndata.h5ad"))


# from lifelines import CoxPHFitter
# bulkClinical["RiskScore"] = pred_prob_bulk
# cph = CoxPHFitter()
# cph.fit(bulkClinical, duration_col='Time', event_col='Event')

# # print the coefficients (log hazard ratios)
# print(cph.summary)