# main
import warnings
warnings.filterwarnings("ignore")

import torch
import scanpy as sc
import pandas as pd
import pickle
import os
import sys

sys.path.append("../../scRank")
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
dataPath = "/home/lenislin/Experiment/data/scRankv2/data/RNAseq_treatment/Melanoma/"
savePath = "./tempfiles/"

if not (os.path.exists(savePath)):
    os.makedirs(savePath)

# load clinical data
bulkClinical = pd.read_table(os.path.join(
    dataPath, "Liu2019_meta.csv"), sep=",", index_col=0)
bulkClinical.columns = ["Group", "OS_status", "OS_time"]
bulkClinical['Group'] = bulkClinical['Group'].apply(
    lambda x: 0 if x in ['PR', 'CR', 'CRPR'] else 1)
bulkClinical = pd.DataFrame(bulkClinical['Group'])

bulkClinical.head()

# load bulk expression profile
bulkExp = pd.read_csv(os.path.join(dataPath, "Liu2019_exp.csv"), index_col=0)
bulkExp = normalize_data(bulkExp)
bulkExp.shape
bulkExp.iloc[0:5, 0:5]

# load validation profile
for dataId in ["Hugo2016", "PUCH2021", "Riaz2017"]:  # "Gide2019", "VanAllen2015"
    val_bulkClinical = pd.read_table(os.path.join(
        dataPath, dataId+"_meta.csv"), sep=",", index_col=0)
    val_bulkClinical.columns = ["Group", "OS_status", "OS_time"]
    val_bulkClinical['Group'] = val_bulkClinical['Group'].apply(
        lambda x: 0 if x in ['PR', 'CR', 'CRPR'] else 1)
    val_bulkClinical = pd.DataFrame(val_bulkClinical['Group'])
    val_bulkExp = pd.read_csv(os.path.join(
        dataPath, dataId+"_exp.csv"), index_col=0)
    val_bulkExp = normalize_data(val_bulkExp)

    # merge two datasets
    bulkExp, bulkClinical = merge_datasets(
        bulkClinical_1=bulkClinical, bulkClinical_2=val_bulkClinical, bulkExp_1=bulkExp, bulkExp_2=val_bulkExp)

bulkExp_train, bulkExp_val, bulkClinical_train, bulkClinical_val = generate_val(bulkExp, bulkClinical, validation_proportion=0.15)

# sampling
bulkExp_train, bulkClinical_train = perform_sampling_on_RNAseq(
    bulkExp=bulkExp_train, bulkClinical=bulkClinical_train, mode="SMOTE", threshold=0.5)


# load RNA-seq and scRNA-seq expression profile
scPath = "/home/lenislin/Experiment/data/scRankv2/data/scRNAseq/SKCM/"
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
    bulk_expression=bulkExp_train,
    clinical_data=bulkClinical_train,
    single_cell_expression=scExp,
    analysis_mode="Bionomial",
    top_var_genes=2500,
    top_gene_pairs=500,
    # padj_value_threshold=0.1,
    p_value_threshold=0.05,
    max_cutoff=0.8,
    min_cutoff=-0.8
)

bulk_gene_pairs_mat, single_cell_gene_pairs_mat = GPextractor.run_extraction()
bulk_gene_pairs_mat = pd.DataFrame(bulk_gene_pairs_mat.T)
val_bulkExp_gene_pairs_mat = transform_test_exp(train_exp = bulk_gene_pairs_mat,test_exp = bulkExp_val) ## validation
single_cell_gene_pairs_mat = pd.DataFrame(single_cell_gene_pairs_mat.T)

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

# Define dataset and dataloader
mode = "Bionomial"

train_dataset_Bulk = BulkDataset(bulk_gene_pairs_mat, bulkClinical_train, mode=mode)
val_dataset_Bulk = BulkDataset(val_bulkExp_gene_pairs_mat, bulkClinical_val, mode=mode)
train_dataset_SC = SCDataset(single_cell_gene_pairs_mat)

train_loader_Bulk = DataLoader(train_dataset_Bulk, batch_size=1024, shuffle=False) ## Bulk - train
val_loader_Bulk = DataLoader(val_dataset_Bulk, batch_size=1024, shuffle=False) ## Bulk - val
train_loader_SC = DataLoader(train_dataset_SC, batch_size=1024, shuffle=True) ## SC

# Assign the mode of analysis
infer_mode = "Cell"
if infer_mode == "Cell":
    adj_A = torch.from_numpy(similarity_df.values)

# Training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define your model here
encoder_type = "MLP"
mode = "Bionomial"

# Hyper-parameter searching
best_params = tune_hyperparameters(
    # Model parameter
    n_features=bulk_gene_pairs_mat.shape[1],
    nhead=2, nhid1=96, nhid2=8, n_output=32,
    nlayers=3, n_pred=2, dropout=0.5, mode=mode, encoder_type=encoder_type,

    # Data
    train_loader_Bulk=train_loader_Bulk,
    val_loader_Bulk=val_loader_Bulk,
    train_loader_SC=train_loader_SC,
    adj_A=adj_A,

    device=device,
    infer_mode=infer_mode,
    n_trials=20
)

print("Best hyperparameters:", best_params)

## Predict
mode = "Bionomial"
model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
               nhid2=8, n_output=32, nlayers=3, n_pred=2, dropout=0.5, mode=mode, encoder_type=encoder_type)
model.load_state_dict(torch.load(os.path.join(
    "./checkpoints/", "model_trial_2_val_loss_0.9336.pt")))
model = model.to("cpu")

bulk_gene_pairs_mat_all = transform_test_exp(train_exp = bulk_gene_pairs_mat,test_exp = bulkExp)
bulk_PredDF, sc_PredDF = Predict(model,
                                 bulk_GPmat=bulk_gene_pairs_mat_all, sc_GPmat=single_cell_gene_pairs_mat,
                                 mode="Bionomial",
                                 bulk_rownames=bulkClinical.index.tolist(), sc_rownames=scAnndata.obs.index.tolist(),
                                 do_reject=True, tolerance=0.05, reject_mode="GMM")

scAnndata = categorize(scAnndata, sc_PredDF, do_cluster=False)
sc_pred_df = scAnndata.obs
sc_pred_df.to_csv(os.path.join(savePath, "sc_predict_score.csv"))

pred_prob_sc = sc_PredDF["Pred_score"]  # scRNA
pred_prob_bulk = bulk_PredDF["Pred_score"]  # Bulk RNA

# Display the prob score distribution
figurePath = os.path.join(savePath, "figures")
plot_prob_distribution(pred_prob_bulk, pred_prob_sc, os.path.join(
    figurePath, 'SKCM scRank Pred Score Distribution.png'))

# Save scRNA data
scAnndata.write(os.path.join(savePath, "scAnndata.h5ad"))

# Evaluate on other data
test_set = ["Gide2019", "Hugo2016", "Liu2019",
            "PUCH2021", "Riaz2017", "VanAllen2015"]
evaluate_on_test_data(model, test_set, data_path=dataPath, save_path=figurePath,
                      bulk_gene_pairs_mat=bulk_gene_pairs_mat)
