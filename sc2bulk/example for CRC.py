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
bulkExp_train, bulkExp_val, bulkClinical_train, bulkClinical_val = generate_val(bulkExp, bulkClinical, validation_proportion=0.15,mode="Cox")

# sampling
# bulkExp, bulkClinical = perform_sampling_on_RNAseq(bulkExp = bulkExp, bulkClinical = bulkClinical, mode="SMOTE", threshold=0.5)

# load RNA-seq and scRNA-seq expression profile
scPath = os.path.join(dataPath,"scRNAseq","CRC")
datasets = ["GSE144735","GSE132465"]

for i in range(len(datasets)):
    # Preprocessing SC data
    # sc_exp = pd.read_csv(os.path.join(scPath,datasets[i]+"_exp.csv"))
    # sc_meta = pd.read_csv(os.path.join(scPath,datasets[i]+"_anno.csv"))
    # scAnndata = ad.AnnData(sc_exp.T)
    # scAnndata.obs = sc_meta
    # del sc_exp,sc_meta

    savePath = os.path.join(savePath_,datasets[i])
    if not (os.path.exists(savePath)):
        os.makedirs(savePath)

    # scAnndata.write(os.path.join(savePath,'anndata.h5ad'), compression="gzip")
    scAnndata = sc.read_h5ad(os.path.join(savePath, "anndata.h5ad"))

    # scAnndata_magic = perform_magic_preprocessing(scAnndata,require_normalization=True)

    ## Calculate similarity matrix
    similarity_df = calculate_cells_similarity(input_data=scAnndata, require_normalization=True)

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
        top_gene_pairs=1000,
        p_value_threshold=0.05,
        padj_value_threshold=None,
        max_cutoff=0.8,
        min_cutoff=-0.8
        )

    bulk_gene_pairs_mat, sc_gene_pairs_mat = GPextractor.run_extraction()
    bulk_gene_pairs_mat = pd.DataFrame(bulk_gene_pairs_mat.T)
    val_bulkExp_gene_pairs_mat = transform_test_exp(train_exp = bulk_gene_pairs_mat,test_exp = bulkExp_val) ## validation
    sc_gene_pairs_mat = pd.DataFrame(sc_gene_pairs_mat.T)

    ## Save temp files
    with open(os.path.join(savePath, 'similarity_df.pkl'), 'wb') as f:
        pickle.dump(similarity_df, f) ## similarity matrix
    f.close()

    with open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'wb') as f:
        pickle.dump(bulk_gene_pairs_mat, f) ## training bulk gene pair matrix
    f.close()
    with open(os.path.join(savePath, 'val_bulkExp_gene_pairs_mat.pkl'), 'wb') as f:
        pickle.dump(val_bulkExp_gene_pairs_mat, f) ## validating gene pair matrix
    f.close()
    with open(os.path.join(savePath, 'sc_gene_pairs_mat.pkl'), 'wb') as f:
        pickle.dump(sc_gene_pairs_mat, f) ## sc gene pair matrix
    f.close()


    # ## Load temp files
    # f = open(os.path.join(savePath, 'similarity_df.pkl'), 'rb')
    # similarity_df = pickle.load(f)
    # f.close()

    # # gene pair matrix
    # f = open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'rb')
    # bulk_gene_pairs_mat = pickle.load(f)
    # f.close()
    # f = open(os.path.join(savePath, 'val_bulkExp_gene_pairs_mat.pkl'), 'rb')
    # val_bulkExp_gene_pairs_mat = pickle.load(f)
    # f.close()
    # f = open(os.path.join(savePath, 'st_gene_pairs_mat.pkl'), 'rb')
    # st_gene_pairs_mat = pickle.load(f)
    # f.close()

    ## Define your train_loader and test_loader here
    mode="Cox"
    train_dataset_Bulk = BulkDataset(bulk_gene_pairs_mat, bulkClinical, mode=mode)
    val_dataset_Bulk = BulkDataset(val_bulkExp_gene_pairs_mat, bulkClinical_val, mode=mode)
    train_dataset_SC = SCDataset(sc_gene_pairs_mat)

    train_loader_Bulk = DataLoader(train_dataset_Bulk, batch_size=1024, shuffle=False)
    val_loader_Bulk = DataLoader(val_dataset_Bulk, batch_size=1024, shuffle=False)
    train_loader_SC = DataLoader(train_dataset_SC, batch_size=12288, shuffle=True)

    ## Training setting
    device = "cuda" if torch.cuda.is_available() else "cpu"
    infer_mode = "Cell"
    encoder_type = "MLP"

    if infer_mode == "Cell":
        adj_A = torch.from_numpy(similarity_df.values)
    elif infer_mode == "Spot":
        adj_A = torch.from_numpy(similarity_df.values)
        # adj_B = torch.from_numpy(distance_df.values)
        patholabels = scAnndata.obs["patho_class"]

    ## Hyper-parameter searching
    model_save_path = os.path.join(savePath,"checkpoints")

    best_params = tune_hyperparameters(
        # Model parameter
        n_features=bulk_gene_pairs_mat.shape[1],
        nhead=2, nhid1=96, nhid2=8, n_output=32,
        nlayers=3, n_pred=1, dropout=0.5, mode=mode, encoder_type=encoder_type,

        # Data
        train_loader_Bulk=train_loader_Bulk,
        val_loader_Bulk=val_loader_Bulk,
        train_loader_SC=train_loader_SC,
        adj_A=adj_A, adj_B=None, pre_patho_labels = None,

        device=device,
        infer_mode=infer_mode,
        n_trials=20,
        model_save_path = model_save_path
    )

    print("Best hyperparameters:", best_params)

    ## Prediction
    ## Extract parameters from best_params
    lr = best_params['lr']
    n_epochs = best_params['n_epochs']
    alpha_0 = best_params['alpha_0']
    alpha_1 = best_params['alpha_1']
    alpha_2 = best_params['alpha_2']
    alpha_3 = best_params['alpha_3']

    filename = os.path.join(model_save_path,f"model_lr_{lr}_epochs_{n_epochs}_alpha0_{alpha_0}_alpha1_{alpha_1}_alpha2_{alpha_2}_alpha3_{alpha_3}.pt")

    model = scRank(n_features=bulk_gene_pairs_mat.shape[1], nhead=2, nhid1=96,
                nhid2=8, n_output=32, nlayers=3, n_pred=1, n_patho=6,dropout=0.5, mode=mode, encoder_type=encoder_type)
    model.load_state_dict(torch.load(filename))
    model = model.to("cpu")

    bulk_gene_pairs_mat_all = transform_test_exp(train_exp = bulk_gene_pairs_mat,test_exp = bulkExp)
    bulk_PredDF, sc_PredDF = Predict(model,
                                    bulk_GPmat=bulk_gene_pairs_mat_all, sc_GPmat=st_gene_pairs_mat,
                                    mode="Cox",
                                    bulk_rownames=bulkClinical.index.tolist(), sc_rownames=scAnndata.obs.index.tolist(),
                                    do_reject=True, tolerance=0.05, reject_mode="GMM")

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