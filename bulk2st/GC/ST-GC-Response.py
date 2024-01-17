# main
import warnings
warnings.filterwarnings("ignore")

import torch
import scanpy as sc
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
from Imageprocessing import GetPathoClass

setup_seed(619)

# Dictionary Path
dataPath = "/home/lenislin/Experiment/data/scRankv2/data/"
savePath_ = "./tempfiles_GC/"

if not (os.path.exists(savePath_)):
    os.makedirs(savePath_)

# load clinical data
bulkClinical = pd.read_table(os.path.join(dataPath,"RNAseq_treatment/GC/PDL1_clinical.csv"), sep=",",index_col=0)
bulkClinical = bulkClinical.loc[:,["Response"]]
bulkClinical['Response'] = bulkClinical['Response'].apply(
    lambda x: 0 if x in ['R'] else 1)
bulkClinical.columns = ["Group"]
bulkClinical.head()

# load bulk expression profile
bulkExp = pd.read_csv(os.path.join(dataPath, "RNAseq_treatment/GC/PDL1_exp.csv"), index_col=0)
# bulkExp = pd.DataFrame(bulkExp.T)

# remain validation set
bulkExp_train, bulkExp_val, bulkClinical_train, bulkClinical_val = generate_val(bulkExp, bulkClinical, validation_proportion=0.15,mode="Bionomial")

# sampling
# bulkExp, bulkClinical = perform_sampling_on_RNAseq(bulkExp = bulkExp, bulkClinical = bulkClinical, mode="SMOTE", threshold=0.5)

# load RNA-seq and scRNA-seq expression profile
stPath = os.path.join(dataPath,"ST","GC_24")
slices = os.listdir(stPath)

# path to H&E pretrained model
pretrain_path = "/home/lenislin/Experiment/projects/scRankv2/scRank/pretrainModel/ctranspath.pth"

for i in range(len(slices)):
    scAnndata = sc.read_visium(os.path.join(stPath, slices[i]))
    savePath = os.path.join(savePath_,slices[i])
    if not (os.path.exists(savePath)):
        os.makedirs(savePath)

    # Preprocessing ST data
    # scAnndata_magic = perform_magic_preprocessing(scAnndata,require_normalization=True)
    # scAnndata = PreprocessingST(scAnndata)
    scAnndata = GetPathoClass(adata = scAnndata, pretrain_path = pretrain_path, n_components = 50, n_clusters = 6, 
    plot_classes = True, image_save_path = os.path.join(savePath,"patho_label.pdf"))

    ## Calculate similarity matrix
    # similarity_df, distance_df = compute_spots_similarity(
    #     input_data=scAnndata, perform_normalization=True)

    similarity_df = compute_spots_similarity(
        input_data=scAnndata, perform_normalization=True,calculate_distance=False)

    # Get gene-pairs matrix
    stExp = pd.DataFrame(scAnndata.X.toarray().T)
    stExp.index = scAnndata.var_names
    stExp.column = scAnndata.obs.index

    GPextractor = GenePairExtractor(
        bulk_expression=bulkExp_train,
        clinical_data=bulkClinical_train,
        single_cell_expression=stExp,
        analysis_mode="Bionomial",
        top_var_genes=2000,
        top_gene_pairs=500,
        p_value_threshold=0.05,
        padj_value_threshold=None,
        max_cutoff=0.8,
        min_cutoff=-0.8
        )

    bulk_gene_pairs_mat, st_gene_pairs_mat = GPextractor.run_extraction()
    bulk_gene_pairs_mat = pd.DataFrame(bulk_gene_pairs_mat.T)
    val_bulkExp_gene_pairs_mat = transform_test_exp(train_exp = bulk_gene_pairs_mat,test_exp = bulkExp_val) ## validation
    st_gene_pairs_mat = pd.DataFrame(st_gene_pairs_mat.T)

    ## Save temp files
    with open(os.path.join(savePath, 'similarity_df.pkl'), 'wb') as f:
        pickle.dump(similarity_df, f) ## similarity matrix
    f.close()

    # with open(os.path.join(savePath, 'distance_df.pkl'), 'wb') as f:
    #     pickle.dump(distance_df, f) ## distance matrix
    # f.close()

    with open(os.path.join(savePath, 'bulk_gene_pairs_mat.pkl'), 'wb') as f:
        pickle.dump(bulk_gene_pairs_mat, f) ## training bulk gene pair matrix
    f.close()
    with open(os.path.join(savePath, 'val_bulkExp_gene_pairs_mat.pkl'), 'wb') as f:
        pickle.dump(val_bulkExp_gene_pairs_mat, f) ## validating gene pair matrix
    f.close()
    with open(os.path.join(savePath, 'st_gene_pairs_mat.pkl'), 'wb') as f:
        pickle.dump(st_gene_pairs_mat, f) ## st gene pair matrix
    f.close()

    # f = open(os.path.join(savePath, 'similarity_df.pkl'), 'rb')
    # similarity_df = pickle.load(f)
    # f.close()

    # f = open(os.path.join(savePath, 'distance_df.pkl'), 'rb')
    # distance_df = pickle.load(f)
    # f.close()

    ## gene pair matrix
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
    mode="Bionomial"
    train_dataset_Bulk = BulkDataset(bulk_gene_pairs_mat, bulkClinical, mode=mode)
    val_dataset_Bulk = BulkDataset(val_bulkExp_gene_pairs_mat, bulkClinical_val, mode=mode)
    train_dataset_ST = STDataset(st_gene_pairs_mat)

    train_loader_Bulk = DataLoader(train_dataset_Bulk, batch_size=1024, shuffle=False)
    val_loader_Bulk = DataLoader(val_dataset_Bulk, batch_size=1024, shuffle=False)
    train_loader_ST = DataLoader(train_dataset_ST, batch_size=1024, shuffle=True)

    ## Training setting
    device = "cuda" if torch.cuda.is_available() else "cpu"
    infer_mode = "Spot"
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
        nlayers=3, n_pred=2, dropout=0.5, mode=mode, encoder_type=encoder_type,

        # Data
        train_loader_Bulk=train_loader_Bulk,
        val_loader_Bulk=val_loader_Bulk,
        train_loader_SC=train_loader_ST,
        adj_A=adj_A, adj_B=None, pre_patho_labels=patholabels,

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
                nhid2=8, n_output=32, nlayers=3, n_pred=2, n_patho=6,dropout=0.5, mode=mode, encoder_type=encoder_type)
    model.load_state_dict(torch.load(filename))
    model = model.to("cpu")

    bulk_gene_pairs_mat_all = transform_test_exp(train_exp = bulk_gene_pairs_mat,test_exp = bulkExp)
    bulk_PredDF, sc_PredDF = Predict(model,
                                    bulk_GPmat=bulk_gene_pairs_mat_all, sc_GPmat=st_gene_pairs_mat,
                                    mode="Bionomial",
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