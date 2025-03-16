import os, pickle, torch, optuna
import pandas as pd
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from TiRank.Dataloader import transform_test_exp
from TiRank.TrainPre import Reject_With_GMM_Bio,Reject_With_GMM_Reg,Reject_With_StrictNumber,get_best_model,Train_one_epoch,Validate_model
from TiRank.Loss import *
from TiRank.Model import TiRankModel


def save_data_(self):
    print(f"Starting save gene pair matrices.")
    savePath_2 = os.path.join(self.savePath, "2_preprocessing")
    savePath_splitData = os.path.join(savePath_2, "split_data")

    ## Load val bulk
    f = open(os.path.join(savePath_splitData, "bulkExp_val.pkl"), "rb")
    bulkExp_val = pickle.load(f)
    f.close()

    train_bulk_gene_pairs_mat = pd.DataFrame(self.bulk_gene_pairs_mat.T)
    val_bulkExp_gene_pairs_mat = transform_test_exp(
        train_exp=train_bulk_gene_pairs_mat, test_exp=bulkExp_val
    )
    sc_gene_pairs_mat = pd.DataFrame(self.single_cell_gene_pairs_mat.T)

    with open(os.path.join(savePath_2, "train_bulk_gene_pairs_mat.pkl"), "wb") as f:
        pickle.dump(train_bulk_gene_pairs_mat, f)  ## training bulk gene pair matrix
    f.close()

    with open(
        os.path.join(savePath_2, "val_bulkExp_gene_pairs_mat.pkl"), "wb"
    ) as f:
        pickle.dump(
            val_bulkExp_gene_pairs_mat, f
        )  ## validating bulk gene pair matrix
    f.close()

    with open(os.path.join(savePath_2, "sc_gene_pairs_mat.pkl"), "wb") as f:
        pickle.dump(sc_gene_pairs_mat, f)  ## single cell gene pair matrix
    f.close()
    print(f"Save gene pair matrices done.")


    ## Save informative gene matrix
    ### SC
    exp = self.single_cell_expression.loc[self.gene_list]
    sc_infoGene_df = pd.DataFrame(exp, index=self.gene_list, columns=self.single_cell_expression.columns)

    ### Bulk
    bulkTrain_infoGene_df = self.bulk_expression.loc[self.gene_list, :] ## Train
    bulkVal_infoGene_df = bulkExp_val.loc[self.gene_list, :] ## Val

    with open(os.path.join(savePath_2, "train_bulk_informative_genes_mat.pkl"), "wb") as f:
        pickle.dump(bulkTrain_infoGene_df, f)  ## training bulk gene pair matrix
    f.close()

    with open(
        os.path.join(savePath_2, "val_bulkExp_informative_genes_mat.pkl"), "wb"
    ) as f:
        pickle.dump(
            bulkVal_infoGene_df, f
        )  ## validating bulk gene pair matrix
    f.close()

    with open(os.path.join(savePath_2, "sc_informative_genes_mat.pkl"), "wb") as f:
        pickle.dump(sc_infoGene_df, f)  ## single cell gene pair matrix
    f.close()
    print(f"Save informative gene matrices done.")

    return None

def run_extraction_(self):
    print(f"Starting gene pair extraction.")

    # Find the intersection of genes in bulk and single-cell datasets
    intersect_genes = np.intersect1d(
        self.single_cell_expression.index, self.bulk_expression.index
    )
    intersect_single_cell_expression = self.single_cell_expression.loc[intersect_genes]

    # Sort genes by variance in the single-cell dataset
    gene_variances = np.var(intersect_single_cell_expression, axis=1)
    sorted_genes = gene_variances.sort_values(ascending=False)

    # Select the top variable genes
    top_variable_genes = sorted_genes[: self.top_var_genes].index.tolist()

    # Extract the candidate genes
    self.bulk_expression, self.single_cell_expression = (
        self.extract_candidate_genes(top_variable_genes)
    )

    print(f"Get candidate genes done.")

    # Obtain the list of candidate genes
    if self.analysis_mode == "Classification":
        regulated_genes_r, regulated_genes_p = self.calculate_binomial_gene_pairs()
        print(
            f"There are {len(regulated_genes_r)} genes up-regulated in Group 0 and {len(regulated_genes_p)} genes up-regulated in Group 1."
        )

    elif self.analysis_mode == "Cox":
        regulated_genes_r, regulated_genes_p = self.calculate_survival_gene_pairs()
        print(
            f"There are {len(regulated_genes_r)} Risk genes and {len(regulated_genes_p)} Protective genes."
        )

    elif self.analysis_mode == "Regression":
        regulated_genes_r, regulated_genes_p = (
            self.calculate_regression_gene_pairs()
        )
        print(
            f"There are {len(regulated_genes_r)} positive-associated genes and {len(regulated_genes_p)} negative-associated genes."
        )

    else:
        raise ValueError(f"Unsupported mode: {self.analysis_mode}")

    if (len(regulated_genes_r) == 0) or (len(regulated_genes_p) == 0):
        raise ValueError(
            "A set of genes is empty. Try increasing the 'top_var_genes' value or loosening the 'p.value' threshold."
        )

    print(f"Get candidate gene pairs done.")

    # Save genes
    self.gene_list = regulated_genes_r.copy()
    self.gene_list.extend(regulated_genes_p)

    # Transform the bulk gene pairs
    bulk_gene_pairs = self.transform_bulk_gene_pairs(
        regulated_genes_r, regulated_genes_p
    )

    # Filter the gene pairs
    bulk_gene_pairs_mat = self.filter_gene_pairs(bulk_gene_pairs)

    # Transform the single-cell gene pairs
    single_cell_gene_pairs_mat = self.transform_single_cell_gene_pairs(
        bulk_gene_pairs_mat
    )

    print(f"Profile transformation done.")

    # Return the bulk and single-cell gene pairs
    self.bulk_gene_pairs_mat = bulk_gene_pairs_mat
    self.single_cell_gene_pairs_mat = single_cell_gene_pairs_mat

    return None

def Predict_(savePath, mode, do_reject=True, tolerance=0.05, reject_mode="GMM", suffix=""):
    savePath_1 = os.path.join(savePath, "1_loaddata")
    savePath_2 = os.path.join(savePath, "2_preprocessing")
    savePath_3 = os.path.join(savePath, "3_Analysis")

    model = get_best_model(savePath)

    print("Starting Inference.")

    # Load data
    ### Training bulk set
    f = open(os.path.join(savePath_2, "train_bulk_gene_pairs_mat.pkl"), "rb")
    train_bulk_gene_pairs_mat = pickle.load(f)
    f.close()

    ### All bulk
    f = open(os.path.join(savePath_1, "bulk_exp.pkl"), "rb")
    bulkExp = pickle.load(f)
    f.close()

    f = open(os.path.join(savePath_1, "bulk_clinical.pkl"), "rb")
    bulkClinical = pickle.load(f)
    f.close()

    bulk_rownames = bulkClinical.index.tolist()

    ### Transfer all bulk into gene pairs
    bulk_GPmat = transform_test_exp(
        train_exp=train_bulk_gene_pairs_mat, test_exp=bulkExp
    )

    ### SC gene pair matrix
    f = open(os.path.join(savePath_2, "sc_gene_pairs_mat.pkl"), "rb")
    sc_GPmat = pickle.load(f)
    f.close()

    f = open(os.path.join(savePath_2, "scAnndata.pkl"), "rb")
    scAnndata = pickle.load(f)
    f.close()

    sc_rownames = scAnndata.obs.index.tolist()

    model.eval()

    # Predict on cell
    Exp_Tensor_sc = torch.from_numpy(np.array(sc_GPmat))
    Exp_Tensor_sc = torch.tensor(Exp_Tensor_sc, dtype=torch.float32)
    embeddings_sc, pred_sc, _ = model(Exp_Tensor_sc)

    # Predict on bulk
    Exp_Tensor_bulk = torch.from_numpy(np.array(bulk_GPmat))
    Exp_Tensor_bulk = torch.tensor(Exp_Tensor_bulk, dtype=torch.float32)
    embeddings_bulk, pred_bulk, _ = model(Exp_Tensor_bulk)

    if mode == "Cox":
        pred_bulk = pred_bulk.detach().numpy().reshape(-1, 1)
        pred_sc = pred_sc.detach().numpy().reshape(-1, 1)

    elif mode == "Classification":
        pred_sc = pred_sc[:, 1].detach().numpy().reshape(-1, 1)
        pred_bulk = pred_bulk[:, 1].detach().numpy().reshape(-1, 1)

    elif mode == "Regression":
        pred_sc = pred_sc.detach().numpy().reshape(-1, 1)
        pred_bulk = pred_bulk.detach().numpy().reshape(-1, 1)

    embeddings_sc = embeddings_sc.detach().numpy()
    embeddings_bulk = embeddings_bulk.detach().numpy()

    if do_reject:
        if reject_mode == "GMM":
            if mode in ["Cox", "Classification"]:
                reject_mask = Reject_With_GMM_Bio(
                    pred_bulk,
                    pred_sc,
                    tolerance=tolerance,
                    min_components=3,
                    max_components=15,
                )
            if mode == "Regression":
                reject_mask = Reject_With_GMM_Reg(
                    pred_bulk, pred_sc, tolerance=tolerance
                )

        elif reject_mode == "Strict":
            if mode in ["Cox", "Classification"]:
                reject_mask = Reject_With_StrictNumber(
                    pred_bulk, pred_sc, tolerance=tolerance
                )

            if mode == "Regression":
                print("Test")
        else:
            raise ValueError(f"Unsupported Rejcetion Mode: {reject_mode}")

    else:
        reject_mask = np.zeros_like(pred_sc)

    saveDF_sc = pd.DataFrame(
        data=np.concatenate((reject_mask, pred_sc, embeddings_sc), axis=1),
        index=sc_GPmat.index,
    )

    colnames = ["Reject", "Pred_score"]
    colnames.extend(["embedding_" + str(i + 1) for i in range(embeddings_sc.shape[1])])

    saveDF_sc.columns = colnames
    saveDF_sc.index = sc_rownames

    reject_mask_ = np.zeros_like(pred_bulk)
    saveDF_bulk = pd.DataFrame(
        data=np.concatenate((reject_mask_, pred_bulk, embeddings_bulk), axis=1),
        index=bulk_GPmat.index,
    )

    saveDF_bulk.columns = colnames
    saveDF_bulk.index = bulk_rownames

    print("Inference Done.")

    with open(os.path.join(savePath_3, "saveDF_bulk_"+suffix+".pkl"), "wb") as f:
        pickle.dump(saveDF_bulk, f)
    f.close()
    with open(os.path.join(savePath_3, "saveDF_sc_"+suffix+".pkl"), "wb") as f:
        pickle.dump(saveDF_sc, f)
    f.close()

    ## Original categorize function
    if saveDF_sc.shape[0] != scAnndata.obs.shape[0]:
        raise ValueError("The prediction matrix was not match with original scAnndata.")

    scAnndata.obsm["Rank_Embedding"] = saveDF_sc.iloc[:, 2:]
    scAnndata.obs["Reject"] = saveDF_sc.iloc[:, 0]
    scAnndata.obs["Rank_Score"] = saveDF_sc.iloc[:, 1]

    if mode in ["Cox", "Classification"]:
        temp = scAnndata.obs["Rank_Score"] * (1 - scAnndata.obs["Reject"])
        scAnndata.obs["Rank_Label"] = [
            "Background" if i == 0 else "Rank-" if 0 <= i < 0.5 else "Rank+"
            for i in temp
        ]

        print(f"We set Rank score < 0.5 as Rank- () while > 0.5 as Rank+ ")

    if mode == "Regression":
        scAnndata.obs["Rank_Label"] = scAnndata.obs["Rank_Score"] * (
            1 - scAnndata.obs["Reject"]
        )

    ## Save
    sc_pred_df = scAnndata.obs
    sc_pred_df.to_csv(os.path.join(savePath_3, "spot_predict_score_"+suffix+".csv"))
    scAnndata.write_h5ad(filename=os.path.join(savePath_3, "final_anndata_"+suffix+".h5ad"))

    return None

def ablation_(
    trial,
    n_features,
    nhead,
    nhid1,
    nhid2,
    n_output,
    nlayers,
    n_pred,
    dropout,
    n_patho,
    mode,
    encoder_type,
    train_loader_Bulk,
    val_loader_Bulk,
    train_loader_SC,
    adj_A,
    adj_B,
    pre_patho_labels,
    device,
    infer_mode,
    model_save_path,
    ablation_index
):

    print(
        f"Initiate model in trail {trial.number} with mode: {mode}, infer mode: {infer_mode} encoder type: {encoder_type} on device: {device} with abation on {ablation_index}."
    )
    model = TiRankModel(
        n_features=n_features,
        nhead=nhead,
        nhid1=nhid1,
        nhid2=nhid2,
        n_output=n_output,
        nlayers=nlayers,
        n_pred=n_pred,
        dropout=dropout,
        n_patho=n_patho,
        mode=mode,
        encoder_type=encoder_type,
    )
    model = model.to(device)

    # Define hyperparameters with trial object
    lr_choices = [1e-2, 6e-3, 3e-3, 1e-3, 6e-4, 3e-4, 1e-4, 1e-5, 1e-6]
    lr = trial.suggest_categorical("lr", lr_choices)

    n_epochs_choices = [300, 350, 400, 450, 500]
    n_epochs = trial.suggest_categorical("n_epochs", n_epochs_choices)

    # Define alpha values as specific choices
    alpha_0_choices = [0.1, 0.06, 0.03, 0.01, 0.005]
    alpha_1_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    alpha_2_choices = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    alpha_3_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    if ablation_index == "1":
        alpha_1_choices = [0]
    if ablation_index == "2":
        alpha_2_choices = [0]
    if ablation_index == "1,2":
        alpha_1_choices = [0]
        alpha_2_choices = [0]

    # Suggest categorical choices for each alpha
    alphas = [
        trial.suggest_categorical("alpha_0", alpha_0_choices),
        trial.suggest_categorical("alpha_1", alpha_1_choices),
        trial.suggest_categorical("alpha_2", alpha_2_choices),
        trial.suggest_categorical("alpha_3", alpha_3_choices),
    ]

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
    train_loss_dcit = dict()

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss_dcit_epoch = Train_one_epoch(
            model=model,
            dataloader_A=train_loader_Bulk,
            dataloader_B=train_loader_SC,
            mode=mode,
            infer_mode=infer_mode,
            adj_A=adj_A,
            adj_B=adj_B,
            pre_patho_labels=pre_patho_labels,
            optimizer=optimizer,
            alphas=alphas,
            device=device,
        )

        train_loss_dcit["Epoch_" + str(epoch + 1)] = train_loss_dcit_epoch

        scheduler.step()

        # Val
        model.eval()
        val_loss = Validate_model(
            model=model,
            dataloader_A=val_loader_Bulk,
            dataloader_B=train_loader_SC,
            mode=mode,
            infer_mode=infer_mode,
            adj_A=adj_A,
            adj_B=adj_B,
            pre_patho_labels=pre_patho_labels,
            alphas=[1, 0.1, 1, 0],
            device=device,
        )

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Generate the filename including hyperparameters
    hyperparams_str = f"lr_{lr}_epochs_{n_epochs}_alpha0_{alphas[0]}_alpha1_{alphas[1]}_alpha2_{alphas[2]}_alpha3_{alphas[3]}"
    model_filename = f"model_{hyperparams_str}.pt"

    # Saving the model and plot with new filenames
    if trial.number == 0 or (
        trial.study.best_trials and val_loss < trial.study.best_value
    ):
        print(f"Saving model and plot for Trial {trial.number} with Validation Loss = {val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(model_save_path, model_filename))

    return val_loss

def tune_hyperparameters_withAb(
    savePath,  # Dictionary containing all model parameters
    device="cpu",  # Computing device ('cpu' or 'cuda')
    n_trials=50,  # Number of optimization trials
    ablation_index=None
):

    savePath_3 = os.path.join(savePath, "3_Analysis")
    savePath_data2train = os.path.join(savePath_3, "data2train")

    # Load dataloaders
    f = open(os.path.join(savePath_data2train, "train_loader_Bulk.pkl"), "rb")
    train_loader_Bulk = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_data2train, "val_loader_Bulk.pkl"), "rb")
    val_loader_Bulk = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_data2train, "train_loader_SC.pkl"), "rb")
    train_loader_SC = pickle.load(f)
    f.close()

    f = open(os.path.join(savePath_data2train, "adj_A.pkl"), "rb")
    adj_A = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_data2train, "adj_B.pkl"), "rb")
    adj_B = pickle.load(f)
    f.close()
    f = open(os.path.join(savePath_data2train, "patholabels.pkl"), "rb")
    pre_patho_labels = pickle.load(f)
    f.close()

    # Initial model parameters
    f = open(os.path.join(savePath_3, "model_para.pkl"), "rb")
    model_para = pickle.load(f)
    f.close()

    # Extract parameters from the model_para dictionary
    n_features = model_para.get("n_features", None)
    nhead = model_para.get("nhead", 8)
    nhid1 = model_para.get("nhid1", 256)
    nhid2 = model_para.get("nhid2", 128)
    n_output = model_para.get("n_output", 10)
    nlayers = model_para.get("nlayers", 2)
    n_pred = model_para.get("n_pred", 1)
    dropout = model_para.get("dropout", 0.5)
    n_patho = model_para.get("n_patho", 0)
    mode = model_para.get("mode", "Cox")
    infer_mode = model_para.get("infer_mode", "Cell")
    encoder_type = model_para.get("encoder_type", "MLP")
    model_save_path = model_para.get("model_save_path", "./checkpoints/")

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: ablation_(
            trial,
            n_features,
            nhead,
            nhid1,
            nhid2,
            n_output,
            nlayers,
            n_pred,
            dropout,
            n_patho,
            mode,
            encoder_type,
            train_loader_Bulk,
            val_loader_Bulk,
            train_loader_SC,
            adj_A,
            adj_B,
            pre_patho_labels,
            device,
            infer_mode,
            model_save_path,
            ablation_index
        ),
        n_trials=n_trials,
    )

    # save the best hyperparameters
    best_params = study.best_trial.params
    with open(os.path.join(savePath_3, "best_params.pkl"), "wb") as f:
        print("Best hyperparameters:", best_params)
        pickle.dump(best_params, f)  ## bet parameters set
    f.close()

    return None