import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import os
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.mixture import GaussianMixture
import scipy.stats as stats

import optuna
from concurrent.futures import ThreadPoolExecutor

from Loss import *
from Model import *

# Training


def Train_one_epoch(model, dataloader_A, dataloader_B, pheno='Cox', infer_mode="Cell", adj_A=None, adj_B=None, pre_patho_labels=None, optimizer=None, alphas=[1, 1, 1, 1], device="cpu"):

    model.train()

    running_loss = 0.0

    # RNA-seq data whole batch training
    iter_A = iter(dataloader_A)

    if pheno == 'Cox':
        (X_a, t, e) = next(iter_A)
        X_a = X_a.to(device)
        t = t.to(device)
        e = e.to(device)

    if pheno in ['Bionomial', 'Regression']:
        (X_a, label) = next(iter_A)
        X_a = X_a.to(device)
        label = label.to(device)
        

    for batch_B in dataloader_B:
        # Get the next batch of data
        (X_b, idx) = batch_B

        # Move the data to the GPU
        X_b = X_b.to(device)

        if adj_A is not None:
            A = adj_A[idx, :][:, idx]
            A = A.to(device)

        if adj_B is not None:
            B = adj_B[idx, :][:, idx]
            B = B.to(device)

        if pre_patho_labels is not None:
            # Convert the tensor idx to numpy for indexing pandas series
            idx_np = idx.cpu().numpy()
            pre_patho = pre_patho_labels.iloc[idx_np].values
            pre_patho = torch.tensor(pre_patho, dtype=torch.uint8)  # Specify dtype if necessary
            pre_patho = pre_patho.to(device)
            
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        embeddings_a, risk_scores_a, _ = model(X_a)
        embeddings_b, _, pred_patho = model(X_b)

        regularization_loss_ =  0.1 * regularization_loss(model.feature_weights)

        # Calculate loss
        if pheno == 'Cox':
            bulk_loss_ = cox_loss(risk_scores_a, t, e)

        elif pheno == 'Bionomial':
            bulk_loss_ = CrossEntropy_loss(risk_scores_a, label)

        elif pheno == 'Regression':
            bulk_loss_ = MSE_loss(risk_scores_a, label)

        if infer_mode == 'Cell':
            cosine_loss_ = cosine_loss(embeddings_b, A)
            mmd_loss_ = mmd_loss(embeddings_a, embeddings_b)

            # total loss

            total_loss = regularization_loss_+ bulk_loss_ * alphas[0] + \
                    cosine_loss_ * alphas[1] + \
                    mmd_loss_ * alphas[2]

        elif infer_mode == 'Spot' and adj_B is not None:
            cosine_loss_exp_ = cosine_loss(embeddings_b, A)
            cosine_loss_spatial_ = cosine_loss(embeddings_b, B)

            mmd_loss_ = mmd_loss(embeddings_a, embeddings_b)

            # total loss

            total_loss = regularization_loss_ + bulk_loss_ * alphas[0] + \
                cosine_loss_exp_ * alphas[1] + \
                cosine_loss_spatial_ * alphas[2] + \
                mmd_loss_ * alphas[3]

        elif infer_mode == 'Spot' and pre_patho is not None:
            cosine_loss_exp_ = cosine_loss(embeddings_b, A)
            pathoLloss = CrossEntropy_loss(pred_patho, pre_patho)

            mmd_loss_ = mmd_loss(embeddings_a, embeddings_b)

            # total loss

            total_loss = regularization_loss_ + bulk_loss_ * alphas[0] + \
                cosine_loss_exp_ * alphas[1] + \
                pathoLloss * alphas[2] + \
                mmd_loss_ * alphas[3]

        else:
            raise ValueError(f"Unsupported mode: {infer_mode}. There are two mode: Cell and Spot.")

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(dataloader_B)

# Validate


def Validate_model(model, dataloader_A, dataloader_B, pheno='Cox', infer_mode="Cell", adj_A=None, adj_B=None, pre_patho_labels=None, alphas=[1, 1, 1, 1], device="cpu"):
    
    model.eval()  # Set the model to evaluation mode

    validation_loss = 0.0
    with torch.no_grad():  # No need to track the gradients
        # RNA-seq data whole batch validation
        iter_A = iter(dataloader_A)

        if pheno == 'Cox':
            (X_a, t, e) = next(iter_A)
            X_a = X_a.to(device)
            t = t.to(device)
            e = e.to(device)

        if pheno in ['Bionomial', 'Regression']:
            (X_a, label) = next(iter_A)
            X_a = X_a.to(device)
            label = label.to(device)
        
        for batch_B in dataloader_B:
            # Similar setup as in the training function
            (X_b, idx) = batch_B
            X_b = X_b.to(device)

            if adj_A is not None:
                A = adj_A[idx, :][:, idx]
                A = A.to(device)

            if adj_B is not None:
                B = adj_B[idx, :][:, idx]
                B = B.to(device)

            if pre_patho_labels is not None:
                idx_np = idx.cpu().numpy()
                pre_patho = pre_patho_labels.iloc[idx_np].values
                pre_patho = torch.tensor(pre_patho, dtype=torch.uint8)
                pre_patho = pre_patho.to(device)
            
            # Forward pass only
            embeddings_a, risk_scores_a, _ = model(X_a)
            embeddings_b, _, pred_patho = model(X_b)

            # Compute loss as in training function but without backward and optimizer steps
            # The computation here assumes that the loss functions and infer_mode conditions are the same as in training
            # Please adapt if your validation conditions differ from the training

            # Calculate loss
            if pheno == 'Cox':
                bulk_loss_ = cox_loss(risk_scores_a, t, e)

            elif pheno == 'Bionomial':
                bulk_loss_ = CrossEntropy_loss(risk_scores_a, label)

            elif pheno == 'Regression':
                bulk_loss_ = MSE_loss(risk_scores_a, label)

            if infer_mode == 'Cell':
                cosine_loss_ = cosine_loss(embeddings_b, A)
                mmd_loss_ = mmd_loss(embeddings_a, embeddings_b)

                # total loss

                total_loss = bulk_loss_ * alphas[0] + \
                    cosine_loss_ * alphas[1] + \
                    mmd_loss_ * alphas[2]

            elif infer_mode == 'Spot' and adj_B is not None:
                cosine_loss_exp_ = cosine_loss(embeddings_b, A)
                cosine_loss_spatial_ = cosine_loss(embeddings_b, B)

                mmd_loss_ = mmd_loss(embeddings_a, embeddings_b)

                # total loss

                total_loss = bulk_loss_ * alphas[0] + \
                    cosine_loss_exp_ * alphas[1] + \
                    cosine_loss_spatial_ * alphas[2] + \
                    mmd_loss_ * alphas[3]

            elif infer_mode == 'Spot' and pre_patho is not None:
                cosine_loss_exp_ = cosine_loss(embeddings_b, A)
                pathoLloss = CrossEntropy_loss(pred_patho, pre_patho)

                mmd_loss_ = mmd_loss(embeddings_a, embeddings_b)

                # total loss

                total_loss = bulk_loss_ * alphas[0] + \
                    cosine_loss_exp_ * alphas[1] + \
                    pathoLloss * alphas[2] + \
                    mmd_loss_ * alphas[3]

            else:
                raise ValueError(f"Unsupported mode: {infer_mode}. There are two mode: Cell and Spot.")
            
            # Add up the total loss for the validation set
            validation_loss += total_loss.item()

    # Calculate the average loss over all batches in the dataloader
    return validation_loss / len(dataloader_B)

# Predict


def Predict(model, bulk_GPmat, sc_GPmat, mode, bulk_rownames = None, sc_rownames = None, do_reject=True, tolerance=0.05, reject_mode = "GMM"):
    model.eval()

    # Predict on cell
    Exp_Tensor_sc = torch.from_numpy(np.array(sc_GPmat))
    Exp_Tensor_sc = torch.tensor(Exp_Tensor_sc, dtype=torch.float32)
    embeddings_sc, pred_sc, _  = model(Exp_Tensor_sc)

    # Predict on bulk
    Exp_Tensor_bulk = torch.from_numpy(np.array(bulk_GPmat))
    Exp_Tensor_bulk = torch.tensor(Exp_Tensor_bulk, dtype=torch.float32)
    embeddings_bulk, pred_bulk, _ = model(Exp_Tensor_bulk)

    if mode == "Cox":
        pred_bulk = pred_bulk.detach().numpy().reshape(-1, 1)
        pred_sc = pred_sc.detach().numpy().reshape(-1, 1)

    elif mode == "Bionomial":
        pred_sc = torch.nn.functional.softmax(
            pred_sc)[:, 1].detach().numpy().reshape(-1, 1)

        pred_bulk = torch.nn.functional.softmax(
            pred_bulk)[:, 1].detach().numpy().reshape(-1, 1)

    elif mode == "Regression":
        pred_sc = pred_sc.detach().numpy().reshape(-1, 1)
        pred_bulk = pred_bulk.detach().numpy().reshape(-1, 1)

    embeddings_sc = embeddings_sc.detach().numpy()
    embeddings_bulk = embeddings_bulk.detach().numpy()

    if do_reject:
        if reject_mode == "GMM":
            if mode in ["Cox","Bionomial"]:
                reject_mask = Reject_With_GMM_Bio(pred_bulk, pred_sc,
                                    tolerance=tolerance, min_components=3, max_components=15)
            if mode == "Regression":
                reject_mask = Reject_With_GMM_Reg(pred_bulk, pred_sc,tolerance=tolerance)

        elif reject_mode == "Strict":
            if mode in ["Cox","Bionomial"]:
                reject_mask = Reject_With_StrictNumber(pred_bulk, pred_sc,tolerance=tolerance)          

            if mode == "Regression":
                print("Test")
        else:
            raise ValueError(f"Unsupported Rejcetion Mode: {reject_mode}")
    
    else:
        reject_mask = np.zeros_like()

    saveDF_sc = pd.DataFrame(data=np.concatenate(
        (reject_mask, pred_sc, embeddings_sc), axis=1), index=sc_GPmat.index)

    colnames = ["Reject", "Pred_score"]
    colnames.extend(["embedding_" + str(i + 1)
                    for i in range(embeddings_sc.shape[1])])

    saveDF_sc.columns = colnames
    saveDF_sc.index = sc_rownames

    reject_mask_ = np.zeros_like(pred_bulk)
    saveDF_bulk = pd.DataFrame(data=np.concatenate(
        (reject_mask_, pred_bulk, embeddings_bulk), axis=1), index=bulk_GPmat.index)

    saveDF_bulk.columns = colnames
    saveDF_bulk.index = bulk_rownames

    return saveDF_bulk, saveDF_sc

# Reject


def Reject_With_GMM_Bio(pred_bulk, pred_sc, tolerance, min_components, max_components):
    print(f"Perform Rejection with GMM mode with tolerance={tolerance}, components=[{min_components},{max_components}]!")

    gmm_bulk = GaussianMixture(n_components=2, random_state=619).fit(pred_bulk)

    gmm_bulk_mean_1 = np.max(gmm_bulk.means_)
    gmm_bulk_mean_0 = np.min(gmm_bulk.means_)

    if (gmm_bulk_mean_1 - gmm_bulk_mean_0) <= 0.5:
        print("Underfitting!")

    # Iterate over the number of components
    for n_components in range(min_components, max_components + 1):
        gmm_sc = GaussianMixture(
            n_components=n_components, random_state=619).fit(pred_sc)

        means = gmm_sc.means_

        # Check if any of the means are close to 0 or 1
        zero_close = any(abs(mean - gmm_bulk_mean_0) <=
                         tolerance for mean in means)
        one_close = any(abs(gmm_bulk_mean_1 - mean) <=
                        tolerance for mean in means)

        if zero_close and one_close:
            print(
                f"Found distributions with means close to 0 and 1 with {n_components} components.")

            # # Print the means and covariances
            # print("Means of the gaussians in gmm_sc: ", gmm_sc.means_)
            # print("Covariances of the gaussians in gmm_sc: ", gmm_sc.covariances_)

            # Find the component whose mean is nearest to 0 and 1
            # 1
            diffs_1 = abs(gmm_bulk_mean_1 - gmm_sc.means_) 
            nearest_component_1 = np.where(diffs_1 <= tolerance)[0]

            # 0
            diffs_0 = abs(gmm_sc.means_ - gmm_bulk_mean_0)
            nearest_component_0 = np.where(diffs_0 <= tolerance)[0]

            # concat
            remain_component = np.concatenate(
                (nearest_component_1, nearest_component_0))

            # The mask of those rejection cell
            labels_sc = gmm_sc.predict(pred_sc)

            mask = np.ones(shape=(len(labels_sc), 1))

            mask[np.isin(labels_sc, remain_component)] = 0

            print(
                f"Reject {int(sum(mask))}({int(sum(mask))*100 / len(mask) :.2f}%) cells.")

            return mask

    print(f"Two distribution rejection faild.")

    print(f"Perform single distribution rejection.")
    for n_components in range(2, max_components + 1):
        gmm_sc = GaussianMixture(
            n_components=n_components, random_state=619).fit(pred_sc)

        means = gmm_sc.means_

        # Check if any of the means are close to 0 or 1
        zero_close = any(abs(mean - gmm_bulk_mean_0) <=
                         tolerance for mean in means)
        one_close = any(abs(gmm_bulk_mean_1 - mean) <=
                        tolerance for mean in means)

        if zero_close or one_close:
            if zero_close:
                print(
                    f"Found distributions with means close to 0 with {n_components} components.")
            if one_close:
                print(
                    f"Found distributions with means close to 1 with {n_components} components.")

            # # Print the means and covariances
            # print("Means of the gaussians in gmm_sc: ", gmm_sc.means_)
            # print("Covariances of the gaussians in gmm_sc: ", gmm_sc.covariances_)

            # Find the component whose mean is nearest to 0 and 1
            # 1
            diffs_1 = gmm_bulk_mean_1 - gmm_sc.means_
            nearest_component_1 = np.where(diffs_1 <= tolerance)[0]

            # 0
            diffs_0 = gmm_sc.means_ - gmm_bulk_mean_0
            nearest_component_0 = np.where(diffs_0 <= tolerance)[0]

            # concat
            remain_component = np.concatenate(
                (nearest_component_1, nearest_component_0))

            # The mask of those rejection cell
            labels_sc = gmm_sc.predict(pred_sc)

            mask = np.ones(shape=(len(labels_sc), 1))

            mask[np.isin(labels_sc, remain_component)] = 0

            print(
                f"Reject {int(sum(mask))}({int(sum(mask))*100 / len(mask) :.2f}%) cells.")

            return mask

    print(f"Single distribution rejection faild.")
    mask = np.zeros(shape=(len(pred_sc), 1))

    return mask

def Reject_With_GMM_Reg(pred_bulk, pred_sc, tolerance):
    print(f"Perform Rejection with GMM mode with tolerance={tolerance}!")

    gmm_bulk = GaussianMixture(n_components=1, random_state=619).fit(pred_bulk)
    gmm_sc = GaussianMixture(n_components=1, random_state=619).fit(pred_sc)

    ## Mean
    gmm_bulk_mean = gmm_bulk.means_[0][0]
    gmm_sc_mean = gmm_sc.means_[0][0]

    ## Std
    gmm_bulk_variance = gmm_bulk.covariances_[0][0]
    gmm_bulk_std = np.sqrt(gmm_bulk_variance)

    if (abs(gmm_sc_mean - gmm_bulk_mean)) >= 0.5:
        print("Integration was failed !")
        mask = np.ones(shape=(len(pred_sc), 1))

    else:
        if tolerance > gmm_bulk_std:
            tolerance = gmm_bulk_std
        lower_bound = gmm_bulk_mean - tolerance
        upper_bound = gmm_bulk_mean + tolerance

        mask = np.ones(shape=(len(pred_sc), 1))
        mask[(pred_sc >= lower_bound) & (pred_sc <= upper_bound)] = 0

    print(f"Reject {int(sum(mask))}({int(sum(mask))*100 / len(mask) :.2f}%) cells.")

    return mask

def Reject_With_StrictNumber(pred_bulk, pred_sc, tolerance):
    print(f"Perform Rejection with Strict Number mode with tolerance={tolerance}!")

    gmm_bulk = GaussianMixture(n_components=2, random_state=619).fit(pred_bulk)

    ## Get mean
    gmm_bulk_means = gmm_bulk.means_.flatten()
    gmm_bulk_mean_1 = np.max(gmm_bulk_means)
    gmm_bulk_mean_0 = np.min(gmm_bulk_means)

    ## Get std
    gmm_bulk_stds = np.sqrt(gmm_bulk.covariances_)
    gmm_bulk_std_1 = gmm_bulk_stds[0][0]
    gmm_bulk_std_0 = gmm_bulk_stds[1][0]

    if (gmm_bulk_mean_1 - gmm_bulk_mean_0) <= 0.8:
        print("Underfitting!")

    # Calculate the percentile range for the tolerance
    lower_percentile = 0.5 - tolerance / 2
    upper_percentile = 0.5 + tolerance / 2

    # Calculate the range for the first Gaussian distribution
    range_low_1 = stats.norm.ppf(lower_percentile, gmm_bulk_mean_1, gmm_bulk_std_1)
    range_high_1 = stats.norm.ppf(upper_percentile, gmm_bulk_mean_1, gmm_bulk_std_1)

    # Calculate the range for the second Gaussian distribution
    range_low_0 = stats.norm.ppf(lower_percentile, gmm_bulk_mean_0, gmm_bulk_std_0)
    range_high_0 = stats.norm.ppf(upper_percentile, gmm_bulk_mean_0, gmm_bulk_std_0)

    print(f"For the first Gaussian distribution with mean {gmm_bulk_mean_1} and std {gmm_bulk_std_1}:")
    print(f"The range around the mean that contains {tolerance*100}% of the samples is approximately from {range_low_1} to {range_high_1}")

    print(f"For the second Gaussian distribution with mean {gmm_bulk_mean_0} and std {gmm_bulk_std_0}:")
    print(f"The range around the mean that contains {tolerance*100}% of the samples is approximately from {range_low_0} to {range_high_0}")

    mask = np.ones(shape=(len(pred_sc), 1))

    # Set mask to zero where the condition for the first Gaussian distribution is met
    mask[(pred_sc >= range_low_1) & (pred_sc <= range_high_1)] = 0

    # Set mask to zero where the condition for the second Gaussian distribution is met
    mask[(pred_sc >= range_low_0) & (pred_sc <= range_high_0)] = 0

    print(f"Reject {int(sum(mask))}({int(sum(mask))*100 / len(mask) :.2f}%) cells.")

    return mask

# categorize
def categorize(scAnndata, sc_PredDF, do_cluster=False, mode = None):
    if sc_PredDF.shape[0] != scAnndata.obs.shape[0]:
        raise ValueError(
            "The prediction matrix was not match with original scAnndata.")

    else:
        if do_cluster:
            sc.tl.umap(scAnndata)
            sc.tl.leiden(scAnndata, key_added="clusters")

        scAnndata.obsm["Rank_Embedding"] = sc_PredDF.iloc[:, 2:]
        scAnndata.obs["Reject"] = sc_PredDF.iloc[:, 0]
        scAnndata.obs["Rank_Score"] = sc_PredDF.iloc[:, 1]

        if mode in ["Cox","Bionomial"]:
            temp = scAnndata.obs["Rank_Score"] * (1 - scAnndata.obs["Reject"])
            scAnndata.obs["Rank_Label"] = [
                "Background" if i == 0 else
                "Rank-" if 0 < i <= 0.5 else
                "Rank+"
                for i in temp
            ]

            print(f"We set Rank score <= 0.5 as Rank- () while > 0.5 as Rank+ ")

        if mode == "Regression":
            scAnndata.obs["Rank_Label"] = scAnndata.obs["Rank_Score"] * (1 - scAnndata.obs["Reject"])

    return scAnndata

# permutation test to determine the phenotype associated cluster

def permute_once(Rank_Labels, Labels, unique_labels):
    shuffled_rank_labels = np.random.permutation(Rank_Labels)
    local_counts = {label: {"Background": 0, "Rank+": 0, "Rank-": 0} for label in unique_labels}
    for label in unique_labels:
        indices = [i for i, x in enumerate(Labels) if x == label]
        subset = [shuffled_rank_labels[i] for i in indices]
        local_counts[label] = dict((x, subset.count(x)) for x in {"Background", "Rank+", "Rank-"})
    return local_counts


def Pcluster(scAnndata, clusterColName, perm_n = 1001):
# Check if the clusterColName is in the observation matrix
    if clusterColName not in scAnndata.obs.keys():
        raise ValueError(f"{clusterColName} was not in anndata observation matrix.")


    # Extract data from the Anndata object
    Labels = scAnndata.obs[clusterColName].tolist()
    Rank_Labels = scAnndata.obs["Rank_Label"].tolist()

   # Count the actual occurrences
    unique_labels = set(Labels)
    actual_counts = {}
    for label in unique_labels:
        indices = [i for i, x in enumerate(Labels) if x == label]
        subset = [Rank_Labels[i] for i in indices]
        actual_counts[label] = dict((x, subset.count(x)) for x in {"Background", "Rank+", "Rank-"})

    # Permutation procedure using multi-threading
    permuted_counts = {label: {"Background": [], "Rank+": [], "Rank-": []} for label in unique_labels}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(permute_once, Rank_Labels, Labels, unique_labels) for _ in range(perm_n)]
        for future in futures:
            result = future.result()
            for label in unique_labels:
                for key in {"Background", "Rank+", "Rank-"}:
                    permuted_counts[label][key].append(result[label][key])

    # Calculate p-values based on the permuted distribution
    p_values = {label: {} for label in unique_labels}
    for label in unique_labels:
        for key in {"Rank+", "Rank-","Background"}:
            observed = actual_counts[label][key]
            if sum(permuted_counts[label][key]) == 0:
                p_values[label][key] = np.nan

            else:
                extreme_count = sum(1 for x in permuted_counts[label][key] if observed > x)
                p_values[label][key] = extreme_count / perm_n

    df_p_values = pd.DataFrame(p_values).T  # transpose the DataFrame to get labels as rows


    return df_p_values

## Hyper-paremeters tuning
def objective(trial, model, train_loader_Bulk, val_loader_Bulk, train_loader_SC, adj_A, device, pheno, infer_mode, model_save_path):
    
    # Define hyperparameters with trial object
    lr_choices = [1e-2, 6e-3, 3e-3, 1e-3, 6e-4, 3e-4, 1e-4,6e-5, 3e-5, 1e-5]
    lr = trial.suggest_categorical("lr", lr_choices)

    n_epochs_choices = [200,250,300,350,400,450,500]
    n_epochs = trial.suggest_categorical("n_epochs", n_epochs_choices)

    # Define alpha values as specific choices
    alpha_0_choices = [0.1, 0.2, 0.3, 0.4, 0.5]
    alpha_1_choices = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    alpha_2_choices = [0.1, 0.2, 0.3, 0.4, 0.5]
    alpha_3_choices = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Suggest categorical choices for each alpha
    alphas = [
        trial.suggest_categorical("alpha_0", alpha_0_choices),
        trial.suggest_categorical("alpha_1", alpha_1_choices),
        trial.suggest_categorical("alpha_2", alpha_2_choices),
        trial.suggest_categorical("alpha_3", alpha_3_choices),
    ]


    # Initialize variables for early stopping
    # best_val_loss = float('inf')
    # patience = (n_epochs // 4) * 3  # You can adjust the patience level
    # patience_counter = 0

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    for epoch in range(n_epochs):
        ## Train
        model.train()
        train_loss = Train_one_epoch(
            model=model,
            dataloader_A=train_loader_Bulk, dataloader_B=train_loader_SC,
            pheno=pheno, infer_mode=infer_mode,
            adj_A=adj_A,
            optimizer=optimizer, alphas=alphas, device=device)

        scheduler.step()

        ## Val
        model.eval()
        val_loss = Validate_model(
            model=model,
            dataloader_A=val_loader_Bulk, dataloader_B=train_loader_SC,
            pheno=pheno, infer_mode=infer_mode,
            adj_A=adj_A,
            alphas=[1,0,1,0], device=device)

        # # Check for early stopping conditions
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter = 0  # Reset patience counter on improvement
        # else:
        #     patience_counter += 1  # Increment patience counter if no improvement

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping check
        # if patience_counter >= patience:
        #     print(f"Early stopping triggered at epoch {epoch} with Validation Loss = {val_loss:.4f}")
        #     break

    if trial.number == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path, "model_trial_{}_val_loss_{:.4f}.pt".format(trial.number, val_loss)))

    if trial.study.best_trials and val_loss <= trial.study.best_value:
        print(f"New best trial found: Trial {trial.number} with Validation Loss = {val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(model_save_path, "model_trial_{}_val_loss_{:.4f}.pt".format(trial.number, val_loss)))

    return val_loss

def tune_hyperparameters(model, train_loader_Bulk, val_loader_Bulk, train_loader_SC, adj_A, device, pheno, infer_mode, n_trials=50):
    model_save_path = "./checkpoints/"
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model, train_loader_Bulk, val_loader_Bulk, train_loader_SC, adj_A, device, pheno, infer_mode, model_save_path), n_trials=n_trials)
    # optuna.visualization.plot_optimization_history(study)

    # Return the best hyperparameters
    return study.best_trial.params

## Extract GP on other datasets
def transform_test_exp(train_exp, test_exp):
    # Initialize a new DataFrame to store the transformed test data
    transformed_test_exp = pd.DataFrame(index=test_exp.columns)

    # Iterate over the columns in the train_exp DataFrame
    for column in train_exp.columns:
        # Parse the column name to get the two gene names
        geneA, geneB = column.split('__')
        
        # Check if both genes are present in the test_exp
        if geneA in test_exp.index and geneB in test_exp.index:
            # Perform the comparison for each sample in test_exp and assign the values to the new DataFrame
            transformed_test_exp[column] = (test_exp.loc[geneA] > test_exp.loc[geneB]).astype(int) * 2 - 1
            
            # Handle cases where geneA == geneB by assigning 0
            transformed_test_exp.loc[:, column][test_exp.loc[geneA] == test_exp.loc[geneB]] = 0
        else:
            # If one or both genes are not present, assign 0 for all samples
            transformed_test_exp[column] = 0

    # Transpose the DataFrame to match the structure of train_exp
    # transformed_test_exp = transformed_test_exp.transpose()

    return transformed_test_exp