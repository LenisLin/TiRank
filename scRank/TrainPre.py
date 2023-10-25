import torch

import pandas as pd
import numpy as np
import scanpy as sc

from sklearn.mixture import GaussianMixture

from concurrent.futures import ThreadPoolExecutor

from Loss import *

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

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(dataloader_B)

# Predict


def Predict(model, bulk_GPmat, sc_GPmat, mode, sc_rownames, do_reject=True, tolerance=0.05):
    model.eval()

    # Predict on cell
    Exp_Tensor_sc = torch.from_numpy(np.array(sc_GPmat))
    Exp_Tensor_sc = torch.tensor(Exp_Tensor_sc, dtype=torch.float32)
    embeddings_sc, pred_sc, _  = model(Exp_Tensor_sc)

    # Predict on bulk
    Exp_Tensor_bulk = torch.from_numpy(np.array(bulk_GPmat))
    Exp_Tensor_bulk = torch.tensor(Exp_Tensor_bulk, dtype=torch.float32)
    _, pred_bulk, _ = model(Exp_Tensor_bulk)

    if mode == "Cox":
        pred_bulk = pred_bulk.detach().numpy().reshape(-1, 1)
        pred_sc = pred_sc.detach().numpy().reshape(-1, 1)

    if mode == "Bionomial":
        pred_sc = torch.nn.functional.softmax(
            pred_sc)[:, 1].detach().numpy().reshape(-1, 1)

        pred_bulk = torch.nn.functional.softmax(
            pred_bulk)[:, 1].detach().numpy().reshape(-1, 1)

    embeddings = embeddings_sc.detach().numpy()

    if do_reject:
        reject_mask = Reject(pred_bulk, pred_sc,
                             tolerance=tolerance, max_components=10)

    saveDF = pd.DataFrame(data=np.concatenate(
        (reject_mask, pred_sc, embeddings), axis=1), index=sc_GPmat.index)

    colnames = ["Reject", "Pred_score"]
    colnames.extend(["embedding_" + str(i + 1)
                    for i in range(embeddings.shape[1])])

    saveDF.columns = colnames
    saveDF.index = sc_rownames

    return saveDF

# Reject


def Reject(pred_bulk, pred_sc, tolerance, max_components):

    gmm_bulk = GaussianMixture(n_components=2, random_state=619).fit(pred_bulk)

    gmm_bulk_mean_1 = np.max(gmm_bulk.means_)
    gmm_bulk_mean_0 = np.min(gmm_bulk.means_)

    if (gmm_bulk_mean_1 - gmm_bulk_mean_0) <= 0.5:
        print("Underfitting!")

    # Iterate over the number of components
    for n_components in range(2, max_components + 1):
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


# categorize
def categorize(scAnndata, sc_PredDF, do_cluster=False):
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
        scAnndata.obs["Rank_Score"] = scAnndata.obs["Rank_Score"] * \
            (1 - scAnndata.obs["Reject"])
        scAnndata.obs["Rank_Label"] = [
            "Background" if i == 0 else
            "Rank-" if 0 < i <= 0.5 else
            "Rank+"
            for i in scAnndata.obs["Rank_Score"]
        ]

        print(f"We set Rank score <= 0.5 as Rank- () while > 0.5 as Rank+ ")

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
