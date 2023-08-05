
from itertools import count
import torch

import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture

from Loss import *

# Training


def Train_one_epoch(model, dataloader_A, dataloader_B, pheno='Cox', infer_mode="Cell", adj_A=None, optimizer=None, alphas=[1, 1, 1, 1], device="cpu"):

    model.train()

    running_loss = 0.0

    # RNA-seq data whole batch training
    iter_A = iter(dataloader_A)

    if pheno == 'Cox':
        (X_a, t, e) = next(iter_A)

    if pheno in ['Bionomial', 'Regression']:
        (X_a, label) = next(iter_A)

    X_a = X_a.to(device)

    if pheno == 'Cox':
        t = t.to(device)
        e = e.to(device)

    if pheno in ['Bionomial', 'Regression']:
        label = label.to(device)

    for batch_B in dataloader_B:
        # Get the next batch of data
        (X_b, idx) = batch_B

        # Move the data to the GPU
        X_b = X_b.to(device)

        if adj_A is not None:
            A = adj_A[idx, :][:, idx]
            A = A.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        embeddings_a, risk_scores_a = model(X_a)
        embeddings_b, _ = model(X_b)

        # Calculate loss
        if pheno == 'Cox':
            bulk_loss_ = cox_loss(risk_scores_a, t, e)

        elif pheno == 'Bionomial':
            bulk_loss_ = CrossEntropy_loss(risk_scores_a, label)

        elif pheno == 'Regression':
            bulk_loss_ = MSE_loss(risk_scores_a, label)

        if infer_mode == 'Subpopulation':
            mmd_loss_ = mmd_loss(embeddings_a, embeddings_b)

            # total loss

            total_loss = bulk_loss_ * \
                alphas[0] + mmd_loss_ * alphas[2]

        elif infer_mode in ['Cell', 'Spot']:
            cosine_loss_ = cosine_loss(embeddings_b, A)
            mmd_loss_ = mmd_loss(embeddings_a, embeddings_b)

            # total loss

            total_loss = bulk_loss_ * alphas[0] + \
                cosine_loss_ * alphas[1] + \
                mmd_loss_ * alphas[2]

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(dataloader_B)

# Predict


def Predict(model, bulk_GPmat, sc_GPmat, mode, sc_rownames, do_reject=True):
    model.eval()

    # Predict on cell
    Exp_Tensor_sc = torch.from_numpy(np.array(sc_GPmat))
    Exp_Tensor_sc = torch.tensor(Exp_Tensor_sc, dtype=torch.float32)
    embeddings_sc, pred_sc = model(Exp_Tensor_sc)

    # Predict on bulk
    Exp_Tensor_bulk = torch.from_numpy(np.array(bulk_GPmat))
    Exp_Tensor_bulk = torch.tensor(Exp_Tensor_bulk, dtype=torch.float32)
    _, pred_bulk = model(Exp_Tensor_bulk)

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
        reject_mask = Reject(pred_sc)
        print(
            f"Reject {int(sum(reject_mask))}({int(sum(reject_mask)) / len(reject_mask) :.2f}%) cells.")

    saveDF = pd.DataFrame(data=np.concatenate(
        (reject_mask, pred_sc, embeddings), axis=1), index=sc_GPmat.index)

    colnames = ["Reject", "Pred_score"]
    colnames.extend(["embedding_" + str(i + 1)
                    for i in range(embeddings.shape[1])])

    saveDF.columns = colnames
    saveDF.index = sc_rownames

    return saveDF

# Reject


def Reject(pred_sc):
    # Fit a GMM with 2 components on bulk
    # gmm_bulk = GaussianMixture(n_components=2, random_state=0).fit(pred_bulk)

    # Fit a GMM with 3 components on sc / spot
    gmm_sc = GaussianMixture(n_components=3, random_state=0).fit(pred_sc)

    # Print the means and covariances
    print("Means of the gaussians in gmm_sc: ", gmm_sc.means_)
    print("Covariances of the gaussians in gmm_sc: ", gmm_sc.covariances_)

    # Find the component whose mean is nearest to 0.5
    diffs = np.abs(gmm_sc.means_ - 0.5)
    nearest_component = np.argmin(diffs)

    # The mask of those rejection cell
    labels_sc = gmm_sc.predict(pred_sc)

    mask = np.zeros(shape=(len(labels_sc), 1))

    mask[labels_sc == nearest_component] = 1

    return mask
