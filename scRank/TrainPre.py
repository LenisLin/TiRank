
import torch

import pandas as pd
import numpy as np

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
        embeddings_a, risk_scores_a, condidence_scores_a = model(X_a)
        embeddings_b, _, _ = model(X_b)

        # Calculate loss
        if pheno == 'Cox':
            bulk_loss_ = cox_loss(risk_scores_a, t, e)

        elif pheno == 'Bionomial':
            bulk_loss_ = CrossEntropy_loss(risk_scores_a, label)

        elif pheno == 'Regression':
            bulk_loss_ = MSE_loss(risk_scores_a, label)

        if infer_mode == 'Subpopulation':
            mmd_loss_ = mmd_loss(embeddings_a, embeddings_b)
            rejection_loss_ = rejection_loss(condidence_scores_a, e)

            # total loss

            total_loss = bulk_loss_ * \
                alphas[0] + mmd_loss_ * \
                alphas[1] + rejection_loss_ * alphas[2]

        elif infer_mode in ['Cell', 'Spot']:
            cosine_loss_ = cosine_loss(embeddings_b, A)
            mmd_loss_ = mmd_loss(embeddings_a, embeddings_b)
            rejection_loss_ = rejection_loss(condidence_scores_a, e)

            # total loss

            total_loss = bulk_loss_ * \
                alphas[0] + cosine_loss_ * alphas[1] + mmd_loss_ * \
                alphas[2] + rejection_loss_ * alphas[3]

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(dataloader_B)

# Predict


def Predict(model, Exp, rownames, filePath):
    model.eval()

    Exp_Tensor = torch.from_numpy(np.array(Exp))
    Exp_Tensor = torch.tensor(Exp_Tensor, dtype=torch.float32)
    embeddings, risk_scores, confidence_scores = model(Exp_Tensor)

    embeddings = embeddings.detach().numpy()
    risk_scores = risk_scores.detach().numpy().reshape(-1, 1)
    confidence_scores = confidence_scores.detach().numpy().reshape(-1, 1)

    saveDF = pd.DataFrame(data=np.concatenate(
        (confidence_scores, risk_scores, embeddings), axis=1), index=Exp.index)

    colnames = ["Confidence_score", "Risk_score"]
    colnames.extend(["embedding_" + str(i + 1)
                    for i in range(embeddings.shape[1])])

    saveDF.columns = colnames
    saveDF.index = rownames

    saveDF.to_csv(filePath)

    return None
