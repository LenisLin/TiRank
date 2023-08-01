
import torch

import pandas as pd
import numpy as np
from itertools import cycle

from Loss import *

# Training


def Train_one_epoch(model, dataloader_A, dataloader_B, adj_A=None, adj_B=None, c = 0.4, optimizer=None, alphas=[1, 1, 1], mode = "SC", device="cpu"):
    model.train()

    running_loss = 0.0

    # Create an iterator for both dataloaders
    iter_A = iter(cycle(dataloader_A))

    for batch_B in dataloader_B:
        # Get the next batch of data
        (X_a, t, e) = next(iter_A)
        (X_b, idx) = batch_B

        # X_a = X_a.unsqueeze(0)
        # X_b = X_b.unsqueeze(0)

        # Move the data to the GPU
        X_a = X_a.to(device)
        t = t.to(device)
        e = e.to(device)
        X_b = X_b.to(device)

        if adj_A is not None:
            A = adj_A[idx, :][:, idx]
            A = A.to(device)

        if adj_B is not None:
            B = adj_B[idx, :][:, idx]
            B = B.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        embeddings_a, risk_scores_a, _ = model(X_a)
        embeddings_b, _ , confidence_scores_b  = model(X_b)

        # Calculate loss
        cox_loss_ = cox_loss(risk_scores_a, t, e)
        cosine_loss_ = cosine_loss(embeddings_b, A)
        mmd_loss_ = mmd_loss(embeddings_a, embeddings_b)

        # total loss
        cell_loss = cosine_loss_ * alphas[1] + mmd_loss_ * alphas[2]
        cell_loss_with_confidence = rejection_loss(cell_loss,confidence_scores_b,c=c,mtype="NGLR")

        total_loss = cox_loss_ * alphas[0] + cell_loss_with_confidence

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(dataloader_B)


def Evaluate(model, test_loader, alphas):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for (x_a, t, e), (x_b, A) in test_loader:
            risk_scores_a, embeddings_a = model(x_a)
            _, embeddings_b = model(x_b)

            cox_loss = cox_loss(risk_scores_a, t, e)
            cosine_loss = cosine_loss(embeddings_b, A)
            mmd_loss = mmd_loss(embeddings_a, embeddings_b)

            loss = all_loss(
                loss_List=[cox_loss, cosine_loss, mmd_loss], alpha_List=alphas)
            total_loss += loss.item()

    return total_loss / len(test_loader)

# Predict


def Predict(model, Exp, filePath):
    model.eval()

    Exp_Tensor = torch.from_numpy(np.array(Exp))
    Exp_Tensor = torch.tensor(Exp_Tensor, dtype=torch.float)
    riskScoreOnCell = model(Exp_Tensor)
    riskScoreOnCell = riskScoreOnCell[0].detach().numpy()

    saveDF = pd.DataFrame(data=riskScoreOnCell, index=Exp.index)
    saveDF.to_csv(filePath)

    return None
