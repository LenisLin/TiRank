# loss function
import torch
import numpy as np

# Cox partial likelihood as the loss function


def cox_loss(pred, t, e, margin=0.1):
    assert len(pred) == len(t) == len(e)
    '''
    Args:
        pred (Tensor): the predicted risk scores from your model, shape [batch_size]
        t (Tensor): the event times, shape [batch_size]
        e (Tensor): the event indicators (1 if the event occurred, 0 otherwise), shape [batch_size]
        margin (float): the margin parameter
    '''
    # Compute pairwise differences between predictions
    pred_diffs = pred.unsqueeze(1) - pred.unsqueeze(0)

    # Compute pairwise time differences
    time_diffs = t.unsqueeze(1) - t.unsqueeze(0)

    # Compute pairwise event differences
    event_diffs = e.unsqueeze(1) - e.unsqueeze(0)

    # Get a mask for pairs where both events occurred, and the first occurred earlier
    mask = (event_diffs == 0) & (time_diffs < 0)

    # Compute the loss for these pairs, incorporating the margin
    losses = torch.log(1 + torch.exp(-(pred_diffs[mask] - margin)))

    # Average the losses
    loss = torch.mean(losses)
    # loss = losses

    return loss

# Reconstruction loss
# Compute pairwise cosine similarity in embeddings


def cosine_similarity_matrix(embeddings):
    """
    Given a tensor of embeddings, compute a cosine similarity matrix.
    """
    # normed_embeddings = F.normalize(embeddings)
    cos_sim_matrix = torch.mm(embeddings, embeddings.transpose(0, 1))
    return cos_sim_matrix


def cosine_loss(embeddings, A):

    B = cosine_similarity_matrix(embeddings)

    # Compute the difference between original matrix and embedding space matrix
    matrix_diff = A - B

    # Use the mean squared difference as your loss
    # cosine_loss = torch.mean(matrix_diff ** 2)
    cosine_loss = torch.mean(matrix_diff ** 2,dim = 0)

    return cosine_loss

# Compute MMD loss between different modality


# def mmd_loss(embeddings_A, embeddings_B):

#     # Calculate the means of the embeddings
#     mean_A = torch.mean(embeddings_A, dim=0)
#     mean_B = torch.mean(embeddings_B, dim=0)

#     # MMD loss is the squared distance between the means
#     mmd_loss_ = torch.sum((mean_A - mean_B) ** 2)

#     return mmd_loss_

def mmd_loss(embeddings_A, embeddings_B):

    # Calculate the squared difference between each instance in B and the mean of A
    mean_A = torch.mean(embeddings_A, dim=0)
    mmd_losses_B = torch.sum((embeddings_B - mean_A[None, :]) ** 2, dim=1)

    return mmd_losses_B


# Cell subpopulation classification entropy loss


def CellClassEntropy():
    return

# Confidence loss


def rejection_loss(InsLoss, Confiscore, c, mtype):
    rej_loss_func_dict = {
        'LR': lambda InsLoss, Confiscore: torch.relu(torch.max(1 + (Confiscore + InsLoss) / 2, c * (1 - (1/(1-2*c)) * Confiscore))),
        'GLR': lambda InsLoss, Confiscore: torch.relu(torch.max(Confiscore + InsLoss, c * (1 - Confiscore))),
        'NGLR': lambda InsLoss, Confiscore: torch.relu(torch.max(InsLoss * (1 + Confiscore), c * (1 - Confiscore))),
        'Sigmoid': lambda InsLoss, Confiscore: InsLoss * torch.sigmoid(Confiscore) + c * torch.sigmoid(-Confiscore)
    }
    return torch.mean(rej_loss_func_dict[mtype](InsLoss, Confiscore)) 


def L1Norm():
    return
