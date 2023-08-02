# loss function
import torch
import torch.nn as nn
import numpy as np


def cox_loss(pred, t, e, margin=0.1):
    assert len(pred) == len(t) == len(e)
    """
    Calculates the Cox partial log-likelihood loss for survival analysis.
    
    Args:
        pred (Tensor): Predicted risk scores, shape [batch_size]
        t (Tensor): Event times, shape [batch_size]
        e (Tensor): Event indicators (1 if event occurred, 0 otherwise), shape [batch_size]
        margin (float, optional): Margin parameter, default 0.1

    Returns:
        Tensor: Cox partial log-likelihood loss, scalar
    """
    # Compute pairwise differences between predictions
    pred_diffs = pred.unsqueeze(1) - pred.unsqueeze(0)

    # Compute pairwise time differences
    time_diffs = t.unsqueeze(1) - t.unsqueeze(0)

    # Get a mask for pairs where both events occurred, and the first occurred earlier
    mask = (e == 1) & (time_diffs < 0)

    # Compute the loss for these pairs, incorporating the margin
    losses = torch.log(1 + torch.exp(-(pred_diffs[mask] - margin)))

    # Average the losses
    loss = torch.mean(losses)

    return loss


def cosine_loss(embeddings, A):
    """
    Calculates the cosine similarity loss between embeddings.

    Args:
        embeddings (Tensor): Input tensor containing embeddings
        A (Tensor): Original matrix to compare against

    Returns:
        Tensor: Cosine loss, scalar
    """
    B = torch.mm(embeddings, embeddings.T)
    B = B * (1 / torch.diag(B)).to(A.device) - \
        torch.eye(B.shape[0]).to(A.device)

    # Compute the difference between original matrix and embedding space matrix
    matrix_diff = B - A

    cosine_loss = torch.mean(matrix_diff ** 2)

    return cosine_loss


def gaussian_kernel(a, b):
    """
    Calculates the Gaussian kernel similarity between two sets of samples.

    Args:
        a (Tensor), b (Tensor): Input tensors where each row is a sample

    Returns:
        Tensor: Gaussian kernel similarities
    """
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2) / depth
    return torch.exp(-numerator)


def mmd_loss(embeddings_A, embeddings_B):
    """
    Calculates the Maximum Mean Discrepancy (MMD) loss between two sets of embeddings.

    Args:
        embeddings_A (Tensor), embeddings_B (Tensor): Input tensors containing embeddings

    Returns:
        Tensor: MMD loss, scalar
    """
    kernel_matrix_A = gaussian_kernel(embeddings_A, embeddings_A)
    kernel_matrix_B = gaussian_kernel(embeddings_B, embeddings_B)
    kernel_matrix_AB = gaussian_kernel(embeddings_A, embeddings_B)

    mmd_loss_ = kernel_matrix_A.mean() + kernel_matrix_B.mean() - \
        2 * kernel_matrix_AB.mean()

    return mmd_loss_


def CrossEntropy_loss(y_pred, y_true):
    """
    Calculates the cross entropy loss between predicted and true labels.

    Args:
        y_pred (Tensor): Predicted labels, tensor of any shape
        y_true (Tensor): True labels, tensor of the same shape as y_pred

    Returns:
        Tensor: Cross entropy loss, scalar
    """

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y_pred, y_true)

    return loss


def MSE_loss(y_pred, y_true):
    """
    Calculates the Mean Squared Error (MSE) loss between predicted and true values.

    Args:
        y_pred (Tensor): Predicted values, tensor of any shape
        y_true (Tensor): True values, tensor of the same shape as y_pred

    Returns:
        Tensor: MSE loss, scalar
    """

    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y_true)

    return loss


def rejection_loss(condidence_score, e):
    """
    Calculates a rejection loss based on confidence scores and event indicators.

    Args:
        confidence_score (Tensor): Confidence score, tensor of shape [batch_size]
        e (Tensor): Event indicators (1 if event occurred, 0 otherwise), tensor of shape [batch_size]

    Returns:
        Tensor: Rejection loss, scalar
    """
    mask1 = (e == 1)
    diff1 = torch.ones_like(condidence_score).to(
        condidence_score.device) - condidence_score

    mask0 = (e == 0)
    diff0 = (-1) * torch.ones_like(condidence_score).to(
        condidence_score.device) - condidence_score

    reject_loss = torch.mean(diff1[mask1] ** 2) + torch.mean(diff0[mask0] ** 2)

    return reject_loss
