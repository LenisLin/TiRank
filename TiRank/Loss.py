# loss function
import torch
import torch.nn as nn

def regularization_loss(feature_weights):
    """
    Calculate the L1 regularization loss.

    Parameters:
    feature_weights (torch.Tensor): The learnable weight matrix.

    Returns:
    torch.Tensor: The calculated regularization loss.
    """
    return torch.mean(torch.abs(feature_weights))

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

    # Compute pairwise event differences
    event_diffs = e.unsqueeze(1) - e.unsqueeze(0)

    # Get a mask for pairs where both events occurred, and the first occurred earlier, or where the first is censored
    # but has a longer observed survival time than the second who experienced the event.
    mask = ((e == 1) | (event_diffs == 1)) & (time_diffs < 0)

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
    embeddings = embeddings.to(A.device)
    B = torch.mm(embeddings, embeddings.T)

    # Calculate magnitudes of embeddings
    magnitudes = torch.norm(embeddings, dim=1, keepdim=True)
    B /= magnitudes
    B /= magnitudes.T

    B = B - torch.eye(B.shape[0]).to(B.device)

    # Compute the difference between original matrix and embedding space matrix
    matrix_diff = B - A

    cosine_loss = torch.mean(torch.abs(matrix_diff))

    return cosine_loss

# def gaussian_kernel(a, b):
#     """
#     Calculates the Gaussian kernel similarity between two sets of samples.

#     Args:
#         a (Tensor), b (Tensor): Input tensors where each row is a sample

#     Returns:
#         Tensor: Gaussian kernel similarities
#     """
#     dim1_1, dim1_2 = a.shape[0], b.shape[0]
#     depth = a.shape[1]
#     a = a.view(dim1_1, 1, depth)
#     b = b.view(1, dim1_2, depth)
#     a_core = a.expand(dim1_1, dim1_2, depth)
#     b_core = b.expand(dim1_1, dim1_2, depth)
#     numerator = (a_core - b_core).pow(2).mean(2) / depth
#     return torch.exp(-numerator)


# def mmd_loss(embeddings_A, embeddings_B):
#     """
#     Calculates the Maximum Mean Discrepancy (MMD) loss between two sets of embeddings.

#     Args:
#         embeddings_A (Tensor), embeddings_B (Tensor): Input tensors containing embeddings

#     Returns:
#         Tensor: MMD loss for each sample in embeddings_B, tensor of shape (embeddings_B.shape[0],)
#     """
#     kernel_matrix_A = gaussian_kernel(embeddings_A, embeddings_A)
#     kernel_matrix_B = gaussian_kernel(embeddings_B, embeddings_B)
#     kernel_matrix_AB = gaussian_kernel(embeddings_A, embeddings_B)

#     mmd_loss_ = kernel_matrix_A.mean() + kernel_matrix_B.mean() - \
#         2 * kernel_matrix_AB.mean()

#     return mmd_loss_


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