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


def cosine_loss(embeddings, A, weight_connected=1.0, weight_unconnected=0.1):
    """
    Compute cosine similarity loss with balanced weighting for sparse adjacency matrix A.

    Args:
        embeddings (Tensor): Embeddings, shape [n_cells, embedding_dim]
        A (Tensor): Sparse adjacency matrix, shape [n_cells, n_cells]
        weight_connected (float): Weight for connected pairs (A = 1)
        weight_unconnected (float): Weight for unconnected pairs (A = 0)

    Returns:
        Tensor: Weighted cosine loss
    """
    embeddings = embeddings.to(A.device)
    # Compute cosine similarity matrix B
    B = torch.mm(embeddings, embeddings.T)
    magnitudes = torch.norm(embeddings, dim=1, keepdim=True)
    B = B / (magnitudes * magnitudes.T)
    B = B - torch.eye(B.shape[0], device=B.device)  # Zero out diagonal

    # Scale A to match B’s range
    A_scaled = 2 * A - 1

    # Weight matrix for balancing
    weights = A * weight_connected + (1 - A) * weight_unconnected

    # Weighted loss
    loss = torch.mean(weights * torch.abs(B - A_scaled))
    return loss

def gaussian_kernel(a, b, sigma = 1.0):
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
    numerator = (a_core - b_core).pow(2).sum(2) / (sigma ** 2)
    return torch.exp(-numerator)


def mmd_loss(embeddings_A, embeddings_B, sigma = 1.0):
    """
    Calculates the Maximum Mean Discrepancy (MMD) loss between two sets of embeddings.

    Args:
        embeddings_A (Tensor), embeddings_B (Tensor): Input tensors containing embeddings

    Returns:
        Tensor: MMD loss for each sample in embeddings_B, tensor of shape (embeddings_B.shape[0],)
    """
    kernel_matrix_A = gaussian_kernel(embeddings_A, embeddings_A, sigma)
    kernel_matrix_B = gaussian_kernel(embeddings_B, embeddings_B, sigma)
    kernel_matrix_AB = gaussian_kernel(embeddings_A, embeddings_B, sigma)
    mmd_loss_ = kernel_matrix_A.mean() + kernel_matrix_B.mean() - 2 * kernel_matrix_AB.mean()

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

    # loss_fn = nn.NLLLoss()
    # loss = loss_fn(torch.log(y_pred), y_true)

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

def prototype_loss(cell_embeddings, bulk_embeddings, bulk_labels, threshold=0.1, margin=1.0):
    # Compute prototypes from bulk RNA-seq data using class indices
    rank_plus_proto = bulk_embeddings[bulk_labels == 0].mean(dim=0)  # 0 for 'Rank+'
    rank_minus_proto = bulk_embeddings[bulk_labels == 1].mean(dim=0)  # 1 for 'Rank-'
    
    # Compute distances for single-cell embeddings to both prototypes
    dist_to_plus = torch.norm(cell_embeddings - rank_plus_proto, dim=1)
    dist_to_minus = torch.norm(cell_embeddings - rank_minus_proto, dim=1)
    
    # Confidence: difference in distances
    confidence = torch.abs(dist_to_plus - dist_to_minus)
    mask = (confidence > threshold).float()  # Only use confident cells
    
    # Pseudo-labels for single-cell data: 0 if closer to Rank+, 1 if closer to Rank-
    pseudo_labels = (dist_to_plus < dist_to_minus).long()
    
    # Contrastive distances
    correct_dist = torch.where(pseudo_labels == 0, dist_to_plus, dist_to_minus)
    incorrect_dist = torch.where(pseudo_labels == 0, dist_to_minus, dist_to_plus)
    
    # Contrastive loss: minimize correct_dist, maximize incorrect_dist up to margin
    loss = torch.mean(mask * (correct_dist + torch.relu(margin - incorrect_dist)))
    
    return loss