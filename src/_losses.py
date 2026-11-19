import torch
import torch.nn as nn


class SupConLoss(nn.Module):
  """Supervised Contrastive Loss computed from a temperature-scaled similarity matrix."""

  def __init__(self) -> None:
    super().__init__()

  def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute Supervised Contrastive Loss.

    Args:
      logits: Temperature-scaled cosine similarity matrix of shape [N*K, N*K].
      labels: Ground-truth episode class labels of shape [N*K].

    Returns:
      Scalar loss tensor.
    """
    device = logits.device
    batch_size = logits.shape[0]

    # Mask of positive pairs (same class label)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # Subtract max for numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # Mask out self-contrast (diagonal)
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    # Compute log-probabilities
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # Mean of log-likelihood over positive pairs
    pos_per_row = mask.sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_per_row

    return -mean_log_prob_pos.mean()