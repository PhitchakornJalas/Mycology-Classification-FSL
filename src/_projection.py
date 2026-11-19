import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
  """Trainable MLP projection head followed by L2 normalization."""

  def __init__(
    self,
    in_dim: int,
    hidden_dim: int = 512,
    out_dim: int = 256,
    dropout: float = 0.1,
  ) -> None:
    """
    Args:
      in_dim: Input embedding dimension from the backbone.
      hidden_dim: Intermediate hidden layer dimension.
      out_dim: Final projected embedding dimension.
      dropout: Dropout probability between linear layers.
    """
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.LayerNorm(hidden_dim),
      nn.GELU(),
      nn.Dropout(p=dropout),
      nn.Linear(hidden_dim, out_dim),
    )

  def forward(self, x) -> torch.Tensor:
    """Project raw embeddings and normalize onto the unit hypersphere.

    Args:
      x: Raw embedding tensor of shape [B, in_dim].

    Returns:
      L2-normalized embedding tensor of shape [B, out_dim].
    """
    projected = self.net(x)
    return F.normalize(projected, p=2, dim=-1)