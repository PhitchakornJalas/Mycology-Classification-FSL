import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarity(nn.Module):
  """Compute pairwise or prototype-based Cosine Similarity for embeddings."""

  def __init__(self, temperature: float = 0.07) -> None:
    """
    Args:
      temperature: Temperature scaling factor for contrastive logits.
    """
    super().__init__()
    self.temperature = temperature

  def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    scale: bool = True,
  ) -> torch.Tensor:
    """
    Compute cosine similarity matrix between query and key embeddings.

    Args:
      query: Embedding tensor of shape [M, D].
      key: Embedding tensor of shape [N, D]. If None, computes self-similarity [M, M].
      scale: Whether to divide similarity scores by `temperature`.

    Returns:
      Similarity matrix of shape [M, N] (or [M, M] if key is None).
    """
    q_norm = F.normalize(query, p=2, dim=-1)
    k_norm = q_norm if key is None else F.normalize(key, p=2, dim=-1)

    sim = torch.matmul(q_norm, k_norm.T)
    if scale:
      sim = sim / self.temperature
    return sim