import torch
import torch.nn as nn

from _embedder import Gemma2ImageEmbedder
from _projection import ProjectionHead
from _similarity import CosineSimilarity
from _losses import SupConLoss


class FewShotContrastivePipeline(nn.Module):
  """
  End-to-end 4-stage pipeline:
  Stage 1: Frozen Gemma2ImageEmbedder (CPU)
  Stage 2: Trainable ProjectionHead + L2 Norm (MPS)
  Stage 3: CosineSimilarity (MPS)
  Stage 4: SupConLoss (MPS)
  """

  def __init__(
    self,
    model_id: str = "google/embeddinggemma-2",
    img_size: int = 224,
    hidden_dim: int = 512,
    out_dim: int = 256,
    dropout: float = 0.1,
    temperature: float = 0.07,
  ) -> None:
    super().__init__()
    self.img_size = img_size
    # Stage 1: Frozen Embedder
    self.embedder = Gemma2ImageEmbedder(
      model_id=model_id,
      img_size=img_size,
    )
    # Stage 2: Trainable Projection Head
    self.head = ProjectionHead(
      in_dim=self.embedder.embed_dim,
      hidden_dim=hidden_dim,
      out_dim=out_dim,
      dropout=dropout,
    )
    # Stage 3: Cosine Similarity
    self.similarity = CosineSimilarity(temperature=temperature)
    # Stage 4: Supervised Contrastive Loss
    self.criterion = SupConLoss()

  def to(self, *args, **kwargs) -> "FewShotContrastivePipeline":
    """Keep Stage 1 pinned to CPU while moving Stages 2-4 to the target device."""
    self.embedder.to(*args, **kwargs)
    self.head = self.head.to(*args, **kwargs)
    self.similarity = self.similarity.to(*args, **kwargs)
    self.criterion = self.criterion.to(*args, **kwargs)
    return self

  def encode(self, images: torch.Tensor) -> torch.Tensor:
    """Run Stage 1 -> Stage 2 to obtain L2-normalized embeddings."""
    head_device = next(self.head.parameters()).device
    raw_emb = self.embedder(images, target_device=head_device)
    return self.head(raw_emb)

  def forward(
    self,
    images: torch.Tensor,
    labels: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    """Execute the full pipeline from images to embeddings, similarity, and optional loss."""
    embeddings = self.encode(images)
    sim_logits = self.similarity(embeddings, scale=True)

    output = {
      "embeddings": embeddings,
      "similarity": sim_logits,
    }

    if labels is not None:
      output["loss"] = self.criterion(sim_logits, labels)

    return output

  @torch.no_grad()
  def summary(self, n_way: int = 3, k_shot: int = 5) -> None:
    """Print a Keras-style architecture & tensor-flow summary across all 4 stages."""
    head_device = next(self.head.parameters()).device
    batch_size = n_way * k_shot

    dummy_images = torch.zeros(batch_size, 3, self.img_size, self.img_size)
    dummy_labels = torch.arange(n_way, device=head_device).repeat_interleave(k_shot)

    # Trace shapes through each stage
    raw_emb = self.embedder(dummy_images, target_device=head_device)
    proj_emb = self.head(raw_emb)
    sim_logits = self.similarity(proj_emb, scale=True)
    loss = self.criterion(sim_logits, dummy_labels)

    def count_params(module: nn.Module) -> tuple[int, int]:
      total = sum(p.numel() for p in module.parameters())
      trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
      return total, trainable

    s1_tot, s1_trn = count_params(self.embedder)
    s2_tot, s2_trn = count_params(self.head)
    s3_tot, s3_trn = count_params(self.similarity)
    s4_tot, s4_trn = count_params(self.criterion)

    rows = [
      (
        "Stage 1: Gemma2Embedder",
        f"{list(dummy_images.shape)}",
        f"{list(raw_emb.shape)}",
        f"cpu -> {raw_emb.device}",
        f"{s1_trn:,} / {s1_tot:,}",
      ),
      (
        "Stage 2: ProjectionHead+L2",
        f"{list(raw_emb.shape)}",
        f"{list(proj_emb.shape)}",
        f"{proj_emb.device}",
        f"{s2_trn:,} / {s2_tot:,}",
      ),
      (
        "Stage 3: CosineSimilarity",
        f"{list(proj_emb.shape)}",
        f"{list(sim_logits.shape)}",
        f"{sim_logits.device}",
        f"{s3_trn:,} / {s3_tot:,}",
      ),
      (
        "Stage 4: SupConLoss",
        f"{list(sim_logits.shape)} + {list(dummy_labels.shape)}",
        f"[] (val={loss.item():.4f})",
        f"{loss.device}",
        f"{s4_trn:,} / {s4_tot:,}",
      ),
    ]

    width = 105
    print("=" * width)
    print(f"MODEL PIPELINE SUMMARY ({n_way}-Way {k_shot}-Shot | Batch Size = {batch_size})")
    print("=" * width)
    print(f"{'Stage (Layer)':<28} | {'Input Shape':<24} | {'Output Shape':<18} | {'Device':<12} | {'Trainable / Total':<18}")
    print("-" * width)
    for name, in_s, out_s, dev, params in rows:
      print(f"{name:<28} | {in_s:<24} | {out_s:<18} | {dev:<12} | {params:<18}")
    print("=" * width)
    total_all = s1_tot + s2_tot + s3_tot + s4_tot
    trainable_all = s1_trn + s2_trn + s3_trn + s4_trn
    print(f"Total params    : {total_all:,}")
    print(f"Trainable params: {trainable_all:,} (Stage 2 ProjectionHead only)")
    print(f"Frozen params   : {total_all - trainable_all:,} (Stage 1 Gemma2 Backbone)")
    print("=" * width)


if __name__ == "__main__":
  device = torch.device("cpu")
  pipeline = FewShotContrastivePipeline(
    model_id="google/embeddinggemma-2",
    img_size=224,
    hidden_dim=512,
    out_dim=256,
  ).to(device)

  pipeline.summary(n_way=3, k_shot=5)