import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class EmbeddingGemmaContrastiveModel(nn.Module):
  """
  Pipeline: google/embeddinggemma-2 -> Fully Connected -> L2-Normalized Embedding
  """

  def __init__(
    self,
    model_id: str = "google/embeddinggemma-2",
    proj_dim: int = 256,
    freeze_backbone: bool = True,
  ):
    super().__init__()

    # 1. โหลด EmbeddingGemma 2 (ใช้ bfloat16 เพื่อประหยัด RAM บน Mac MPS)
    self.backbone = AutoModel.from_pretrained(
      model_id,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True,
    )

    if freeze_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False

    hidden_size = self.backbone.config.hidden_size

    # 2. Fully Connected Projection Head (ส่วนที่อัปเดตน้ำหนักตอนเทรน)
    self.fc = nn.Sequential(
      nn.Linear(hidden_size, proj_dim),
      nn.ReLU(inplace=True),
      nn.Linear(proj_dim, proj_dim),
    )

  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Args:
      pixel_values: Tensor รูปภาพขนาด [B, 3, H, W]
    Returns:
      z: L2-normalized embedding ขนาด [B, proj_dim] พร้อมคำนวณ Cosine Similarity
    """
    pixel_values = pixel_values.to(dtype=torch.bfloat16)

    with torch.no_grad():
      if hasattr(self.backbone, "get_image_features"):
        raw_emb = self.backbone.get_image_features(pixel_values=pixel_values)
      else:
        outputs = self.backbone(pixel_values=pixel_values)
        raw_emb = (
          outputs.pooler_output
          if hasattr(outputs, "pooler_output")
          else outputs.last_hidden_state.mean(dim=1)
        )

    # ส่งผ่าน Fully Connected (แปลงเป็น float32 ก่อนเพื่อความเสถียรของ Gradient บน MPS)
    proj_emb = self.fc(raw_emb.float())

    # L2 Normalize เพื่อให้ dot product (z @ z.T) เท่ากับ Cosine Similarity พอดี
    return F.normalize(proj_emb, p=2, dim=-1)