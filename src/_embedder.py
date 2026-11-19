import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

class Gemma2ImageEmbedder(nn.Module):
  """Frozen multimodal backbone wrapper using google/embeddinggemma-2 on CPU."""

  def __init__(
    self,
    model_id: str = "google/embeddinggemma-2",
    img_size: int = 224,
  ) -> None:
    """
    Args:
      model_id: Hugging Face model repository ID.
      img_size: Input image height/width used to infer output embedding dimension.
    """
    super().__init__()
    self.model_id = model_id

    self.processor = AutoProcessor.from_pretrained(
      model_id,
      trust_remote_code=True,
    )
    self.backbone = AutoModel.from_pretrained(
      model_id,
      dtype=torch.float32,
      device_map="cpu",
      low_cpu_mem_usage=False,
      trust_remote_code=True,
    )

    self.backbone.eval()
    for param in self.backbone.parameters():
      param.requires_grad = False

    self.embed_dim = self._infer_embed_dim(img_size)

  def train(self, mode: bool = True) -> "Gemma2ImageEmbedder":
    """Keep backbone permanently in eval mode."""
    super().train(mode)
    self.backbone.eval()
    return self

  def to(self, *args, **kwargs) -> "Gemma2ImageEmbedder":
    """Keep backbone pinned to CPU to prevent Apple Metal (MPS) runtime crashes."""
    self.backbone.cpu()
    return self

  @torch.no_grad()
  def forward(
    self,
    images: torch.Tensor,
    target_device: torch.device | str | None = None,
  ) -> torch.Tensor:
    """
    Extract raw image embedding vectors independently for each image in the batch.

    Args:
      images: Normalized image tensor of shape [B, C, H, W].
      target_device: Device to move output embeddings onto (e.g., 'mps' or 'cpu').

    Returns:
      Raw embedding tensor of shape [B, embed_dim] in float32.
    """
    out_device = target_device if target_device is not None else images.device
    images_cpu = images.detach().cpu().float()

    emb_list = []

    for i in range(images_cpu.shape[0]):
      single_img = images_cpu[i]  # [C, H, W]

      inputs = self.processor(
        images=[single_img],
        return_tensors="pt",
        images_kwargs={"do_resize": False, "do_rescale": False, "do_normalize": False},
      )

      model_inputs = {
        k: v.cpu() for k, v in inputs.items() if isinstance(v, torch.Tensor)
      }

      outputs = self.backbone(**model_inputs)

      if hasattr(outputs, "embeddings") and outputs.embeddings is not None:
        emb = outputs.embeddings
      elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        emb = outputs.pooler_output
      else:
        last_hidden = outputs.last_hidden_state
        if "attention_mask" in model_inputs:
          mask = model_inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
          emb = (last_hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
          emb = last_hidden.mean(dim=1)

      if emb.ndim == 1:
        emb = emb.unsqueeze(0)

      emb_list.append(emb.float())

    batch_emb = torch.cat(emb_list, dim=0)
    return batch_emb.to(out_device)

  def _infer_embed_dim(self, img_size: int) -> int:
    """Run a single dummy image to determine output embedding size."""
    dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    out = self.forward(dummy, target_device="cpu")
    return int(out.shape[-1])