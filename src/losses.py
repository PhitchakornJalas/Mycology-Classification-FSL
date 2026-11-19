import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
  """
  Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
  คำนวณ Cosine Similarity ระหว่างทุกคู่ใน Episode และดึงคลาสเดียวกันเข้าหากัน
  """

  def __init__(self, temperature: float = 0.07):
    super().__init__()
    self.temperature = temperature

  def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Args:
      embeddings: เวกเตอร์ที่ผ่าน F.normalize มาแล้ว ขนาด [B, D]
      labels:     ป้ายกำกับคลาสใน Episode ขนาด [B]
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # 1. คำนวณ Cosine Similarity Matrix ทุกคู่ -> [B, B]
    sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

    # 2. สร้าง Mask หาคู่ที่มี Label เดียวกัน (Positive Pairs)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # ตัดเส้นทแยงมุมออก (ไม่เทียบตัวมันเองกับตัวมันเอง)
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    # 3. คำนวณ Log-Softmax เทียบกับคู่ทั้งหมด (ป้องกันค่าล้นด้วย max trick)
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    # 4. เฉลี่ยค่า Log-Likelihood เฉพาะคู่ที่เป็นคลาสเดียวกัน (Positive Pairs)
    pos_per_row = mask.sum(1)
    valid_rows = pos_per_row > 0
    mean_log_prob_pos = (mask[valid_rows] * log_prob[valid_rows]).sum(1) / pos_per_row[valid_rows]

    loss = -mean_log_prob_pos.mean()
    return loss