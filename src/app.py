import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

from transforms import ImageTransformBuilder
from models import FewShotContrastivePipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"


def load_inference_engine(bundle_path: Path):
  if not bundle_path.exists():
    raise FileNotFoundError(
      f"Deployed bundle not found: {bundle_path}\n"
      f"Please run 'python src/export_prototypes.py' first."
    )

  device = torch.device("cpu")
  bundle = torch.load(bundle_path, map_location=device, weights_only=True)

  model_id = bundle["model_id"]
  img_size = int(bundle["img_size"])
  mean = tuple(bundle["mean"])
  std = tuple(bundle["std"])
  hidden_dim = int(bundle["hidden_dim"])
  out_dim = int(bundle["out_dim"])
  temperature = float(bundle["temperature"])

  class_names = bundle["class_names"]
  prototypes = bundle["prototypes"].to(device)  # [Num_Classes, D]

  tf_builder = ImageTransformBuilder(
    img_size=img_size,
    mean=mean,
    std=std,
    use_augmentation=False,
    use_normalize=True,
  )
  eval_tf = tf_builder.get_eval_transform()

  pipeline = FewShotContrastivePipeline(
    model_id=model_id,
    img_size=img_size,
    hidden_dim=hidden_dim,
    out_dim=out_dim,
    dropout=0.0,
    temperature=temperature,
  ).to(device)
  pipeline.head.load_state_dict(bundle["head_state_dict"])
  pipeline.eval()

  @torch.no_grad()
  def predict_plate(image: Image.Image | None) -> tuple[dict[str, float], str]:
    if image is None:
      return {}, "กรุณาอัปโหลดภาพจานเพาะเชื้อ (Agar Plate) หรือถ่ายภาพจากกล้อง"

    img_tensor = eval_tf(image.convert("RGB")).unsqueeze(0)  # [1, 3, H, W]
    backbone_emb = pipeline.embedder(img_tensor).to(device)
    query_z = F.normalize(pipeline.head(backbone_emb), p=2, dim=-1)  # [1, D]

    # คำนวณ Cosine Similarity กับ Prototype ทุกคลาส
    cos_sim = torch.matmul(query_z, prototypes.T).squeeze(0)  # [Num_Classes]

    # แปลงเป็น Confidence Probability ด้วย Temperature-scaled Softmax
    probs = F.softmax(cos_sim / temperature, dim=-1)

    top_k = min(3, len(class_names))
    top_probs, top_idxs = torch.topk(probs, k=top_k)

    # Gradio Label รับ dict[str, float] เพื่อเรนเดอร์แถบ Top-3 Confidence Score อัตโนมัติ
    confidences = {
      class_names[int(idx.item())]: float(prob.item())
      for prob, idx in zip(top_probs, top_idxs)
    }

    best_idx = int(top_idxs[0].item())
    best_name = class_names[best_idx]
    best_prob = float(top_probs[0].item()) * 100.0
    best_sim = float(cos_sim[best_idx].item())

    detail_md = (
      f"### ผลการจำแนกเบื้องต้น: **{best_name}**\n"
      f"- **Confidence Score:** `{best_prob:.2f}%`\n"
      f"- **Cosine Similarity:** `{best_sim:.4f}`"
    )
    return confidences, detail_md

  return predict_plate, class_names


def main() -> None:
  parser = argparse.ArgumentParser(description="Fungal Plate Few-Shot Classification Web Demo")
  parser.add_argument("--bundle", type=str, default=str(CKPT_DIR / "deployed_model_registry.pt"), help="Path to exported .pt bundle")
  parser.add_argument("--port", type=int, default=7860, help="Server port")
  parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
  args = parser.parse_args()

  predict_fn, class_names = load_inference_engine(Path(args.bundle))

  with gr.Blocks(title="Fungal Colony Plate Classifier (Few-Shot AI)") as demo:
    gr.Markdown(
      "# 🧫 Fungal Colony Plate Classifier (Few-Shot AI Demo)\n"
      "อัปโหลดภาพจานเพาะเชื้อ (Agar Plate) หรือกดไอคอนกล้องเพื่อถ่ายภาพสด ระบบจะเปรียบเทียบกับ Prototypes และแสดง **Top-3 Confidence Score**"
    )

    with gr.Row():
      with gr.Column(scale=1):
        # sources=["upload", "webcam"] ทำให้มีทั้งปุ่มอัปโหลดและกล้องถ่ายภาพในช่องเดียว
        input_img = gr.Image(
          type="pil",
          sources=["upload", "webcam"],
          label="ภาพจานเพาะเชื้อ (Upload หรือ ถ่ายจากกล้อง)",
          height=360,
        )
        with gr.Row():
          clear_btn = gr.ClearButton([input_img], value="ล้างภาพ")
          submit_btn = gr.Button("วิเคราะห์ชนิดเชื้อรา", variant="primary")

      with gr.Column(scale=1):
        top3_output = gr.Label(num_top_classes=3, label="Top-3 Confidence Score")
        summary_md = gr.Markdown("รอรับภาพจานเพาะเชื้อเพื่อเริ่มการวิเคราะห์...")

    gr.Markdown(f"**คลาสที่รองรับในระบบปัจจุบัน ({len(class_names)} คลาส):** {', '.join(class_names)}")

    # กดปุ่ม หรืออัปโหลด/ถ่ายภาพเสร็จ ให้ทำนายทันที
    submit_btn.click(fn=predict_fn, inputs=input_img, outputs=[top3_output, summary_md])
    input_img.change(fn=predict_fn, inputs=input_img, outputs=[top3_output, summary_md])

  demo.launch(server_port=args.port, share=args.share)

if __name__ == "__main__":
  main()