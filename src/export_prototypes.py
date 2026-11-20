import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transforms import ImageTransformBuilder
from models import FewShotContrastivePipeline
from train import get_model_preprocess_params

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CKPT_DIR = PROJECT_ROOT / "checkpoints"


@torch.no_grad()
def export_deployed_bundle(args) -> None:
  stem = Path(args.filename).stem
  pool_path = DATA_DIR / f"pools_{stem}.json"
  index2label_path = DATA_DIR / f"index2label_{stem}.json"
  ckpt_path = Path(args.checkpoint) if args.checkpoint else (CKPT_DIR / f"best_head_{stem}.pth")
  export_path = Path(args.output) if args.output else (CKPT_DIR / f"deployed_model_{stem}.pt")

  if not pool_path.exists():
    raise FileNotFoundError(f"Pool JSON not found: {pool_path}")
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

  device = torch.device("cpu")
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

  model_id = ckpt.get("model_id", "google/embeddinggemma-2")
  hidden_dim = int(ckpt.get("hidden_dim", 512))
  out_dim = int(ckpt.get("out_dim", 256))
  temperature = float(ckpt.get("temperature", 0.07))

  img_size, mean, std = get_model_preprocess_params(model_id)

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
  pipeline.head.load_state_dict(ckpt["head_state_dict"])
  pipeline.eval()

  with open(pool_path, "r", encoding="utf-8") as f:
    pool_meta = json.load(f)

  index2label = {}
  if index2label_path.exists():
    with open(index2label_path, "r", encoding="utf-8") as f:
      raw_map = json.load(f)
      index2label = {str(k): str(v) for k, v in raw_map.items()}
    print(f"[*] Loaded label mapping from {index2label_path.name}")
  else:
    print(f"[!] Warning: {index2label_path.name} not found, falling back to numeric class IDs.")

  target_pool = (
    pool_meta["pools"][args.split]
    if "pools" in pool_meta and args.split in pool_meta["pools"]
    else pool_meta[args.split]
  )

  sorted_keys = sorted(target_pool.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
  class_names = []
  prototypes = []

  print(f"[*] Computing class prototypes from '{args.split}' pool in {pool_path.name}...")

  for cls_key in tqdm(sorted_keys, desc="Building Prototypes"):
    rel_paths = target_pool[cls_key]
    tensors = []
    for rel_p in rel_paths:
      with Image.open(PROJECT_ROOT / rel_p) as img:
        tensors.append(eval_tf(img.convert("RGB")))

    batch_tensor = torch.stack(tensors, dim=0)
    backbone_embs = pipeline.embedder(batch_tensor).to(device)
    z = pipeline.head(backbone_embs)  # [N_cls, D]

    proto = F.normalize(z.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
    prototypes.append(proto.cpu())

    species_name = index2label.get(str(cls_key), "").replace("_", " ")
    display_name = f"Class {cls_key}: {species_name}" if species_name else f"Class {cls_key}"
    class_names.append(display_name)

  proto_matrix = torch.stack(prototypes, dim=0)  # [Num_Classes, D]

  bundle = {
    "model_id": model_id,
    "img_size": img_size,
    "mean": mean,
    "std": std,
    "hidden_dim": hidden_dim,
    "out_dim": out_dim,
    "temperature": temperature,
    "head_state_dict": ckpt["head_state_dict"],
    "class_names": class_names,
    "prototypes": proto_matrix,
  }

  export_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(bundle, export_path)
  print(f"\n[+] Exported deployment bundle successfully -> {export_path}")
  for name in class_names:
    print(f"    - {name}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Export Model Head & Class Prototypes for Inference")
  parser.add_argument("--filename", type=str, default="registry.csv", help="Name of registry file .csv")
  parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_head_*.pth")
  parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Pool split to build prototypes")
  parser.add_argument("--output", type=str, default=None, help="Output .pt bundle path")
  args = parser.parse_args()
  export_deployed_bundle(args)