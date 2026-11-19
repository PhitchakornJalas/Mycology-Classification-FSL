import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from tqdm import tqdm

from dataset import EpisodicDataset
from transforms import ImageTransformBuilder
from models import FewShotContrastivePipeline


def get_model_preprocess_params(
  model_id: str,
  default_size: int = 224,
) -> tuple[int, tuple[float, float, float], tuple[float, float, float]]:
  """Extract image size, mean, and std from the Hugging Face processor."""
  processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
  img_proc = getattr(processor, "image_processor", processor)

  img_size = None
  for attr in ("size", "crop_size"):
    val = getattr(img_proc, attr, None)
    if isinstance(val, int):
      img_size = val
      break
    elif isinstance(val, dict):
      img_size = val.get("height") or val.get("shortest_edge") or val.get("longest_edge")
      if img_size is not None:
        break

  if img_size is None:
    img_size = default_size

  raw_mean = getattr(img_proc, "image_mean", (0.5, 0.5, 0.5))
  raw_std = getattr(img_proc, "image_std", (0.5, 0.5, 0.5))
  mean = (float(raw_mean[0]), float(raw_mean[1]), float(raw_mean[2]))
  std = (float(raw_std[0]), float(raw_std[1]), float(raw_std[2]))
  return int(img_size), mean, std


def save_history(history: list[dict[str, int | float]], json_path: Path, csv_path: Path) -> None:
  """Save training history logs to both JSON and CSV files."""
  json_path.parent.mkdir(parents=True, exist_ok=True)
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

  if history:
    fieldnames = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(history)

@torch.no_grad()
def compute_loo_accuracy(
  pipeline: FewShotContrastivePipeline,
  embeddings: torch.Tensor,
  n_way: int,
  k_shot: int,
) -> float:
  """Compute Leave-One-Out Nearest Prototype Accuracy in [0, 1] for an episode."""
  z_grid = embeddings.detach().view(n_way, k_shot, -1)  # [N, K, D]
  class_sums = z_grid.sum(dim=1, keepdim=True)          # [N, 1, D]

  correct = 0
  for c in range(n_way):
    for k in range(k_shot):
      query_vec = z_grid[c, k].unsqueeze(0)             # [1, D]
      protos = z_grid.mean(dim=1)                       # [N, D]
      protos[c] = (class_sums[c, 0] - z_grid[c, k]) / (k_shot - 1)
      protos = F.normalize(protos, p=2, dim=-1)

      sim_scores = pipeline.similarity(query_vec, protos, scale=False)
      pred = int(sim_scores.argmax(dim=-1).item())
      if pred == c:
        correct += 1

  return correct / (n_way * k_shot)


def train_one_epoch(
  pipeline: FewShotContrastivePipeline,
  loader: DataLoader[dict[str, torch.Tensor | list[str]]],
  optimizer: torch.optim.Optimizer,
  n_way: int,
  k_shot: int,
  device: torch.device,
  epoch: int,
  total_epochs: int,
) -> tuple[float, float]:
  """Run one online training epoch and return (avg_train_loss, avg_train_acc)."""
  pipeline.train()
  total_loss = 0.0
  total_acc = 0.0
  num_episodes = len(loader)

  pbar = tqdm(
    enumerate(loader, start=1),
    total=num_episodes,
    desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Train",
    unit="ep",
  )

  for step, episode in pbar:
    images = episode["images"]
    labels = episode["labels"].to(device)

    optimizer.zero_grad()
    outputs = pipeline(images, labels)
    loss = outputs["loss"]

    loss.backward()
    optimizer.step()

    acc = compute_loo_accuracy(pipeline, outputs["embeddings"], n_way, k_shot)
    total_loss += loss.item()
    total_acc += acc

    pbar.set_postfix(
      loss=f"{total_loss / step:.4f}",
      acc=f"{total_acc / step:.4f}",
    )

  return total_loss / num_episodes, total_acc / num_episodes


@torch.no_grad()
def validate(
  pipeline: FewShotContrastivePipeline,
  loader: DataLoader[dict[str, torch.Tensor | list[str]]],
  n_way: int,
  k_shot: int,
  device: torch.device,
  epoch: int,
  total_epochs: int,
) -> tuple[float, float]:
  """Evaluate validation SupConLoss and Leave-One-Out Nearest Prototype Accuracy in [0, 1]."""
  pipeline.eval()
  total_loss = 0.0
  total_acc = 0.0
  num_episodes = len(loader)

  pbar = tqdm(
    enumerate(loader, start=1),
    total=num_episodes,
    desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Val  ",
    unit="ep",
  )

  for step, episode in pbar:
    images = episode["images"]
    labels = episode["labels"].to(device)

    outputs = pipeline(images, labels)
    acc = compute_loo_accuracy(pipeline, outputs["embeddings"], n_way, k_shot)

    total_loss += outputs["loss"].item()
    total_acc += acc

    pbar.set_postfix(
      val_loss=f"{total_loss / step:.4f}",
      val_acc=f"{total_acc / step:.4f}",
    )

  return total_loss / num_episodes, total_acc / num_episodes


def main(args: argparse.Namespace) -> None:
  if args.k_shot < 2:
    raise ValueError("SupConLoss and Leave-One-Out validation require k_shot >= 2.")

  stem = Path(args.filename).stem
  PROJECT_ROOT = Path(__file__).resolve().parent.parent
  POOL_PATH = PROJECT_ROOT / "data" / f"pools_{stem}.json"
  SAVE_DIR = PROJECT_ROOT / "checkpoints"
  LOG_DIR = PROJECT_ROOT / "logs"
  SAVE_DIR.mkdir(parents=True, exist_ok=True)
  LOG_DIR.mkdir(parents=True, exist_ok=True)

  history_json_path = LOG_DIR / f"history_{stem}.json"
  history_csv_path = LOG_DIR / f"history_{stem}.csv"

  device = torch.device("cpu")
  print(f"Using device for trainable stages (2-4): {device}")

  # 1. Load model preprocessing parameters & build Petri-dish transforms
  img_size, mean, std = get_model_preprocess_params(args.model_id)
  print(f"Preprocess -> Size: {img_size}x{img_size} | Mean: {mean} | Std: {std}")

  tf_builder = ImageTransformBuilder(
    img_size=img_size,
    mean=mean,
    std=std,
    use_augmentation=True,
    use_normalize=True,
  )

  # 2. Build Online Episodic Datasets
  train_dataset = EpisodicDataset(
    pool_json_path=POOL_PATH,
    split="train",
    project_root=PROJECT_ROOT,
    n_way=args.n_way,
    k_shot=args.k_shot,
    episodes_per_epoch=args.train_episodes,
    transform=tf_builder.get_train_transform(),
    seed=None,
  )

  val_dataset = EpisodicDataset(
    pool_json_path=POOL_PATH,
    split="val",
    project_root=PROJECT_ROOT,
    n_way=args.n_way,
    k_shot=args.k_shot,
    episodes_per_epoch=args.val_episodes,
    transform=tf_builder.get_eval_transform(),
    seed=args.seed,
  )

  train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
  val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

  # 3. Initialize 4-Stage Pipeline & Optimizer
  pipeline = FewShotContrastivePipeline(
    model_id=args.model_id,
    img_size=img_size,
    hidden_dim=args.hidden_dim,
    out_dim=args.out_dim,
    dropout=args.dropout,
    temperature=args.temperature,
  ).to(device)

  optimizer = torch.optim.AdamW(
    pipeline.head.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
  )

  # 4. Online Training Loop with History Logging & Early Stopping
  best_val_loss = float("inf")
  best_val_acc = 0.0
  epochs_without_improve = 0
  history = []

  print(f"\nStarting Online Training ({args.n_way}-way {args.k_shot}-shot | EarlyStopping patience={args.patience})...")

  for epoch in range(1, args.epochs + 1):
    start_time = time.time()
    current_lr = scheduler.get_last_lr()[0]
    print(f"\nEpoch [{epoch:02d}/{args.epochs:02d}] (LR: {current_lr:.6f})")

    train_loss, train_acc = train_one_epoch(
      pipeline, train_loader, optimizer, args.n_way, args.k_shot, device, epoch, args.epochs
    )
    val_loss, val_acc = validate(
      pipeline, val_loader, args.n_way, args.k_shot, device, epoch, args.epochs
    )
    scheduler.step()

    elapsed = time.time() - start_time

    # Record epoch metrics
    epoch_record = {
      "epoch": epoch,
      "lr": round(current_lr, 8),
      "train_loss": round(train_loss, 4),
      "train_acc": round(train_acc, 4),
      "val_loss": round(val_loss, 4),
      "val_acc": round(val_acc, 4),
      "elapsed_sec": round(elapsed, 2),
    }
    history.append(epoch_record)
    save_history(history, history_json_path, history_csv_path)

    print(
      f"--> Epoch {epoch:02d} Summary | "
      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
      f"Time: {elapsed:.1f}s"
    )

    # Save best model checkpoint (prioritize higher val_acc, tie-break with lower val_loss)
    is_best_acc: bool = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
    if is_best_acc:
      best_val_acc = val_acc
      save_path: Path = SAVE_DIR / f"best_head_{stem}.pth"
      torch.save(
        {
          "epoch": epoch,
          "val_acc": round(best_val_acc, 4),
          "val_loss": round(val_loss, 4),
          "model_id": args.model_id,
          "img_size": img_size,
          "in_dim": pipeline.embedder.embed_dim,
          "hidden_dim": args.hidden_dim,
          "out_dim": args.out_dim,
          "temperature": args.temperature,
          "head_state_dict": pipeline.head.state_dict(),
        },
        save_path,
      )
      print(f"    [*] Saved new best projection head -> {save_path} (Val Acc: {best_val_acc:.4f}, Val Loss: {val_loss:.4f})")

    # Early Stopping check monitored on val_loss
    if val_loss < best_val_loss - args.min_delta:
      best_val_loss = val_loss
      epochs_without_improve = 0
    else:
      epochs_without_improve += 1
      print(f"    [!] No val_loss improvement for {epochs_without_improve}/{args.patience} epoch(s) (Best Val Loss: {best_val_loss:.4f})")

    if args.patience > 0 and epochs_without_improve >= args.patience:
      print(f"\n[Early Stopping] Stopped at epoch {epoch:02d} because val_loss did not improve for {args.patience} epochs.")
      break

  print(f"\nTraining finished! Best Val Acc: {best_val_acc:.4f} | Best Val Loss: {best_val_loss:.4f}")
  print(f"Logs saved to:\n  - {history_json_path}\n  - {history_csv_path}")


if __name__ == "__main__":
  parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Online Few-Shot Contrastive Training")
  parser.add_argument("--filename", type=str, default="registry.csv", help="Name of registry file .csv")
  parser.add_argument("--model_id", type=str, default="google/embeddinggemma-2", help="Hugging Face model ID")
  parser.add_argument("--n_way", type=int, default=5, help="Number of classes per episode")
  parser.add_argument("--k_shot", type=int, default=5, help="Number of images per class (must be >= 2)")
  parser.add_argument("--train_episodes", type=int, default=100, help="Episodes per training epoch")
  parser.add_argument("--val_episodes", type=int, default=50, help="Episodes per validation epoch")
  parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
  parser.add_argument("--patience", type=int, default=5, help="Early stopping patience epochs (0 to disable)")
  parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum val_loss decrease to count as improvement")
  parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
  parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
  parser.add_argument("--hidden_dim", type=int, default=512, help="Projection head hidden dimension")
  parser.add_argument("--out_dim", type=int, default=256, help="Output embedding dimension")
  parser.add_argument("--dropout", type=float, default=0.1, help="Projection head dropout rate")
  parser.add_argument("--temperature", type=float, default=0.07, help="Cosine similarity temperature")
  parser.add_argument("--seed", type=int, default=224, help="Validation seed")

  args: argparse.Namespace = parser.parse_args()
  main(args)