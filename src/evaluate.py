import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import EpisodicDataset
from transforms import ImageTransformBuilder
from models import FewShotContrastivePipeline
from train import get_model_preprocess_params

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
PLOT_DIR = PROJECT_ROOT / "plots"


def compute_detailed_metrics(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  class_names: list[str],
) -> tuple[pd.DataFrame, dict[str, float], np.ndarray]:
  """Compute Confusion Matrix, Overall Accuracy, and Per-Class/Macro Sensitivity, Specificity, PPV, NPV, F1."""
  num_classes = len(class_names)
  labels_idx = list(range(num_classes))
  cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

  total_samples = int(cm.sum())
  overall_acc = float(np.trace(cm) / total_samples) if total_samples > 0 else 0.0

  rows = []
  for i, cls_name in enumerate(class_names):
    tp = int(cm[i, i])
    fn = int(cm[i, :].sum() - tp)
    fp = int(cm[:, i].sum() - tp)
    tn = int(total_samples - (tp + fn + fp))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = (2.0 * ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0

    rows.append(
      {
        "Class": cls_name,
        "Support": tp + fn,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Sensitivity (TPR)": sensitivity,
        "Specificity (TNR)": specificity,
        "PPV (Precision)": ppv,
        "NPV": npv,
        "F1 Score": f1,
      }
    )

  per_class_df = pd.DataFrame(rows)

  summary_metrics = {
    "Overall Episode Query Acc": overall_acc,
    "Macro Sensitivity (TPR)": float(per_class_df["Sensitivity (TPR)"].mean()),
    "Macro Specificity (TNR)": float(per_class_df["Specificity (TNR)"].mean()),
    "Macro PPV (Precision)": float(per_class_df["PPV (Precision)"].mean()),
    "Macro NPV": float(per_class_df["NPV"].mean()),
    "Macro F1 Score": float(per_class_df["F1 Score"].mean()),
  }

  return per_class_df, summary_metrics, cm


def plot_confusion_matrix(
  cm: np.ndarray,
  class_names: list[str],
  title: str,
  save_path: Path | None = None,
  show: bool = True,
) -> None:
  """Plot both Raw Count and Row-Normalized Confusion Matrices side-by-side."""
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), dpi=150)

  disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp_raw.plot(ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
  axes[0].set_title("Accumulated Episode Confusion Matrix (Counts)", fontsize=12, fontweight="bold")
  axes[0].tick_params(axis="x", rotation=30)

  row_sums = cm.sum(axis=1, keepdims=True)
  cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
  disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
  disp_norm.plot(ax=axes[1], cmap="Greens", colorbar=False, values_format=".2f")
  axes[1].set_title("Normalized Confusion Matrix (Row Recall)", fontsize=12, fontweight="bold")
  axes[1].tick_params(axis="x", rotation=30)

  fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
  plt.tight_layout()

  if save_path is not None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format=save_path.suffix.lstrip("."), bbox_inches="tight")
    print(f"[*] Saved confusion matrix plot -> {save_path}")

  if show:
    plt.show()
  else:
    plt.close(fig)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
  if args.k_shot < 2:
    raise ValueError("Episodic Leave-One-Out evaluation requires --k_shot >= 2.")

  stem = Path(args.filename).stem
  POOL_PATH = DATA_DIR / f"pools_{stem}.json"
  ckpt_path = Path(args.checkpoint) if args.checkpoint else (CKPT_DIR / f"best_head_{stem}.pth")

  if not POOL_PATH.exists():
    raise FileNotFoundError(f"Pool JSON not found: {POOL_PATH}")

  device = torch.device("cpu")
  if args.baseline:
    model_id = "google/embeddinggemma-2"
    img_size = 224
    hidden_dim = 512
    out_dim = 256
    temperature = 0.07
  else:
    if not ckpt_path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_id = ckpt.get("model_id", "google/embeddinggemma-2")
    img_size = int(ckpt.get("img_size", 224))
    hidden_dim = int(ckpt.get("hidden_dim", 512))
    out_dim = int(ckpt.get("out_dim", 256))
    temperature = float(ckpt.get("temperature", 0.07))

  img_size, mean, std = get_model_preprocess_params(model_id)
  print(f"Preprocess -> Size: {img_size}x{img_size} | Mean: {mean} | Std: {std}")

  tf_builder = ImageTransformBuilder(
    img_size=img_size,
    mean=mean,
    std=std,
    use_augmentation=False,
    use_normalize=True,
  )

  test_dataset = EpisodicDataset(
    pool_json_path=POOL_PATH,
    split="test",
    project_root=PROJECT_ROOT,
    n_way=args.n_way,
    k_shot=args.k_shot,
    episodes_per_epoch=args.test_episodes,
    transform=tf_builder.get_eval_transform(),
    seed=args.seed,
  )

  class_names = sorted(
    [str(c) for c in test_dataset.classes],
    key=lambda k: int(k) if k.isdigit() else k,
  )
  class_to_idx = {name: i for i, name in enumerate(class_names)}

  print("=" * 95)
  print(f"EPISODIC TEST EVALUATION ({stem}) | {args.n_way}-way {args.k_shot}-shot ({args.test_episodes} episodes)")
  if args.baseline:
    print(f"Checkpoint : None (Zero-Shot Raw Backbone: {model_id})")
  else:
    print(f"Checkpoint : {ckpt_path.name} (Epoch {ckpt.get('epoch', '?')} | Val Acc: {ckpt.get('val_acc', 0.0):.4f})")
  print("=" * 95)

  pipeline = FewShotContrastivePipeline(
    model_id=model_id,
    img_size=img_size,
    hidden_dim=hidden_dim,
    out_dim=out_dim,
    dropout=0.0,
    temperature=temperature,
  ).to(device)
  if not args.baseline:
    pipeline.head.load_state_dict(ckpt["head_state_dict"])
  pipeline.eval()

  episode_accs = []
  all_y_true = []
  all_y_pred = []

  for idx in tqdm(range(len(test_dataset)), desc="Evaluating Test Episodes", unit="ep"):
    batch = test_dataset[idx]
    images = batch["images"]                 # [N*K, C, H, W]
    sampled_classes = batch["class_names"]      # List of N global class names

    ep_global_labels = [class_to_idx[str(c)] for c in sampled_classes]

    backbone_embs: torch.Tensor = pipeline.embedder(images).to(device)
    if args.baseline:
      z: torch.Tensor = F.normalize(backbone_embs, p=2, dim=-1)
    else:
      z = pipeline.head(backbone_embs)

    # Reshape into [N_way, K_shot, D]
    z_grid = z.view(args.n_way, args.k_shot, -1)
    class_sums = z_grid.sum(dim=1, keepdim=True)  # [N, 1, D]

    correct = 0
    for local_c in range(args.n_way):
      global_c = ep_global_labels[local_c]
      for k in range(args.k_shot):
        query_vec = z_grid[local_c, k].unsqueeze(0)  # [1, D]
        protos = z_grid.mean(dim=1)  # [N, D]
        protos[local_c] = (class_sums[local_c, 0] - z_grid[local_c, k]) / (args.k_shot - 1)
        protos = F.normalize(protos, p=2, dim=-1)

        sim_scores = torch.matmul(query_vec, protos.T)  # [1, N]
        pred_local = int(sim_scores.argmax(dim=-1).item())
        pred_global = ep_global_labels[pred_local]

        all_y_true.append(global_c)
        all_y_pred.append(pred_global)
        if pred_local == local_c:
          correct += 1

    episode_accs.append(correct / (args.n_way * args.k_shot))

  acc_arr = np.array(episode_accs, dtype=float)
  mean_acc = float(acc_arr.mean())
  std_acc = float(acc_arr.std(ddof=1)) if len(acc_arr) > 1 else 0.0
  ci95 = 1.96 * (std_acc / np.sqrt(len(acc_arr))) if len(acc_arr) > 0 else 0.0

  per_class_df, summary_metrics, cm = compute_detailed_metrics(
    np.array(all_y_true), np.array(all_y_pred), class_names
  )

  print("\n" + "=" * 95)
  print("EPISODIC TEST SUMMARY")
  print("=" * 95)
  print(
    f"  {args.n_way}-way {args.k_shot}-shot Accuracy    : "
    f"{mean_acc:.4f} ± {ci95:.4f} ({mean_acc * 100:.2f}% ± {ci95 * 100:.2f}%)"
  )
  for k, v in summary_metrics.items():
    print(f"  {k:<28}: {v:.4f} ({v * 100:.2f}%)")

  print("\n" + "=" * 95)
  print("PER-CLASS DIAGNOSTIC METRICS (Accumulated Across Episodes)")
  print("=" * 95)
  disp_df = per_class_df.copy()
  for col in ["Sensitivity (TPR)", "Specificity (TNR)", "PPV (Precision)", "NPV", "F1 Score"]:
    disp_df[col] = disp_df[col].map(lambda x: f"{x:.4f}")
  print(disp_df.to_string(index=False))
  print("=" * 95)

  if args.export:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    out_suffix = f"{stem}_baseline" if args.baseline else stem
    metrics_csv = LOG_DIR / f"eval_per_class_{out_suffix}.csv"
    summary_json = LOG_DIR / f"eval_summary_{out_suffix}.json"
    cm_plot_path = PLOT_DIR / f"cm_{out_suffix}.svg"

    per_class_df.to_csv(metrics_csv, index=False)
    payload = {
      "pool_source": str(POOL_PATH),
      "checkpoint": None if args.baseline else str(ckpt_path),
      "n_way": args.n_way,
      "k_shot": args.k_shot,
      "test_episodes": args.test_episodes,
      "episodic_accuracy": round(mean_acc, 4),
      "ci95": round(ci95, 4),
      "std_acc": round(std_acc, 4),
      "summary_metrics": {k: round(v, 4) for k, v in summary_metrics.items()},
      "confusion_matrix": cm.tolist(),
      "class_names": class_names,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2)

    print(f"[*] Saved per-class metrics -> {metrics_csv}")
    print(f"[*] Saved summary JSON      -> {summary_json}")
  else:
    cm_plot_path = None

  title_str = (
    f"Episodic Test ({args.n_way}-way {args.k_shot}-shot, {args.test_episodes} eps) — "
    f"Acc: {mean_acc:.4f} ± {ci95:.4f}"
  )
  plot_confusion_matrix(
    cm=cm,
    class_names=class_names,
    title=title_str,
    save_path=cm_plot_path,
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Episodic Few-Shot Evaluation on Test Split")
  parser.add_argument("--filename", type=str, default="registry.csv", help="Name of registry file .csv")
  parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint")
  parser.add_argument("--baseline", action="store_true", help="Evaluate raw frozen backbone without trained projection head")
  parser.add_argument("--n_way", type=int, default=3, help="Number of classes per test episode")
  parser.add_argument("--k_shot", type=int, default=2, help="Number of shots per class (must be >= 2)")
  parser.add_argument("--test_episodes", type=int, default=60, help="Number of test episodes to evaluate")
  parser.add_argument("--seed", type=int, default=224, help="Random seed for EpisodicDataset")
  parser.add_argument("--export", action="store_true", help="Enable to Export metrics CSV/JSON and Confusion Matrix plot")

  args = parser.parse_args()
  evaluate(args)