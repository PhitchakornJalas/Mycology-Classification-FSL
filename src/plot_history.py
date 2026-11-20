from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
PLOT_DIR = PROJECT_ROOT / "plots"


def load_history(stem: str) -> pd.DataFrame:
  """Load training history from CSV (or fallback to JSON) inside logs/ directory."""
  csv_path = LOG_DIR / f"history_{stem}.csv"
  json_path = LOG_DIR / f"history_{stem}.json"

  if csv_path.exists():
    return pd.read_csv(csv_path)
  if json_path.exists():
    return pd.read_json(json_path)

  raise FileNotFoundError(
    f"History log not found for '{stem}'. Expected:\n  - {csv_path}\n  - {json_path}"
  )


def plot_history(args) -> None:
  """Plot Train vs Val Loss and Accuracy curves and highlight the best checkpoint epoch."""
  stem = Path(args.filename).stem
  df = load_history(stem)

  if df.empty:
    raise ValueError(f"History file for '{stem}' is empty.")

  epochs = df["epoch"]

  best_idx = int(
    df.sort_values(by=["val_acc", "val_loss"], ascending=[False, True]).index[0]
  )
  best_row = df.loc[best_idx]
  best_epoch = int(best_row["epoch"])
  best_val_acc = float(best_row["val_acc"])
  best_val_loss = float(best_row["val_loss"])

  print("=" * 65)
  print(f"TRAINING HISTORY SUMMARY ({stem})")
  print("=" * 65)
  print(f"Total Epochs Logged : {len(df)}")
  print(f"Best Epoch          : {best_epoch:02d}")
  print(f"Best Val Accuracy   : {best_val_acc:.4f}")
  print(f"Best Val Loss       : {best_val_loss:.4f}")
  print(f"Total Train Time    : {df['elapsed_sec'].sum() / 60.0:.1f} mins")
  print("=" * 65)

  plt.style.use("seaborn-v0_8-whitegrid")
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5.5), dpi=150)

  # --- Subplot 1: SupCon Loss ---
  ax_loss = axes[0]
  ax_loss.plot(
    epochs,
    df["train_loss"],
    marker="o",
    linewidth=2.0,
    color="#1f77b4",
    label="Train Loss",
  )
  ax_loss.plot(
    epochs,
    df["val_loss"],
    marker="s",
    linewidth=2.0,
    color="#ff7f0e",
    label="Val Loss",
  )
  ax_loss.axvline(
    best_epoch,
    color="#2ca02c",
    linestyle="--",
    alpha=0.75,
    label=f"Best Epoch ({best_epoch})",
  )
  ax_loss.scatter(
    [best_epoch],
    [best_val_loss],
    color="#d62728",
    s=70,
    zorder=5,
  )
  ax_loss.annotate(
    f"Val Loss: {best_val_loss:.4f}",
    xy=(best_epoch, best_val_loss),
    xytext=(8, 10),
    textcoords="offset points",
    fontsize=9,
    fontweight="bold",
    color="#d62728",
  )
  ax_loss.set_title("Supervised Contrastive Loss", fontsize=12, fontweight="bold")
  ax_loss.set_xlabel("Epoch", fontsize=11)
  ax_loss.set_ylabel("Loss", fontsize=11)
  ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
  ax_loss.legend(loc="best", frameon=True)

  # --- Subplot 2: Leave-One-Out Nearest Prototype Accuracy ---
  ax_acc = axes[1]
  if "train_acc" in df.columns:
    ax_acc.plot(
      epochs,
      df["train_acc"],
      marker="o",
      linewidth=2.0,
      color="#1f77b4",
      label="Train Acc",
    )
  ax_acc.plot(
    epochs,
    df["val_acc"],
    marker="s",
    linewidth=2.0,
    color="#2ca02c",
    label="Val Acc",
  )
  ax_acc.axvline(
    best_epoch,
    color="#2ca02c",
    linestyle="--",
    alpha=0.75,
    label=f"Best Epoch ({best_epoch})",
  )
  ax_acc.scatter(
    [best_epoch],
    [best_val_acc],
    color="#d62728",
    s=70,
    zorder=5,
  )
  ax_acc.annotate(
    f"Best: {best_val_acc:.4f}",
    xy=(best_epoch, best_val_acc),
    xytext=(8, -14),
    textcoords="offset points",
    fontsize=9,
    fontweight="bold",
    color="#d62728",
  )
  ax_acc.set_title("Leave-One-Out Prototype Accuracy", fontsize=12, fontweight="bold")
  ax_acc.set_xlabel("Epoch", fontsize=11)
  ax_acc.set_ylabel("Accuracy (0.0 - 1.0)", fontsize=11)
  ax_acc.set_ylim(0.0, 1.05)
  ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
  ax_acc.legend(loc="best", frameon=True)

  fig.suptitle(
    f"Few-Shot Training History — {stem}",
    fontsize=14,
    fontweight="bold",
    y=1.02,
  )
  plt.tight_layout()

  if args.export:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / f"history_{stem}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"[*] Saved plot -> {out_path}")

  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Plot Few-Shot Training History Curves")
  parser.add_argument("--filename", type=str, default="registry.csv", help="Name of registry file .csv")
  parser.add_argument("--export", action="store_true", help="Enable if you want to create a plot to plots/history_<stem>.svg",)

  args = parser.parse_args()
  plot_history(args)