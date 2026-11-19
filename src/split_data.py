from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOT_PATH = PROJECT_ROOT / "plots"

def split_data(args):
  """
  Split registry data into stratified train, validation, and test sets, display distributions, and optionally export results.

  Args:
    args: Parsed command-line arguments containing 'filename', 'seed', and 'export'.

  Raises:
    FileNotFoundError: If the specified registry CSV file does not exist.
  """
  
  csv_path = DATA_DIR / f"{args.filename}"
  if not csv_path.exists():
    raise FileNotFoundError(f"Not Found {csv_path} (please run build_registry and --export True)")
  df = pd.read_csv(csv_path)

  train_parts = []
  val_parts = []
  test_parts = []

  for label, group in df.groupby("label", sort=True):
    n_cls = len(group)
    if n_cls < args.min_samples * 3:
      raise ValueError(f"Label '{label}' has only {n_cls} samples (need >= {args.min_samples * 3})")

    n_test = max(args.min_samples, round(n_cls * 0.2))
    n_val = max(args.min_samples, round(n_cls * 0.2))

    if n_cls - n_test - n_val < args.min_samples:
      n_val = (n_cls - args.min_samples) // 2
      n_test = n_cls - args.min_samples - n_val

    remain, test_g = train_test_split(group, test_size=n_test, random_state=args.seed)
    train_g, val_g = train_test_split(remain, test_size=n_val, random_state=args.seed)

    train_parts.append(train_g)
    val_parts.append(val_g)
    test_parts.append(test_g)

  train: pd.DataFrame = pd.concat(train_parts, ignore_index=True)
  val: pd.DataFrame = pd.concat(val_parts, ignore_index=True)
  test: pd.DataFrame = pd.concat(test_parts, ignore_index=True)

  train_dist = train.groupby("label").size().reset_index(name="#")
  train_dist["%"] = train_dist["#"] / train_dist["#"].sum()
  val_dist = val.groupby("label").size().reset_index(name="#")
  val_dist["%"] = val_dist["#"] / val_dist["#"].sum()
  test_dist = test.groupby("label").size().reset_index(name="#")
  test_dist["%"] = test_dist["#"] / test_dist["#"].sum()

  summary = (
    train_dist.rename(columns={"#": "Train (#)", "%": "Train (%)"})
    .merge(val_dist.rename(columns={"#": "Val (#)", "%": "Val (%)"}), on="label", how="outer")
    .merge(test_dist.rename(columns={"#": "Test (#)", "%": "Test (%)"}), on="label", how="outer")
    .fillna(0)
  )

  summary["Total (#)"] = summary[["Train (#)", "Val (#)", "Test (#)"]].sum(axis=1).astype(int)

  for col in ["Train (%)", "Val (%)", "Test (%)"]:
    summary[col] = summary[col].map("{:.2%}".format)

  for col in ["Train (#)", "Val (#)", "Test (#)"]:
    summary[col] = summary[col].astype(int)

  print("=" * 75)
  print("CLASS DISTRIBUTION SUMMARY")
  print("=" * 75)
  print(summary.to_string(index=False))
  print("-" * 75)
  print(f"Total Images -> Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,} | All: {len(train)+len(val)+len(test):,}")
  print("=" * 75)

  fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

  ax[0].bar(train_dist["label"], train_dist["%"])
  ax[0].set_ylim(0, 1)
  ax[0].set_title("Training Data")

  ax[1].bar(val_dist["label"], val_dist["%"])
  ax[1].set_ylim(0, 1)
  ax[1].set_title("Validation Data")

  ax[2].bar(test_dist["label"], test_dist["%"])
  ax[2].set_ylim(0, 1)
  ax[2].set_title("Test Data")

  if args.export:
    stem = Path(args.filename).stem

    train.to_csv(f"{DATA_DIR}/train_{stem}.csv", index=False)
    val.to_csv(f"{DATA_DIR}/val_{stem}.csv", index=False)
    test.to_csv(f"{DATA_DIR}/test_{stem}.csv", index=False)
    
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{PLOT_PATH}/classdist_{stem}.svg", format="svg")

  plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Split Data")
  parser.add_argument("--filename", type=str, default="registry.csv", help="Name of registry file .csv")
  parser.add_argument("--min_samples", type=int, default=2, help="Minimum samples per class in each split")
  parser.add_argument("--seed", type=int, default=224, help="Random seed for reproducibility")
  parser.add_argument("--export", action="store_true", help="Enable if you want to create a new train/val/test")

  args = parser.parse_args()
  split_data(args)