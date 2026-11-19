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

  n = int(0.2 * len(df.index))
  remain, test = train_test_split(df, test_size=n, stratify=df["label"], random_state=args.seed)
  train, val = train_test_split(remain, test_size=n, stratify=remain["label"], random_state=args.seed)

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

  plt.show()

  if args.export:
    stem = Path(args.filename).stem

    train.to_csv(f"{DATA_DIR}/train_{stem}.csv", index=False)
    val.to_csv(f"{DATA_DIR}/val_{stem}.csv", index=False)
    test.to_csv(f"{DATA_DIR}/test_{stem}.csv", index=False)
    
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{PLOT_PATH}/classdist_{stem}.svg", format="svg")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Split Data")
  parser.add_argument("--filename", type=str, default="registry.csv", help="Name of registry file .csv")
  parser.add_argument("--seed", type=int, default=224, help="Random seed for reproducibility")
  parser.add_argument("--export", type=bool, default=False, help="Enable export=True if you want to create a new train/val/test")

  args = parser.parse_args()
  split_data(args)