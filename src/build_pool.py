from pathlib import Path
import pandas as pd
import json
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def df_to_pool(df: pd.DataFrame) -> dict[str, list[str]]:
  """
  Convert a split DataFrame into a mapping from class label to image file paths.

  Args:
    df: DataFrame containing at least 'label' and 'img_filepath' columns.

  Returns:
    A dictionary mapping each class label to a list of image file paths.
  """
  
  return (
    df.groupby("label")["img_filepath"]
    .apply(list)
    .to_dict()
  )

def build_pools(args):
  """
  Build episodic data pools for train, val, and test splits from CSV files.

  Args:
    args: Parsed command-line arguments containing 'filename' and 'export'.

  Raises:
    FileNotFoundError: If any expected split CSV file does not exist.
  """

  stem = Path(args.filename).stem
  splits = ["train", "val", "test"]
  pools = {}

  print("=" * 65)
  print(f"BUILDING EPISODIC POOLS FROM: {stem}")
  print("=" * 65)

  for split in splits:
    csv_path = DATA_DIR / f"{split}_{stem}.csv"
    if not csv_path.exists():
      raise FileNotFoundError(f"Not Found {csv_path} (please run split_data and --export True)")

    df = pd.read_csv(csv_path)
    pool = df_to_pool(df)
    pools[split] = pool

    counts = [len(paths) for paths in pool.values()]
    print(
      f"{split.upper():<6} | Classes: {len(pool):>3} | "
      f"Total Imgs: {len(df):>6,} | "
      f"Min/Class: {min(counts):>4} | Max/Class: {max(counts):>4}"
    )

  print("=" * 65)

  if args.export:
    output_path = DATA_DIR / f"pools_{stem}.json"
    with open(output_path, "w") as f:
      json.dump(pools, f, indent=2)
    print(f"Saved pools to: {output_path}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Build Episodic Data Pools from Split CSVs")
  parser.add_argument("--filename", type=str, default="registry.csv", help="Name of registry file .csv")
  parser.add_argument("--export", type=bool, default=False, help="Enable export=True if you want to create a new registry")

  args = parser.parse_args()
  build_pools(args)