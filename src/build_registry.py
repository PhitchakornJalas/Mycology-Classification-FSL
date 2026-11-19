from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def build_registry(args):
  REGISTRY_PATH = DATA_DIR / args.filename
  REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

  dataset_dir = DATA_DIR / args.dataset
  if not dataset_dir.exists():
    raise FileNotFoundError(f"Not Found {dataset_dir} (please download dataset)")
  
  image_files = sorted(dataset_dir.glob("*/*.*"))

  if not image_files:
    raise FileNotFoundError(f"No image files found inside {dataset_dir}")

  records = []
  for img_path in tqdm(image_files, desc="Building Registry"):
    label = img_path.parts[-2]

    records.append({
      "img_filepath": str(img_path.relative_to(PROJECT_ROOT)),
      "label": label,
    })

  unique_labels = sorted({r["label"] for r in records if "label" in r})
  index2label = dict(zip(range(len(unique_labels)), unique_labels))
  label2index = dict(zip(unique_labels, range(len(unique_labels))))

  for r in records:
    r["label"] = label2index[r["label"]]

  df = pd.DataFrame(records)
  if args.export:
    stem = Path(args.filename).stem

    df.to_csv(REGISTRY_PATH, index=False)
    with open(f"{DATA_DIR}/label2index_{stem}.json", "w") as f:
      json.dump(label2index, f)
    with open(f"{DATA_DIR}/index2label_{stem}.json", "w") as f:
      json.dump(index2label, f)
    print(f"\n[SAVED] {REGISTRY_PATH}")
  
  print(f"({len(df)} rows)")
  print(df["label"].value_counts())
  print(f"index2label: {index2label}")
  print(f"label2index: {label2index}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Build Regitry File")
  parser.add_argument("--dataset", type=str, default="", help="Name of dataset folder")
  parser.add_argument("--filename", type=str, default="registry.csv", help="Name of registry file .csv")
  parser.add_argument("--export", type=bool, default=False, help="Enable export=True if you want to create a new registry")

  args = parser.parse_args()
  build_registry(args)