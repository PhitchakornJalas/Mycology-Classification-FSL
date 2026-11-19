import argparse
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import EpisodicDataset

def plot_episode(args) -> None:
  """Visualize sampled N-way K-shot episodes for both train and validation splits.

  Args:
    args: Parsed command-line arguments containing 'filename', 'n_way', 'k_shot', and 'export'.
  """
  stem = Path(args.filename).stem
  PROJECT_ROOT = Path(__file__).resolve().parent.parent
  POOL_PATH = PROJECT_ROOT / "data" / f"pools_{stem}.json"
  PLOT_PATH = PROJECT_ROOT / "plots"

  simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
  ])

  train_dataset = EpisodicDataset(
    pool_json_path=POOL_PATH,
    split="train",
    project_root=PROJECT_ROOT,
    n_way=args.n_way,
    k_shot=args.k_shot,
    episodes_per_epoch=500,
    transform=simple_transform,
    seed=None,
  )

  val_dataset = EpisodicDataset(
    pool_json_path=POOL_PATH,
    split="val",
    project_root=PROJECT_ROOT,
    n_way=args.n_way,
    k_shot=args.k_shot,
    episodes_per_epoch=100,
    transform=simple_transform,
    seed=224,
  )

  train_loader = DataLoader(train_dataset, batch_size=None, num_workers=2, pin_memory=False)
  val_loader = DataLoader(val_dataset, batch_size=None, num_workers=2, pin_memory=False)

  splits = [("train", train_loader), ("val", val_loader)]

  for split_name, loader in splits:
    episode = next(iter(loader))
    print(f"\n[{split_name.upper()}]")
    print("Classes in episode:", episode["class_names"])
    print("Images shape      :", episode["images"].shape)
    print("Labels            :", episode["labels"])

    images = episode["images"]            
    labels = episode["labels"]            
    class_names = episode["class_names"]

    n_way = len(class_names)
    k_shot = images.shape[0] // n_way

    fig, axes = plt.subplots(
      nrows=n_way,
      ncols=k_shot,
      figsize=(k_shot * 2.2, n_way * 2.4),
    )

    if n_way == 1:
      axes = axes[None, :]
    if k_shot == 1:
      axes = axes[:, None]

    for episode_label in range(n_way):
      cls_name = class_names[episode_label]
      cls_images = images[labels == episode_label]

      for s_idx in range(k_shot):
        img_tensor = cls_images[s_idx]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy().clip(0, 1)

        ax = axes[episode_label, s_idx]
        ax.imshow(img_np)
        ax.set_xticks([])
        ax.set_yticks([])

        if episode_label == 0:
          ax.set_title(f"Shot {s_idx + 1}", fontsize=10, fontweight="bold", color="#1f77b4")
        if s_idx == 0:
          ax.set_ylabel(f"Label {episode_label}\n({cls_name})", fontsize=10, fontweight="bold")

    plt.suptitle(
      f"[{split_name.upper()}] Episode Preview: {n_way}-Way {k_shot}-Shot",
      fontsize=13,
      fontweight="bold",
    )
    plt.tight_layout()

    if args.export:
      PLOT_PATH.mkdir(parents=True, exist_ok=True)
      save_path = PLOT_PATH / f"episode_{split_name}_{stem}.svg"
      plt.savefig(save_path, bbox_inches="tight")
      print(f"Saved episode preview to: {save_path}")

    plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Visualize N-Way K-Shot Images")
  parser.add_argument("--filename", type=str, default="registry.csv", help="Name of registry file .csv")
  parser.add_argument("--n_way", type=int, default=5, help="Number of classes per episode")
  parser.add_argument("--k_shot", type=int, default=5, help="Number of images per class per episode")
  parser.add_argument("--export", type=bool, default=False, help="Enable export=True to save episode preview plots")

  args = parser.parse_args()
  plot_episode(args)