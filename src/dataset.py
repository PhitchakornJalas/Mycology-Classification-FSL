import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class EpisodicDataset(Dataset):
  def __init__(
    self,
    pool_json_path,
    split,
    project_root,
    n_way = 5,
    k_shot = 5,
    q_query = 15,
    episodes_per_epoch = 500,
    transform = None,
    seed = None
  ):
    """
    Args:
      pool_json_path: Path to the pools_*.json file.
      split: Dataset split, either "train", "val", or "test".
      project_root: Root directory of the project (joined with relative image filepaths).
      n_way: Number of classes per episode.
      k_shot: Number of support images per class.
      q_query: Number of query images per class.
      episodes_per_epoch: Number of episodes to sample per epoch.
      transform: Torchvision transforms to apply to the images.
      seed: If specified (e.g., for val/test), ensures deterministic episode sampling for consistent evaluation.
    """
    self.project_root = Path(project_root)
    self.split = split
    self.n_way = n_way
    self.k_shot = k_shot
    self.q_query = q_query
    self.episodes_per_epoch = episodes_per_epoch
    self.transform = transform
    self.base_seed = seed

    with open(pool_json_path, "r") as f:
      all_pools = json.load(f)

    raw_pool = all_pools[split]

    min_required = k_shot + q_query
    self.pool = {
      cls_name: paths
      for cls_name, paths in raw_pool.items()
      if len(paths) >= min_required
    }
    self.classes = sorted(self.pool.keys())

    if len(self.classes) < n_way:
      raise ValueError(
        f"[{split.upper()}] Only {len(self.classes)} classes have at least {min_required} images"
        f"(insufficient for n_way={n_way})"
      )
    
  def __len__(self):
    return self.episodes_per_epoch

  def __getitem__(self, idx):
    rng = random.Random(self.base_seed + idx) if self.base_seed is not None else random

    sampled_classes = rng.sample(self.classes, self.n_way)

    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for episode_label, cls_name in enumerate(sampled_classes):
      sampled_paths = rng.sample(self.pool[cls_name], self.k_shot + self.q_query)

      supp_paths = sampled_paths[: self.k_shot]
      quer_paths = sampled_paths[self.k_shot :]

      for p in supp_paths:
        support_images.append(self._load_image(p))
        support_labels.append(episode_label)

      for p in quer_paths:
        query_images.append(self._load_image(p))
        query_labels.append(episode_label)

    return {
      "support_x": torch.stack(support_images),                    # [N*K, C, H, W]
      "support_y": torch.tensor(support_labels, dtype=torch.long), # [N*K]
      "query_x": torch.stack(query_images),                        # [N*Q, C, H, W]
      "query_y": torch.tensor(query_labels, dtype=torch.long),     # [N*Q]
      "class_names": sampled_classes,                            
    }

  def _load_image(self, rel_path):
    img_path = self.project_root / rel_path
    img = Image.open(img_path).convert("RGB")
    if self.transform:
      img = self.transform(img)
    return img

def plot_episode(episode, max_query_show=5, save_path=None):
  support_x = episode["support_x"]       # [N*K, C, H, W]
  query_x = episode["query_x"]           # [N*Q, C, H, W]
  class_names = episode["class_names"]   # list ของชื่อคลาสยาว N

  n_way = len(class_names)
  k_shot = support_x.shape[0] // n_way
  q_query = query_x.shape[0] // n_way
  q_show = min(q_query, max_query_show)  # จำกัดจำนวนรูป Query ที่โชว์ไม่ให้กว้างล้นจอ

  sorted_order = sorted(range(n_way), key=lambda i: class_names[i])

  total_cols = k_shot + q_show
  fig, axes = plt.subplots(
    nrows=n_way,
    ncols=total_cols,
    figsize=(total_cols * 2.2, n_way * 2.4),
  )

  # ป้องกันกรณี n_way=1 หรือ total_cols=1 ที่ axes จะไม่เป็น 2D array
  if n_way == 1:
    axes = axes[None, :]
  if total_cols == 1:
    axes = axes[:, None]

  for row_idx, cls_name in enumerate(sorted_order):
    # 1. วาดรูปฝั่ง Support Set
    for s_idx in range(k_shot):
      img_tensor = support_x[row_idx * k_shot + s_idx]
      # แปลง Tensor [C, H, W] -> Numpy [H, W, C] และ clip ค่าให้อยู่ในช่วง [0, 1]
      img_np = img_tensor.permute(1, 2, 0).cpu().numpy().clip(0, 1)

      ax = axes[row_idx, s_idx]
      ax.imshow(img_np)
      ax.set_xticks([])
      ax.set_yticks([])

      if row_idx == 0:
        ax.set_title(f"Support {s_idx + 1}", fontsize=10, fontweight="bold", color="#1f77b4")
      if s_idx == 0:
        ax.set_ylabel(f"Label {row_idx}\n({cls_name})", fontsize=10, fontweight="bold")

    # 2. วาดรูปฝั่ง Query Set (เลือกมาโชว์ q_show รูปแรก)
    for q_idx in range(q_show):
      col_idx = k_shot + q_idx
      img_tensor = query_x[row_idx * q_query + q_idx]
      img_np = img_tensor.permute(1, 2, 0).cpu().numpy().clip(0, 1)

      ax = axes[row_idx, col_idx]
      ax.imshow(img_np)
      ax.set_xticks([])
      ax.set_yticks([])

      # ใส่กรอบสีส้มบางๆ ให้แยกฝั่ง Query ออกจาก Support ชัดเจน
      for spine in ax.spines.values():
        spine.set_edgecolor("#ff7f0e")
        spine.set_linewidth(1.5)

      if row_idx == 0:
        ax.set_title(f"Query {q_idx + 1}", fontsize=10, fontweight="bold", color="#ff7f0e")

  plt.suptitle(
    f"Episode Preview: {n_way}-Way {k_shot}-Shot (Showing {q_show}/{q_query} Queries)",
    fontsize=13,
    fontweight="bold",
  )
  plt.tight_layout()

  if save_path:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved episode preview to: {save_path}")

  plt.show()

if __name__ == "__main__":
  PROJECT_ROOT = Path(__file__).resolve().parent.parent
  POOL_PATH = PROJECT_ROOT / "data" / "pools_registry.json"

  train_dataset = EpisodicDataset(
    pool_json_path=POOL_PATH,
    split="train",
    project_root=PROJECT_ROOT,
    n_way=3,
    k_shot=5,
    q_query=15,
    episodes_per_epoch=500,
    transform=transforms.ToTensor(),
    seed=None,
  )

  val_dataset = EpisodicDataset(
    pool_json_path=POOL_PATH,
    split="val",
    project_root=PROJECT_ROOT,
    n_way=3,
    k_shot=5,
    q_query=15,
    episodes_per_epoch=100,
    transform=transforms.ToTensor(),
    seed=224,
  )

  train_loader = DataLoader(train_dataset, batch_size=None, num_workers=2, pin_memory=False)
  val_loader = DataLoader(val_dataset, batch_size=None, num_workers=2, pin_memory=False)

  episode = next(iter(train_loader))
  print("Classes in episode:", episode["class_names"])
  print("Support X shape   :", episode["support_x"].shape)
  print("Support Y         :", episode["support_y"])     
  print("Query X shape     :", episode["query_x"].shape)  
  print("Query Y shape     :", episode["query_y"].shape)

  plot_episode(episode, max_query_show=5)