import json
import random
from pathlib import Path
from typing import Any, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset

class EpisodicDataset(Dataset):
  """PyTorch Dataset for sampling N-way K-shot episodes for few-shot learning."""

  def __init__(
    self,
    pool_json_path: str | Path,
    split: str,
    project_root: str | Path,
    n_way: int = 5,
    k_shot: int = 5,
    episodes_per_epoch: int = 500,
    transform: Callable[[Image.Image], torch.Tensor] | None = None,
    seed: int | None = None,
  ) -> None:
    """
    Args:
      pool_json_path: Path to the pools_*.json file.
      split: Dataset split, either "train", "val", or "test".
      project_root: Root directory of the project (joined with relative image filepaths).
      n_way: Number of classes per episode.
      k_shot: Number of support images per class.
      episodes_per_epoch: Number of episodes to sample per epoch.
      transform: Torchvision transforms to apply to the images.
      seed: If specified (e.g., for val/test), ensures deterministic episode sampling for consistent evaluation.

    Raises:
      ValueError: If fewer than `n_way` classes have at least `k_shot` images.
    """
    self.project_root = Path(project_root)
    self.split = split
    self.n_way = n_way
    self.k_shot = k_shot
    self.episodes_per_epoch = episodes_per_epoch
    self.transform = transform
    self.base_seed = seed

    with open(pool_json_path, "r") as f:
      all_pools = json.load(f)

    raw_pool = all_pools[split]

    self.pool = {
      cls_name: paths
      for cls_name, paths in raw_pool.items()
      if len(paths) >= k_shot
    }
    self.classes = sorted(self.pool.keys())

    if len(self.classes) < n_way:
      raise ValueError(
        f"[{split.upper()}] Only {len(self.classes)} classes have at least {k_shot} images "
        f"(insufficient for n_way={n_way})"
      )
    
  def __len__(self) -> int:
    """Return the total number of episodes sampled per epoch."""

    return self.episodes_per_epoch

  def __getitem__(self, idx: int) -> dict[str, Any]:
    """
    Sample a single N-way K-shot episode.

    Args:
      idx: Episode index

    Returns:
      A dictionary containing:
        - "images": Image tensor of shape [N*K, C, H, W].
        - "labels": Episode labels of shape [N*K] in range 0 .. N-1.
        - "class_names": List of sampled class names of length N.
    """

    rng = random.Random(self.base_seed + idx) if self.base_seed is not None else random
    sampled_classes = rng.sample(self.classes, self.n_way)

    images = []
    labels = []

    for episode_label, cls_name in enumerate(sampled_classes):
      sampled_paths = rng.sample(self.pool[cls_name], self.k_shot)

      for p in sampled_paths:
        images.append(self._load_image(p))
        labels.append(episode_label)

    return {
      "images": torch.stack(images),                    
      "labels": torch.tensor(labels, dtype=torch.long),
      "class_names": sampled_classes,
    }

  def _load_image(self, rel_path: str) -> torch.Tensor:
    """
    Load an image from disk, convert to RGB, and apply transformations.

    Args:
      rel_path: Image file path relative to `project_root`.

    Returns:
      Transformed image tensor of shape [C, H, W].
    """

    img_path = self.project_root / rel_path
    img = Image.open(img_path).convert("RGB")
    if self.transform:
      img = self.transform(img)
    return img