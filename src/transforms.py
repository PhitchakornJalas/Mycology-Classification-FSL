import torch
from torchvision import transforms

# Standard ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageTransformBuilder:
  """
  Builder class for creating image transformation pipelines separated by Train and Eval (Val/Test) modes.
  """

  def __init__(
    self,
    img_size: int = 84,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    use_augmentation: bool = True,
    use_normalize: bool = True,
  ):
    """
    Args:
      img_size: Target image size.
      mean: Per-channel mean values for normalization.
      std: Per-channel standard deviation values for normalization.
      use_augmentation: Whether to enable data augmentation in train mode.
      use_normalize: Whether to apply normalization.
    """
    self.img_size = img_size
    self.mean = mean
    self.std = std
    self.use_augmentation = use_augmentation
    self.use_normalize = use_normalize

  def get_train_transform(self) -> transforms.Compose:
    """Build image transformations for the training split."""
    ops = []

    if self.use_augmentation:
      ops.extend([
        transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
      ])
    else:
      ops.append(transforms.Resize((self.img_size, self.img_size)))

    ops.append(transforms.ToTensor())

    if self.use_normalize:
      ops.append(transforms.Normalize(mean=self.mean, std=self.std))

    return transforms.Compose(ops)

  def get_eval_transform(self) -> transforms.Compose:
    """Build deterministic image transformations for the validation and test splits."""
    resize_dim = int(self.img_size * 1.14)

    ops = [
      transforms.Resize(resize_dim),
      transforms.CenterCrop(self.img_size),
      transforms.ToTensor(),
    ]

    if self.use_normalize:
      ops.append(transforms.Normalize(mean=self.mean, std=self.std))

    return transforms.Compose(ops)

  def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
    """
    Revert a normalized tensor back to the [0, 1] range for visualization with matplotlib.
    Supports both 3D [C, H, W] and 4D [B, C, H, W] tensors.
    """
    if not self.use_normalize:
      return tensor.clamp(0, 1)

    mean = torch.tensor(self.mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(self.std, device=tensor.device).view(-1, 1, 1)

    if tensor.ndim == 4:
      mean = mean.unsqueeze(0)
      std = std.unsqueeze(0)

    return (tensor * std + mean).clamp(0, 1)