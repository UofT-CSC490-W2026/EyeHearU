"""
PyTorch Dataset for ASL sign images.

Supports loading from:
  - WLASL preprocessed frames
  - ASL Citizen preprocessed frames
  - Custom collected images

Each sample is a (image_tensor, label_index) tuple.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ASLImageDataset(Dataset):
    """
    Dataset of ASL sign images.

    Expected directory structure:
        processed_dir/
        ├── images/
        │   ├── hello/
        │   │   ├── 0001.jpg
        │   │   ├── 0002.jpg
        │   │   └── ...
        │   ├── goodbye/
        │   │   └── ...
        │   └── ...
        └── label_map.json   # {"hello": 0, "goodbye": 1, ...}
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = 224,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size

        # Load label map
        label_map_path = self.data_dir / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = {}

        # Collect all image paths and their labels
        self.samples = self._collect_samples()

        # Build transforms
        self.transform = self._build_transforms(augment)

    def _collect_samples(self) -> list[tuple[Path, int]]:
        """Walk the images directory and collect (path, label_idx) pairs."""
        images_dir = self.data_dir / "images" / self.split
        samples = []

        if not images_dir.exists():
            print(f"[WARNING] Image directory not found: {images_dir}")
            return samples

        for class_dir in sorted(images_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label_name = class_dir.name
            if label_name not in self.label_map:
                continue
            label_idx = self.label_map[label_name]
            for img_path in sorted(class_dir.glob("*.jpg")):
                samples.append((img_path, label_idx))
            for img_path in sorted(class_dir.glob("*.png")):
                samples.append((img_path, label_idx))

        print(f"[{self.split}] Loaded {len(samples)} samples across {len(self.label_map)} classes")
        return samples

    def _build_transforms(self, augment: bool):
        """Build image transforms for training or evaluation."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if augment:
            return transforms.Compose([
                transforms.Resize((self.image_size + 32, self.image_size + 32)),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.0),  # ASL signs are NOT symmetric — don't flip!
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label_idx
