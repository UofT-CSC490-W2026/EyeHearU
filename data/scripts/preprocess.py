"""
Preprocessing utilities for ASL datasets.

This script handles:
  - Resizing images to consistent dimensions
  - Hand region cropping using MediaPipe Hands
  - Train/val/test splitting
  - Dataset statistics and validation
"""

import json
import random
import shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[WARNING] MediaPipe not installed. Hand cropping will be skipped.")


PROCESSED_DIR = Path(__file__).parent.parent / "processed"
IMAGE_SIZE = 224


def crop_hand_region(image: np.ndarray, padding: float = 0.2) -> np.ndarray | None:
    """
    Use MediaPipe Hands to detect and crop the hand region from an image.

    Args:
        image: BGR image (OpenCV format)
        padding: Fraction of extra space around the detected hand

    Returns:
        Cropped hand region, or None if no hand detected.
    """
    if not HAS_MEDIAPIPE:
        return None

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        # Get bounding box of the first detected hand
        h, w, _ = image.shape
        landmarks = results.multi_hand_landmarks[0]
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Add padding
        pad_x = int((x_max - x_min) * padding)
        pad_y = int((y_max - y_min) * padding)
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        return image[y_min:y_max, x_min:x_max]


def split_dataset(images_dir: Path, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split a flat directory of class folders into train/val/test.

    Expects:
        images_dir/
        ├── hello/
        ├── goodbye/
        └── ...

    Creates:
        images_dir/train/hello/...
        images_dir/val/hello/...
        images_dir/test/hello/...
    """
    random.seed(seed)

    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name in ("train", "val", "test"):
            continue

        files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": files[:n_train],
            "val": files[n_train: n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            dest = images_dir / split_name / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dest / f.name)

        print(f"  {class_dir.name}: {n} images → train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")


def compute_dataset_stats(images_dir: Path):
    """Print dataset statistics: samples per class, total images, etc."""
    stats = {}
    total = 0

    for split in ("train", "val", "test"):
        split_dir = images_dir / split
        if not split_dir.exists():
            continue
        class_counts = {}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                class_counts[class_dir.name] = count
                total += count
        stats[split] = class_counts
        print(f"\n[{split}] {sum(class_counts.values())} images across {len(class_counts)} classes")

    print(f"\nTotal images: {total}")
    return stats


if __name__ == "__main__":
    images_dir = PROCESSED_DIR / "images"
    if images_dir.exists():
        print("Computing dataset statistics...")
        compute_dataset_stats(images_dir)
    else:
        print(f"No processed images found at {images_dir}")
        print("Run download_wlasl.py first.")
