"""
Image preprocessing pipeline for inference.

Handles:
  - Decoding uploaded image bytes → PIL Image
  - Resizing to model input dimensions
  - Normalizing pixel values
  - Converting to PyTorch tensor
"""

import io
import numpy as np
from PIL import Image

# Model input dimensions (will match training config)
INPUT_SIZE = (224, 224)

# ImageNet normalization (standard for pretrained CNN backbones)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes into a normalized numpy array
    ready for model inference.

    Args:
        image_bytes: Raw bytes from the uploaded image file.

    Returns:
        Numpy array of shape (1, 3, H, W), float32, normalized.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(INPUT_SIZE, Image.BILINEAR)

    # Convert to float32 array in [0, 1]
    arr = np.array(image, dtype=np.float32) / 255.0

    # Normalize with ImageNet stats
    arr = (arr - MEAN) / STD

    # HWC → CHW, add batch dimension
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr
