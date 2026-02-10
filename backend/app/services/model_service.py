"""
Model loading and inference service.

This module is responsible for:
  - Loading the trained PyTorch model from disk
  - Running inference on preprocessed images
  - Returning top-k predictions with confidence scores
"""

import torch
import numpy as np
from pathlib import Path


# The label map will be generated during training and saved alongside the model
LABEL_MAP_PATH = Path(__file__).parent.parent.parent.parent / "ml" / "checkpoints" / "label_map.json"


def load_model(model_path: str, device: str = "cpu"):
    """
    Load a trained PyTorch model from disk.

    Args:
        model_path: Path to the saved .pt checkpoint
        device: "cpu", "cuda", or "mps"

    Returns:
        The loaded model in eval mode.
    """
    # TODO: Implement once model architecture is finalized
    # model = ASLClassifier(...)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.to(device)
    # model.eval()
    # return model
    raise NotImplementedError("Model loading not yet implemented — train a model first.")


def predict(model, image_tensor: torch.Tensor, top_k: int = 5):
    """
    Run inference on a single preprocessed image tensor.

    Args:
        model: The loaded PyTorch model
        image_tensor: Preprocessed image tensor of shape (1, C, H, W)
        top_k: Number of top predictions to return

    Returns:
        List of (label, confidence) tuples.
    """
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

    # TODO: Map indices back to sign labels using label_map.json
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append({
            "sign": f"sign_{idx.item()}",  # placeholder
            "confidence": round(prob.item(), 4),
        })

    return results
