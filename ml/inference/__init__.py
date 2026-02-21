"""
Inference utilities placeholder.

This module is reserved for any future *deployment-specific* inference helpers,
such as:
- ONNX Runtime wrappers
- batched inference servers
- on-device (Core ML / TensorFlow Lite) adapters

For the current CSC490 MVP:
  - All training code lives in `ml/training/`.
  - Evaluation and error analysis live in `ml/evaluation/evaluate.py`.
  - The FastAPI backend loads the trained PyTorch model directly
    from `ml/checkpoints/best_model.pt`.
"""