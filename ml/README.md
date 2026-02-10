## ML Pipeline (`ml/`) – ASL Classifier Training & Evaluation

This folder contains the **machine learning code** for Eye Hear U.  
It defines the ASL classifier model architecture, training loop, and evaluation scripts.

---

### High-Level Responsibilities

- Define the **CNN + Transformer** ASL sign classifier (`ASLClassifier`).
- Load preprocessed ASL sign images and labels.
- Train the model to recognize ~62 classes (scenario vocabulary + A–Z + 1–10).
- Evaluate accuracy and common confusions.
- Export the best checkpoint to be loaded by the backend.

---

### Key Files & Folders

- `config.py`  
  - Central configuration for:
    - Data paths & target vocabulary (greetings, restaurant, medical, etc.).
    - Model settings (backbone, Transformer depth, number of classes).
    - Training hyperparameters (LR, batch size, epochs, device, etc.).

- `models/classifier.py`  
  - `ASLClassifier`:
    - CNN backbone (e.g., ResNet18) → 7×7 feature map.
    - Patch projection → sequence of patch embeddings.
    - Positional encoding + 2-layer Transformer encoder.
    - CLS token + linear classification head → logits over classes.

- `training/dataset.py`  
  - `ASLImageDataset`:
    - Loads images from `data/processed/images/{train,val,test}/<class>/*.jpg`.
    - Applies augmentations for training (color jitter, random crop, small rotations).
    - Uses `label_map.json` to map sign names → indices.

- `training/train.py`  
  - End-to-end training script:
    - Builds train/val DataLoaders.
    - Instantiates `ASLClassifier` using `Config`.
    - Trains with AdamW + cosine LR schedule.
    - Freezes backbone for first N epochs, then fine-tunes.
    - Early stopping on validation accuracy.
    - Saves `checkpoints/best_model.pt`.

- `evaluation/evaluate.py`  
  - Evaluation script:
    - Loads a checkpoint & test set.
    - Computes overall accuracy, top-5 accuracy.
    - Computes per-class accuracy.
    - Lists most confused sign pairs.
    - Saves `evaluation_results.json`.

- `checkpoints/`  
  - Contains model weights and metadata files:
    - `best_model.pt` (to be loaded by the backend).
    - `label_map.json` (copy of the label map used during training).
    - Optional `config.json` snapshot.

> Note: `ml/inference/` is currently a **placeholder** for future deployment helpers  
> (e.g., ONNX Runtime, Core ML adapters). The FastAPI backend directly loads the PyTorch model for now.

---

### How to Train the Model

1. **Prepare data via the `data/` pipeline** (see `docs/data_schema.md`):  
   - Run `data/scripts/download_wlasl.py` to get WLASL metadata + `label_map.json`.  
   - Download WLASL videos and place them under `data/raw/wlasl/videos/`.  
   - Run the script again to extract frames into `data/processed/images/`.  
   - Optionally run `data/scripts/preprocess.py` for hand cropping and splitting.

2. **Set up a Python environment and install ML deps:**

   ```bash
   cd ml
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run training:**

   ```bash
   python -m training.train
   ```

   - Monitors epoch-by-epoch train/val loss and accuracy.
   - Writes checkpoints to `ml/checkpoints/`.

4. **Evaluate the model:**

   ```bash
   python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt
   ```

   - Prints overall + per-class accuracy and top confusions.
   - Saves results to `evaluation_results.json`.

5. **Hand model off to backend:**
   - Ensure `best_model.pt` and `label_map.json` are accessible to `backend` (via `MODEL_PATH` in `.env` and shared path).
   - Coordinate with backend to implement `load_model` and wire `app.state.model`.

---

### Who Should Work Here & Typical Tasks

- **Siyi & Chloe (ML):**
  - Define and iterate on the model architecture.
  - Tune hyperparameters to hit usable accuracy.
  - Diagnose failure modes via confusion matrices.
  - Maintain training configs in `config.py`.

- **Zhixiao (Backend):**
  - Uses this folder mainly to:
    - Know where `best_model.pt` comes from.
    - Understand the `label_map.json` structure for inference.

