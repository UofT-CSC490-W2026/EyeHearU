## Backend (`backend/`) – FastAPI ASL Inference API

This folder contains the **FastAPI backend** that powers Eye Hear U.  
It exposes HTTP endpoints for **health checks** and **ASL sign prediction** and will load the trained PyTorch model for inference.

---

### Key Files & Folders

- `app/main.py`  
  - FastAPI app entrypoint.  
  - Registers routers and CORS.  
  - **TODO (Backend + ML):** In the `startup` event, call `load_model` and attach it to `app.state.model`.

- `app/config.py`  
  - Central config using `pydantic_settings`.  
  - Reads `.env` for:
    - `MODEL_PATH` – path to `best_model.pt`.  
    - `MODEL_DEVICE` – `cpu`, `cuda`, or `mps`.  
    - Firebase credentials and project ID.

- `app/routers/health.py`  
  - `GET /health` – liveness probe.  
  - `GET /ready` – readiness probe, returns `model_loaded: true/false` based on `app.state.model`.

- `app/routers/predict.py`  
  - `POST /api/v1/predict` – accepts an image upload (`file`) and returns:
    - `sign` (top-1 label),
    - `confidence`,
    - `top_k` predictions.
  - **TODO:** Replace placeholder response with real inference via `model_service` + `preprocess_image`.

- `app/services/preprocessing.py`  
  - Image preprocessing (PIL → resized → normalized NumPy array).

- `app/services/model_service.py`  
  - **Skeleton for model loading & prediction.**  
  - **TODO (ML):** Implement `load_model` using `ASLClassifier` from `ml/models` and integrate `label_map.json` to map indices → labels.

- `app/services/firebase_service.py`  
  - Firestore integration for translation history (optional but recommended).

- `tests/test_health.py`  
  - Basic tests for `/health` and `/ready`.

---

### How to Run Locally

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # Fill in MODEL_PATH, MODEL_DEVICE, Firebase settings

uvicorn app.main:app --reload --port 8000
```

Then:
- Visit `http://localhost:8000/health` → should return `{"status": "ok"}`.  
- Visit `http://localhost:8000/docs` → interactive Swagger UI.

---

### Integration Points with Other Folders

- **ML / Model:**  
  - Expects a trained checkpoint at the path in `MODEL_PATH` (`ml/checkpoints/best_model.pt` by default).  
  - Expects a `label_map.json` (class index → sign label).

- **Mobile App:**  
  - `mobile` calls `POST /api/v1/predict` with a JPEG image.  
  - `mobile` may also call `/health` for a quick backend reachability check.

