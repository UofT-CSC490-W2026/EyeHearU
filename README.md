# Eye Hear U

Eye Hear U is a real-time ASL-to-English translation project for mobile, backed by a FastAPI inference API and an I3D video classifier.  
This repository is organized by assignment milestones (`a1` through `a5`), with **`a5/` as the latest integrated system** and primary implementation.

---

## Repository Map

| Folder | What it contains | Current relevance |
|---|---|---|
| `a1/` | Initial proposal and project framing | Historical |
| `a2/` | First full-stack Eye Hear U implementation (backend/mobile/ml/data/infra) | Historical baseline |
| `a3/` | Nanochat ablations on Modal (SwiGLU, RMSNorm) | Separate experiment track |
| `a4/` | Additional nanochat-related deliverables | Separate experiment track |
| `a5/` | Latest end-to-end Eye Hear U stack (app, API, ML, data, infra, docs) | **Primary** |

---

## `a5` at a Glance (Primary System)

`a5/` is the most complete and current version of Eye Hear U. It includes:

- `a5/backend/`: FastAPI inference service with `/health`, `/ready`, and `/api/v1/predict`.
- `a5/mobile/`: Expo React Native app (`index`, `camera`, `history`) with local history and TTS.
- `a5/ml/`: I3D model code (`i3d_msft`), training/evaluation scripts, Modal training entrypoint.
- `a5/data/`: ingestion, preprocessing, split-planning, and validation scripts.
- `a5/infrastructure/`: Terraform modules plus Kubernetes manifests.
- `a5/docs/`: user/dev/testing/production/eval/profiling/benchmarking docs.

Primary reference docs:

- `a5/README.md`
- `a5/docs/DEVELOPER_GUIDE.md`
- `a5/docs/USER_GUIDE.md`
- `a5/docs/TESTING.md`
- `a5/docs/PRODUCTION.md`

---

## `a5` Architecture Summary

```text
Mobile (Expo) -> FastAPI Backend -> I3D Model Inference
      |                |                 |
      |                |                 -> PyTorch I3D checkpoint + label map
      |                -> preprocessing + top-k prediction response
      -> camera capture, upload, result display, history, TTS

Data + Training paths:
- data/scripts: ingest/preprocess/unify/validate + split planning
- ml: training/eval (local + S3 workflows, Modal execution)
- infrastructure: Terraform (AWS) and optional Kubernetes manifests
```

---

## Quick Start (from `a5`)

### Backend

```bash
cd a5/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export PYTHONPATH=..
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Mobile

```bash
cd a5/mobile
npm install --legacy-peer-deps
npm run start:lan
```

Set `EXPO_PUBLIC_API_URL` (or `app.json` extra config) to your backend URL.

### ML Training (Modal)

```bash
pip install modal
modal setup
modal run a5/ml/modal_train_i3d.py --bucket eye-hear-u-public-data-ca1 --epochs 1 --clip-limit 200
```

### Data Pipeline

```bash
cd a5/data/scripts
pip install -r requirements.txt
export PIPELINE_ENV=local
python ingest_asl_citizen.py
python ingest_wlasl.py
python ingest_msasl.py
python preprocess_clips.py
python build_unified_dataset.py
python validate.py
```

### Infrastructure

```bash
cd a5/infrastructure
terraform init
terraform apply -var-file=environments/dev.tfvars
```

---

## Testing (Primary Targets in `a5`)

```bash
# Backend
cd a5/backend && pytest tests/ -v --cov=app --cov-fail-under=100

# ML
cd a5/ml && python -m pytest tests/ -v --cov --cov-fail-under=100

# Mobile
cd a5/mobile && npx jest --coverage
```

---

## Notes on Historical Folders

- `a1/` through `a4/` are preserved assignment deliverables and experiment tracks.
- For active development of the ASL app/API/ML pipeline, prefer working in `a5/`.
- When in doubt, treat `a5/docs/` and `a5/README.md` as the canonical project references.
