## Eye Hear U

[![CI](https://github.com/MariaMa-GitHub/EyeHearU/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MariaMa-GitHub/EyeHearU/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MariaMa-GitHub/EyeHearU/branch/main/graph/badge.svg)](https://codecov.io/gh/MariaMa-GitHub/EyeHearU)

**Real-time ASL-to-English translation on iOS** — one sign at a time.

Eye Hear U translates isolated American Sign Language (ASL) signs into English text and speech using a mobile app, a backend inference API, and a video classifier trained on public ASL datasets. Infrastructure is provisioned on AWS via Terraform.

### Documentation

| Audience | Document |
|----------|----------|
| End users | [User guide](docs/USER_GUIDE.md) |
| Developers | [Developer guide](docs/DEVELOPER_GUIDE.md) |
| Testing & coverage | [Testing](docs/TESTING.md) |
| Production deployment | [Production](docs/PRODUCTION.md) |
| Inference preprocessing | [Preprocessing (I3D)](docs/PREPROCESSING.md) |
| Benchmarking & evaluation | [Benchmarking](docs/BENCHMARKING.md) |

**Codecov:** Register the repository at [codecov.io](https://about.codecov.io/) and add the `CODECOV_TOKEN` secret under GitHub → Settings → Secrets → Actions so PR comments and the badge update automatically. CI still enforces **100%** backend line/branch coverage without Codecov.

---

## Architecture Overview

```
┌──────────────────────┐        ┌──────────────────────────────────────┐
│   Mobile App         │  HTTP  │   Inference API                      │
│   (React Native /    │───────▶│   (FastAPI on ECS Fargate behind ALB)│
│    Expo)             │        │                                      │
│                      │        │   POST /api/v1/predict               │
│  - Camera capture    │◀───────│   → sign label + confidence          │
│  - Display results   │  JSON  │                                      │
│  - Text-to-speech    │        │   Loads model from S3 at startup     │
│  - Translation       │        └─────────────────┬────────────────────┘
│    history           │                          │
└──────────────────────┘                          │
                                       ┌──────────▼──────────┐
                                       │   ML Model          │
                                       │   (PyTorch)         │
                                       │                     │
                                       │   Inception I3D     │
                                       │   (spatiotemporal   │
                                       │    3D CNN)          │
                                       │        ↓            │
                                       │   Classification    │
                                       │   (856 glosses)     │
                                       └─────────────────────┘

  ┌─────────────────────┐   ┌────────────────────┐   ┌──────────────────┐
  │   Amazon S3         │   │   Firebase         │   │   CloudWatch     │
  │   (Data Lake)       │   │   (Firestore)      │   │   (Logs/Alerts)  │
  │                     │   │                    │   │                  │
  │   raw/ → processed/ │   │   - Translation    │   │   - Pipeline     │
  │   → models/         │   │     history        │   │     metrics      │
  │                     │   │   - User feedback  │   │   - API latency  │
  └─────────────────────┘   └────────────────────┘   └──────────────────┘
```

## Datasets

| Dataset | Role | Size |
|---|---|---|
| **ASL Citizen** | Primary (train / val / test) | 2,731 glosses, ~83K videos, 52 signers |
| **WLASL** | Supplementary training | 2,000 glosses, ~21K videos |
| **MS-ASL** | Supplementary training | 1,000 glosses, ~25K videos |

## Project Structure

```
.
├── backend/                  # FastAPI backend server
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routers/          # health.py, predict.py
│   │   ├── schemas/          # Pydantic models
│   │   └── services/         # model_service, preprocessing, firebase
│   ├── tests/                # 66 tests, 100% coverage
│   └── requirements.txt
│
├── mobile/                   # React Native (Expo) mobile app
│   ├── app/                  # _layout, index, camera, history
│   ├── __tests__/            # 59 tests, 100% line coverage
│   ├── services/api.ts
│   └── package.json
│
├── ml/                       # Machine learning code
│   ├── i3d_msft/             # Inception I3D (deployed model)
│   │   ├── pytorch_i3d.py
│   │   └── videotransforms.py
│   ├── i3d_label_map_mvp-sft-full-v1.json  # 856-class label map (v4)
│   ├── models/classifier.py  # ASLVideoClassifier (in-repo baseline)
│   ├── config.py             # Video classifier config
│   ├── training/
│   │   ├── train.py
│   │   └── dataset.py        # ASLVideoDataset (PyTorch)
│   ├── evaluation/
│   │   └── evaluate.py       # Accuracy, F1, confusion matrix, latency
│   ├── tests/                # 73 ML unit tests
│   └── requirements.txt
│
├── data/                     # Data pipeline
│   ├── Dockerfile            # Pipeline container image
│   ├── scripts/
│   │   ├── pipeline_config.py          # Shared config (local + S3)
│   │   ├── ingest_asl_citizen.py
│   │   ├── ingest_wlasl.py
│   │   ├── ingest_msasl.py
│   │   ├── preprocess_clips.py
│   │   ├── build_unified_dataset.py
│   │   ├── validate.py
│   │   └── requirements.txt
│   ├── raw/                  # Raw video files (gitignored)
│   └── processed/            # Processed clips + metadata (gitignored)
│
├── infrastructure/           # Infrastructure as Code
│   ├── main.tf               # Root module, provider config
│   ├── variables.tf
│   ├── outputs.tf
│   ├── environments/
│   │   ├── dev.tfvars
│   │   ├── staging.tfvars
│   │   └── prod.tfvars
│   ├── modules/              # Terraform modules
│   │   ├── s3/               # Data lake bucket
│   │   ├── ecr/              # Container registries
│   │   ├── batch/            # Pipeline job compute
│   │   ├── ecs/              # API cluster + ALB
│   │   ├── iam/              # Roles and policies
│   │   ├── networking/       # VPC, subnets, NAT
│   │   └── monitoring/       # CloudWatch, SNS alerts
│   └── k8s/                  # Kubernetes manifests
│       ├── namespace.yaml
│       ├── deployment.yaml   # API Deployment (2 replicas)
│       ├── service.yaml      # ClusterIP Service
│       ├── ingress.yaml      # ALB Ingress
│       ├── configmap.yaml    # Environment configuration
│       └── hpa.yaml          # Horizontal Pod Autoscaler
│
├── docs/
│   ├── USER_GUIDE.md         # End-user guide
│   ├── DEVELOPER_GUIDE.md    # Setup and day-to-day development
│   ├── TESTING.md            # Tests, coverage, CI
│   ├── PRODUCTION.md         # Production deployment
│   ├── PREPROCESSING.md      # I3D inference preprocessing
│   ├── BENCHMARKING.md       # Evaluation metrics and reproduction
│   ├── architecture.md       # System design and use cases
│   ├── data_pipeline.md      # Data processing pipeline
│   ├── data_schema.md        # Data schemas
│   ├── terraform_guide.md    # Terraform IaC guide
│   └── a2_writeup.md         # Datasets writeup
│
├── .github/workflows/ci.yml  # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

## Quick Start

### Infrastructure (Terraform)

```bash
cd infrastructure
terraform init
terraform apply -var-file=environments/dev.tfvars
```

### Kubernetes Deployment (alternative to ECS)

```bash
kubectl apply -k infrastructure/k8s/
```

### Data Pipeline

```bash
cd data/scripts
pip install -r requirements.txt

# Local run
export PIPELINE_ENV=local
python ingest_asl_citizen.py
python ingest_wlasl.py
python ingest_msasl.py
python preprocess_clips.py
python build_unified_dataset.py
python validate.py
```

### ML Training

```bash
cd ml
pip install -r requirements.txt
python -m training.train
python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt
```

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Mobile App

```bash
cd mobile
npm install
npx expo start
```

---

For additional design notes see `docs/architecture.md`, `docs/data_schema.md`, `docs/data_pipeline.md`, and `docs/terraform_guide.md`.
