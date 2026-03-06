# Eye Hear U

**Real-time ASL-to-English translation on iOS** 

Eye Hear U translates isolated American Sign Language (ASL) signs into English text and speech using a mobile app, a backend inference API, and a video classifier trained on public ASL datasets. This repository contains deliverables for CSC490 assignments A1, A2, and A3.

---

## Assignments Overview

| Assignment | Focus | Contents |
|------------|-------|----------|
| **A1** | Project proposal and planning | Initial project idea, problem statement, and team planning |
| **A2** | Full-stack ASL app + infrastructure | Datasets, architecture, data pipeline, Terraform IaC, backend, mobile, ML |
| **A3** | LLM ablations on Modal | Nanochat baseline + SwiGLU / RMSNorm ablations, cloud training |

---

### A1 — Project Proposal

- **Location:** `a1/`
- **Deliverables:** Project proposal document (`a1.pdf`)
- Defines the problem (no simple real-time ASL-to-English tool), target users, and high-level solution approach.

---

### A2 — Eye Hear U: Datasets, Architecture & Data Pipeline

- **Location:** `a2/`
- **Parts:**
  - **Part 1:** Aspirational datasets & schemas — ideal training/eval data for isolated signs and mobile-captured scenarios
  - **Part 2:** System architecture — mobile app, FastAPI backend, 3D CNN classifier, Firebase, AWS
  - **Part 3:** Data pipeline — ingestion (ASL Citizen, WLASL, MS-ASL), preprocessing, unified dataset, validation
  - **Part 5:** Disaster recovery — Terraform state, S3, DynamoDB locks
- **Components:**
  - `backend/` — FastAPI inference API
  - `mobile/` — React Native (Expo) iOS app
  - `ml/` — PyTorch video classifier (3D CNN / R3D-18)
  - `data/` — Data pipeline scripts (ingest, preprocess, build dataset)
  - `infrastructure/` — Terraform modules (S3, ECR, Batch, ECS, IAM, networking, monitoring)
  - `docs/` — `architecture.md`, `data_schema.md`, `data_pipeline.md`, `a2_writeup.md`, `terraform_guide.md`

---

### A3 — Nanochat Ablations on Modal

- **Location:** `a3/nanochat-modal/`
- **Focus:** Ablation studies on Karpathy’s [nanochat](https://github.com/karpathy/nanochat) — baseline (ReLU²) vs SwiGLU and Learnable RMSNorm.
- **Deliverables:**
  - `nanochat_modal.py` — Modal app for data staging, tokenizer, pretrain, and eval
  - `ablation_swiglu/` — SwiGLU model and training entry
  - `ablation_rmsnorm/` — Learnable RMSNorm model and training entry
- **Setup:** Clone nanochat into the repo, install `uv` + Modal, configure secrets. See `a3/nanochat-modal/README.md`.

---

## Project Structure

```
.
├── a1/                     # A1: Project proposal
│   └── a1.pdf
│
├── a2/                     # A2: Eye Hear U full project
│   ├── backend/            # FastAPI API
│   ├── mobile/             # React Native app
│   ├── ml/                 # Video classifier
│   ├── data/               # Data pipeline scripts
│   ├── infrastructure/     # Terraform IaC
│   ├── docs/               # A2 documentation
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── Project Doc.pdf
│
├── a3/                     # A3: LLM ablations
│   └── nanochat-modal/     # Modal + SwiGLU + RMSNorm
│       ├── nanochat_modal.py
│       ├── ablation_swiglu/
│       └── ablation_rmsnorm/
│
└── .gitignore
```

---

## Quick Start

### A2 — Eye Hear U (Infrastructure)

```bash
cd a2/infrastructure
terraform init
terraform apply -var-file=environments/dev.tfvars
```

### A2 — Data Pipeline

```bash
cd a2/data/scripts
pip install -r requirements.txt
export PIPELINE_ENV=local
python ingest_asl_citizen.py
python preprocess_clips.py
python build_unified_dataset.py
```

### A2 — Backend & Mobile

```bash
cd a2/backend && pip install -r requirements.txt && cp .env.example .env && uvicorn app.main:app --reload
cd a2/mobile && npm install && npx expo start
```

### A3 — Nanochat on Modal

```bash
cd a3/nanochat-modal
git clone https://github.com/karpathy/nanochat.git
uv sync && modal setup
uv run modal run nanochat_modal.py::stage_data
uv run modal run nanochat_modal.py::stage_pretrain
```

---

## Datasets (A2)

| Dataset     | Role                    | Size                          |
|-------------|-------------------------|-------------------------------|
| **ASL Citizen** | Primary (train/val/test) | 2,731 glosses, ~83K videos    |
| **WLASL**   | Supplementary            | 2,000 glosses, ~21K videos    |
| **MS-ASL**  | Supplementary            | 1,000 glosses, ~25K videos    |

---

For detailed documentation, see `a2/docs/architecture.md`, `a2/docs/data_schema.md`, `a2/docs/data_pipeline.md`, and `a2/docs/terraform_guide.md`.
