# Eye Hear U

**Real-time ASL-to-English translation on iOS** — one sign at a time.

Eye Hear U is a mobile application that translates isolated American Sign Language (ASL) signs into English text and speech. Built for everyday communication scenarios like restaurant ordering and medical emergencies.

---

## Architecture Overview

```
┌──────────────────────┐        ┌──────────────────────┐
│   Mobile App         │  HTTP  │   Backend API        │
│   (React Native /    │───────▶│   (FastAPI / Python)  │
│    Expo)             │        │                      │
│                      │        │  POST /api/v1/predict│
│  - Camera capture    │◀───────│  → sign label +      │
│  - Display results   │  JSON  │    confidence score   │
│  - Text-to-speech    │        │                      │
│  - Translation       │        │  /health, /ready     │
│    history           │        └──────────┬───────────┘
└──────────────────────┘                   │
                                           │ loads
                                ┌──────────▼───────────┐
                                │   ML Model           │
                                │   (PyTorch)          │
                                │                      │
                                │  CNN backbone         │
                                │  (ResNet18)          │
                                │       ↓              │
                                │  Patch projection    │
                                │       ↓              │
                                │  Transformer encoder │
                                │  (2 layers, 4 heads) │
                                │       ↓              │
                                │  Classification head │
                                │  → 62 classes        │
                                └──────────────────────┘

                                ┌──────────────────────┐
                                │   Firebase           │
                                │   (Firestore)        │
                                │                      │
                                │  - Translation       │
                                │    history           │
                                │  - Session tracking  │
                                │  - Usage analytics   │
                                └──────────────────────┘
```

## Project Structure

```
.
├── backend/                  # FastAPI backend server
│   ├── app/
│   │   ├── main.py           # App entrypoint
│   │   ├── config.py         # Settings from env vars
│   │   ├── routers/
│   │   │   ├── health.py     # /health, /ready endpoints
│   │   │   └── predict.py    # POST /api/v1/predict
│   │   ├── schemas/
│   │   │   └── prediction.py # Pydantic request/response models
│   │   └── services/
│   │       ├── model_service.py    # Model loading & inference
│   │       ├── preprocessing.py    # Image preprocessing pipeline
│   │       └── firebase_service.py # Firestore integration
│   ├── tests/
│   └── requirements.txt
│
├── mobile/                   # React Native (Expo) mobile app
│   ├── app/
│   │   ├── _layout.tsx       # Root navigation layout
│   │   ├── index.tsx         # Home screen
│   │   ├── camera.tsx        # Camera + prediction screen
│   │   └── history.tsx       # Translation history
│   ├── services/
│   │   └── api.ts            # Backend API client
│   ├── package.json
│   └── app.json
│
├── ml/                       # Machine learning pipeline
│   ├── config.py             # Training/model configuration
│   ├── models/
│   │   └── classifier.py     # CNN-Transformer ASL classifier
│   ├── training/
│   │   ├── train.py          # Training loop
│   │   └── dataset.py        # PyTorch dataset
│   ├── evaluation/
│   │   └── evaluate.py       # Evaluation & error analysis
│   └── requirements.txt
│
├── data/                     # Data pipeline
│   ├── scripts/
│   │   ├── download_wlasl.py # WLASL dataset downloader
│   │   └── preprocess.py     # Preprocessing & splitting
│   ├── raw/                  # Raw downloaded data (gitignored)
│   └── processed/            # Processed images + label map (gitignored)
│
├── docs/
│   ├── architecture.md       # Detailed system architecture & roles
│   └── data_schema.md        # Database & data schemas
│
├── Dockerfile                # Backend container
├── docker-compose.yml        # Local dev orchestration
└── .gitignore
```

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- iOS device or simulator (for Expo)
- Docker (optional, for containerized backend)

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env

# Run the server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Mobile App

```bash
cd mobile
npm install
npx expo start
```

Scan the QR code with Expo Go on your phone, or press `i` to open in the iOS simulator.

### ML Pipeline

```bash
cd ml
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download and preprocess data
python -m data.scripts.download_wlasl

# Train the model
python -m training.train

# Evaluate
python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt
```

### Docker (Backend only)

```bash
docker-compose up --build
```

---

## Target Vocabulary

We focus on **specific real-world scenarios** rather than trying to cover all of ASL:

| Scenario           | Signs                                                    |
|--------------------|----------------------------------------------------------|
| **Greetings**      | hello, goodbye, please, thank you, sorry, yes, no       |
| **Basic needs**    | help, stop, wait, water, food, bathroom                  |
| **Restaurant**     | eat, drink, hot, cold, more, enough, check               |
| **Medical**        | hurt, emergency, doctor, medicine, pain, sick, allergic   |
| **Fingerspelling** | A–Z (fallback for any word not in vocabulary)            |
| **Numbers**        | 1–10                                                     |

**Total: ~62 classes** (26 letters + ~36 words)

---

## Datasets

| Dataset      | Size    | Use Case                                    |
|--------------|---------|---------------------------------------------|
| **WLASL**    | 2K glosses, 21K videos | Primary training data            |
| **ASL Citizen** | 2.7K glosses, 84K videos | Robustness testing (varied backgrounds) |
| **MS-ASL**   | 1K glosses, 25K videos | Supplementary training data       |
| **Custom**   | TBD     | Fill gaps in target vocabulary               |

See `docs/data_schema.md` for the full aspirational vs. actual dataset comparison.

---

## Team

| Name         | Role                              |
|--------------|-----------------------------------|
| Maria Ma     | Project Manager & iOS Frontend    |
| Zhixiao Fu   | Backend & Database Engineer       |
| Siyi Zhu     | Modeling Engineer (CV/ML)         |
| Chloe Yang   | Modeling Engineer (CV/ML)         |

---

## Revised Milestones (addressing professor feedback)

| Week  | Focus                          | Key Deliverables                                           |
|-------|--------------------------------|------------------------------------------------------------|
| 1–2   | **User research + data setup** | Interview 3-5 ASL users, download WLASL, set up infra     |
| 3–4   | **Baseline model + backend**   | Train ResNet18 baseline, deploy FastAPI, Firebase setup    |
| 5–6   | **Integration + UX testing**   | Connect mobile → backend → model, test with 3+ users      |
| 7     | **Audio + polish**             | Add TTS output, improve latency, handle edge cases        |
| 8     | **Evaluation in scenarios**    | Test in restaurant/medical scenarios, final report + demo  |

### Key changes from A1 (per professor feedback)
- **User testing moved to Week 1** — talk to ASL users early, not just at the end
- **Scenario-specific evaluation** — test in restaurant and medical contexts (aligns with press release)
- **Image vs video**: Starting with image-based for simplicity; will prototype video if time allows
- **Inference location**: Model runs on the backend server (not on-device) for the MVP; explore on-device (Core ML) as a stretch goal
