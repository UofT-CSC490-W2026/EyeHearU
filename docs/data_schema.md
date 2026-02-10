# Data Schema — Eye Hear U

This document describes the data models used across the system.
Start with the MVP schema and evolve as features are added.

---

## 1. Firestore Collections (Database)

### `translations` collection
Stores each prediction made by the app (for analytics and history).

| Field            | Type      | Description                                    |
|------------------|-----------|------------------------------------------------|
| `session_id`     | string    | Anonymous session identifier                   |
| `predicted_sign` | string    | The top-1 predicted ASL sign label             |
| `confidence`     | number    | Model confidence score (0.0 - 1.0)             |
| `top_k`          | array     | Top-k predictions [{sign, confidence}]         |
| `timestamp`      | timestamp | When the prediction was made                   |
| `image_hash`     | string    | Hash of the input image (for dedup, no raw img)|
| `feedback`       | string    | Optional user feedback ("correct" / "wrong")   |

### `sessions` collection (future)
If user accounts are added later.

| Field          | Type      | Description                          |
|----------------|-----------|--------------------------------------|
| `session_id`   | string    | Unique session identifier            |
| `device_info`  | map       | {model, os_version, screen_size}     |
| `created_at`   | timestamp | Session start time                   |
| `last_active`  | timestamp | Last activity time                   |

### `vocabulary` collection (future — learning mode)
| Field        | Type      | Description                               |
|--------------|-----------|-------------------------------------------|
| `gloss`      | string    | The sign label (e.g., "hello")            |
| `category`   | string    | Grouping (greeting, medical, restaurant)  |
| `difficulty` | number    | 1=easy, 2=medium, 3=hard                 |
| `video_url`  | string    | Reference video showing the sign          |

---

## 2. ML Data Schema

### Training Data Layout
```
data/processed/
├── images/
│   ├── train/
│   │   ├── hello/
│   │   │   ├── frame_00001.jpg
│   │   │   └── ...
│   │   ├── goodbye/
│   │   └── ...
│   ├── val/
│   │   └── (same structure)
│   └── test/
│       └── (same structure)
├── label_map.json          # {"hello": 0, "goodbye": 1, ...}
└── dataset_stats.json      # {class_counts, total, splits}
```

### label_map.json
Maps human-readable sign names to integer class indices used by the model.

```json
{
  "A": 0,
  "B": 1,
  "hello": 26,
  "thank you": 27,
  ...
}
```

### Model Checkpoint
```
ml/checkpoints/
├── best_model.pt           # Best validation accuracy weights
├── label_map.json          # Copy of label map (for inference)
└── config.json             # Training config snapshot
```

---

## 3. API Request/Response Schema

### POST /api/v1/predict

**Request:** `multipart/form-data`
- `file`: JPEG/PNG image file

**Response:**
```json
{
  "sign": "hello",
  "confidence": 0.92,
  "top_k": [
    {"sign": "hello", "confidence": 0.92},
    {"sign": "help",  "confidence": 0.05},
    {"sign": "hi",    "confidence": 0.02}
  ],
  "message": null
}
```

---

## 4. Aspirational Datasets

### Ideal dataset characteristics (what we wish existed)
| Property               | Ideal Value                              |
|------------------------|------------------------------------------|
| Number of glosses      | 100+ common signs                        |
| Videos per gloss       | 50+ from diverse signers                 |
| Signer diversity       | 20+ signers, varied skin tones, ages     |
| Backgrounds            | Real-world (offices, homes, restaurants) |
| Lighting               | Varied (natural, fluorescent, dim)       |
| Camera angles          | Front-facing, slightly angled            |
| Resolution             | 720p+                                    |
| Frame rate             | 30fps                                    |
| Annotations            | Gloss label, hand bounding box, timestamps|

### Actual datasets available
| Dataset      | Glosses | Videos | Signers | Quality Notes                           |
|--------------|---------|--------|---------|----------------------------------------|
| WLASL        | 2,000   | ~21K   | 100+    | YouTube sourced, many links broken     |
| ASL Citizen  | 2,731   | ~84K   | 52      | Crowdsourced, varied quality           |
| MS-ASL       | 1,000   | ~25K   | 222     | Cleaned YouTube data, good diversity   |
| ASL-LEX      | 2,723   | ~2.7K  | 1       | Single signer, very clean              |

### Gap analysis
- **Broken links**: WLASL videos are YouTube-sourced; many are now unavailable
- **Background diversity**: Most datasets have clean backgrounds, unlike real-world use
- **Mobile camera quality**: No dataset specifically captures from phone cameras
- **Our target vocab**: Not all our target signs may be in available datasets
- **Mitigation**: Supplement with custom-recorded data from team members
