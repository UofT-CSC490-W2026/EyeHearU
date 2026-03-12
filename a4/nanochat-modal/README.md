# Ablation SwiGLU + RMSNorm — Modal runs

Code to run nanochat picochat (d12) baseline and ablations (SwiGLU, Learnable RMSNorm) on Modal.  
Uses the same `nanochat_modal.py` and ablation code as the tutorial repo.

## What’s in this repo

- **`nanochat_modal.py`** — Modal app (stage_data, stage_tokenizer, stage_pretrain, stage_pretrain_swiglu, stage_pretrain_rmsnorm, stage_post_pretrain_eval, etc.)
- **`ablation_swiglu/gpt_swiglu.py`** — SwiGLU model (copy into nanochat before Run 2)
- **`ablation_swiglu/scripts/base_train_swiglu.py`** — SwiGLU training entry (copy into nanochat before Run 2)
- **`ablation_rmsnorm/gpt_rmsnorm.py`** — Learnable RMSNorm model (copy into nanochat before Run 3)
- **`ablation_rmsnorm/scripts/base_train_rmsnorm.py`** — RMSNorm training entry (copy into nanochat before Run 3)

## Setup (one-time)

### 1. Clone nanochat into this repo

From this repo root:

```bash
git clone https://github.com/karpathy/nanochat.git
```

You must have a `./nanochat` directory so Modal can build the image with `.add_local_dir("./nanochat", ...)`.

### 2. Remove nanochat’s local venv (if present)

Modal builds a Linux image; a Mac `.venv` inside nanochat can cause “Exec format error”:

```bash
rm -rf nanochat/.venv
```

### 3. Install uv and Modal

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install deps (includes modal)
uv sync
```

### 4. Modal auth and secret

```bash
modal setup
```

Create the secret (W&B + HuggingFace token for data download):

```bash
modal secret create nanochat-secrets \
  WANDB_API_KEY=your_wandb_key \
  HF_TOKEN=hf_your_huggingface_token
```

- W&B key: https://wandb.ai/authorize  
- HF token: https://huggingface.co/settings/tokens (needed for FineWeb-EDU)

## Run 1: Baseline (ReLU²)

1. In **`nanochat_modal.py`** set **`WANDB_RUN = "picochat_baseline"`** (or leave as you prefer).
2. From this repo root:

```bash
uv run modal run nanochat_modal.py::stage_data
uv run modal run nanochat_modal.py::stage_tokenizer
uv run modal run nanochat_modal.py::stage_pretrain
```

## Run 2: SwiGLU

1. Copy SwiGLU code into nanochat (so the next image build includes it):

```bash
cp ablation_swiglu/gpt_swiglu.py nanochat/nanochat/
mkdir -p nanochat/scripts
cp ablation_swiglu/scripts/base_train_swiglu.py nanochat/scripts/
```

2. In **`nanochat_modal.py`** set **`WANDB_RUN = "picochat_swiglu"`**.
3. Run SwiGLU pretrain (data and tokenizer from Run 1 are reused):

```bash
uv run modal run nanochat_modal.py::stage_pretrain_swiglu
```

## Run 3: Learnable RMSNorm

1. Copy RMSNorm code into nanochat (so the next image build includes it):

```bash
cp ablation_rmsnorm/gpt_rmsnorm.py nanochat/nanochat/
cp ablation_rmsnorm/scripts/base_train_rmsnorm.py nanochat/scripts/
cp ablation_rmsnorm/scripts/base_eval_rmsnorm.py nanochat/scripts/
```

2. In **`nanochat_modal.py`** set **`WANDB_RUN = "picochat_rmsnorm"`**.
3. Run RMSNorm pretrain (data and tokenizer from Run 1 are reused):

```bash
uv run modal run nanochat_modal.py::stage_pretrain_rmsnorm
```
## Config in `nanochat_modal.py`

- **`DEPTH = 12`**, **`NUM_SHARDS = 8`** — picochat, 8 shards (as in ABLATION_STEPS).
- **`WANDB_RUN`** — set to `"picochat_baseline"` for Run 1, `"picochat_swiglu"` for Run 2, `"picochat_rmsnorm"` for Run 3.

Checkpoints and data live on the Modal Volume **nanochat-vol**; no need to re-download for Run 2 or Run 3.
