# nanochat-modal — A4 Assignment Infrastructure

Modal-based infrastructure for running nanochat SFT, RL, and evaluation stages on cloud GPUs.

## What's in this directory

- **`nanochat_modal.py`** — Modal app with all training and evaluation stages
- **`nanochat/`** — Clone of [karpathy/nanochat](https://github.com/karpathy/nanochat) (gitignored)
  - `scripts/chat_sft.py` — SFT training (includes consolidated midtraining data)
  - `scripts/chat_rl.py` — Baseline RL (simplified GRPO on GSM8K)
  - `scripts/chat_eval.py` — Benchmark evaluation (ARC, MMLU, GSM8K, HumanEval, SpellingBee)
  - `scripts/chat_rl_combined2rwd.py` — Custom RL with additional rewards (Part 4)
  - `scripts/gsm8k_detailed_eval.py` — Per-problem GSM8K evaluation (Part 3/4 analysis)
- **`ablation_swiglu/`** — SwiGLU ablation code from A3 (not used in A4)
- **`ablation_rmsnorm/`** — Learnable RMSNorm ablation code from A3 (not used in A4)

## Setup (one-time)

### 1. Clone nanochat into this directory

```bash
cd a4/nanochat-modal
git clone https://github.com/karpathy/nanochat.git
```

### 2. Remove nanochat's local venv (if present)

Modal builds a Linux image; a Mac `.venv` inside nanochat can cause "Exec format error":

```bash
rm -rf nanochat/.venv
```

### 3. Install uv and Modal

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 4. Modal auth and secret

```bash
modal setup
modal secret create nanochat-secrets \
  WANDB_API_KEY=your_wandb_key \
  HF_TOKEN=hf_your_huggingface_token
```

## A4 Stages

### Part 2: SFT with original configuration

```bash
# Runs SFT on d12 baseline pretrained checkpoint, then evaluates
uv run modal run --detach nanochat_modal.py::stage_sft
```

### Part 3: Baseline RL on GSM8K

```bash
# Runs RL (simplified GRPO) on GSM8K, starting from SFT checkpoint
uv run modal run --detach nanochat_modal.py::stage_rl

# Evaluate RL checkpoint on all benchmarks
uv run modal run --detach nanochat_modal.py::stage_eval_rl

# Detailed per-problem GSM8K eval (for Part 3 analysis)
uv run modal run --detach nanochat_modal.py::stage_gsm8k_detailed_eval
```

### Part 4: RL with custom rewards

```bash
# Combined rewards (correctness + format + steps + close)
uv run modal run --detach nanochat_modal.py::stage_rl_combined

# Format-only reward ablation
uv run modal run --detach nanochat_modal.py::stage_rl_format_only

# Close-arithmetic-only reward ablation
uv run modal run --detach nanochat_modal.py::stage_rl_close_only

# Evaluate all Part 4 checkpoints
uv run modal run --detach nanochat_modal.py::stage_eval_part4

# Detailed per-problem GSM8K eval for Part 4 models
uv run modal run --detach nanochat_modal.py::stage_gsm8k_detailed_eval_part4
```

## Configuration

Key settings in `nanochat_modal.py`:

| Variable | Value | Description |
|----------|-------|-------------|
| `DEPTH` | `12` | Model depth (d12 baseline, ~286M params) |
| `NUM_SHARDS` | `8` | FineWeb-EDU pretraining shards |
| `GPU_FINETUNE` | `H100:4` | 4x H100 for SFT/RL |
| `WANDB_RUN` | `a4_baseline_rl_d12` | Default W&B run name |
| `FINETUNE_TIMEOUT_SEC` | `4 hours` | Container timeout for SFT/RL |

Checkpoints and data live on Modal Volume **nanochat-vol** and persist between runs.

## A3 Ablation Stages (legacy)

The following stages are from A3 and are not used in A4:

```bash
uv run modal run nanochat_modal.py::stage_pretrain_swiglu
uv run modal run nanochat_modal.py::stage_pretrain_rmsnorm
uv run modal run nanochat_modal.py::stage_post_pretrain_eval
```
