# Part 2: SFT & Midtraining

## Overview

Run the nanochat **SFT** (Supervised Fine-Tuning) script on our pretrained `d12_swiglu` model
from A3, using the **original nanochat configuration** (no hyperparameter changes). The run is
logged to Weights & Biases, and benchmark evaluations (ChatCORE: ARC, MMLU, GSM8K, HumanEval,
SpellingBee) run automatically after training.

## Prerequisites

1. **Modal** installed and authenticated (`modal token set`)
2. **Modal secrets** configured:
   ```bash
   modal secret create nanochat-secrets WANDB_API_KEY=<key> HF_TOKEN=<token>
   ```
3. **Pretrained model** from A3 (`d12_swiglu`, step 2205) on the Modal volume (`nanochat-vol`)
4. **nanochat repo** cloned into `nanochat-modal/nanochat/`:
   ```bash
   cd a4/nanochat-modal
   git clone https://github.com/karpathy/nanochat.git
   ```
   Our custom scripts (listed below) are already in the repo and will overlay into the cloned tree.

## Repository Structure

Only our custom additions are tracked in git. The upstream nanochat repo is `.gitignore`d and
must be cloned locally before running.

```
a4/
├── A4_report.md                              # Main report
├── nanochat_chat_model_a4.py                 # Shared config (model tag, W&B names, GPU)
├── nanochat-modal/
│   ├── nanochat_modal.py                     # Modal entry point (stage_sft_task1, etc.)
│   ├── pyproject.toml                        # Modal dependencies
│   ├── README.md                             # Modal setup instructions
│   └── nanochat/                             # ← clone nanochat repo here (gitignored)
│       ├── nanochat/
│       │   ├── gpt_swiglu.py                 # ★ Custom: GPTSwiGLU architecture
│       │   └── gpt_rmsnorm.py                # ★ Custom: GPT with RMSNorm (from A3)
│       └── scripts/
│           ├── chat_sft_swiglu.py            # ★ Custom: patches GPT→GPTSwiGLU, runs chat_sft
│           ├── chat_rl_swiglu.py             # ★ Custom: patches GPT→GPTSwiGLU, runs chat_rl
│           ├── chat_eval_swiglu.py           # ★ Custom: patches GPT→GPTSwiGLU, runs chat_eval
│           ├── chat_cli_swiglu.py            # ★ Custom: patches GPT→GPTSwiGLU, runs chat_cli
│           ├── base_train_swiglu.py          # ★ Custom: SwiGLU pretraining script (from A3)
│           └── base_train_rmsnorm.py         # ★ Custom: RMSNorm pretraining script (from A3)
└── part2/
    ├── README.md                             # This file
    ├── task1_results.md                      # Task 1 results and analysis
    └── reports/                              # Modal training reports (PDF)
        ├── base-model-training.pdf           # Pretrained model baseline metrics
        ├── sft-training.pdf                  # SFT training process report
        ├── chat-evaluation-sft.pdf           # SFT benchmark evaluation results
        └── header.pdf                        # Model configuration metadata
```

Files marked with ★ are our custom additions; everything else in `nanochat/` is upstream.

## Why the `_swiglu` wrapper scripts?

Our pretrained model (`d12_swiglu`) uses the **GPTSwiGLU** architecture, which has a different
MLP structure (`gate`/`up`/`down`) than the standard GPT (`c_fc`/`c_proj`). The original
`chat_sft.py`, `chat_rl.py`, and `chat_eval.py` hardcode `from nanochat.gpt import GPT` and
would fail to load SwiGLU checkpoints.

Each wrapper script (e.g. `chat_sft_swiglu.py`) monkey-patches `GPT → GPTSwiGLU` before
running the original script, following the same pattern used in `chat_cli_swiglu.py`.

## Configuration

All A4 config lives in `a4/nanochat_chat_model_a4.py`:

| Variable | Value | Description |
|----------|-------|-------------|
| `A4_MODEL_TAG` | `d12_swiglu` | Pretrained model from A3 (SwiGLU variant) |
| `A4_MODEL_STEP` | `2205` | Pretrain checkpoint step |
| `WANDB_RUN_TASK1_SFT` | `a4_task1_sft` | W&B run name for SFT |
| `GPU_FINETUNE` | `H100:4` | 4× H100 for fine-tuning |

## How to Run

### Task 1: SFT with original configuration

```bash
cd a4/nanochat-modal

# Loads d12_swiglu pretrained checkpoint (step 2205)
# Trains on default data mixture: SmolTalk + MMLU + GSM8K + SpellingBee + identity
# Evaluates on all ChatCORE benchmarks after training
# Logs to W&B project "nanochat-sft", run "a4_task1_sft"
uv run modal run --detach nanochat_modal.py::stage_sft_task1
```

### Task 2: SFT with additional datasets

[TODO — not yet implemented]

## W&B Logging

**SFT** (project: `nanochat-sft`):
- `train/loss` — training loss curve
- `val/bpb` — validation bits-per-byte
- `chatcore_metric` — composite ChatCORE score
- `chatcore/GSM8K`, `chatcore/MMLU`, etc. — per-task accuracy

## Results

See `task1_results.md` for the full comparison between pretrained and SFT models, including
benchmark accuracy tables and analysis.
