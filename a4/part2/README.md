# Part 2: SFT & Midtraining

## Overview

Run the nanochat **SFT** (Supervised Fine-Tuning) script on our pretrained **baseline `d12`** model
(step 2205) using the **original nanochat configuration** (no hyperparameter changes). The run is
logged to Weights & Biases, and benchmark evaluations (ChatCORE: ARC, MMLU, GSM8K, HumanEval,
SpellingBee) run automatically after training.

## Note on Midtraining

Karpathy removed the separate `mid_train.py` script from nanochat on January 31, 2025 (commit
`1ddaad1`). However, the midtraining *data* was not removed — all task datasets (MMLU, GSM8K,
SpellingBee, SimpleSpelling) were folded into `chat_sft.py` as a unified training mixture. We use
this current (post-January 2025) version, so our single SFT stage is functionally equivalent to
running both midtraining and SFT. The professor confirmed either the new or original scripts are
acceptable: *"you can use either the new chatSFT or the original scripts, just note which version."*

## Model Choice

We use the **baseline GPT d12** architecture rather than the SwiGLU variant from A3. The baseline
architecture matches Karpathy's original nanochat configuration exactly, ensuring our SFT and RL
results are directly comparable to the reference run without confounding architectural differences.
Both d12 variants achieved similar pretraining quality (Val BPB 0.8899 baseline vs. 0.9064 SwiGLU),
so this choice does not sacrifice model quality.

## Prerequisites

1. **Modal** installed and authenticated (`modal token set`)
2. **Modal secrets** configured:
   ```bash
   modal secret create nanochat-secrets WANDB_API_KEY=<key> HF_TOKEN=<token>
   ```
3. **Pretrained model** from A3 (`d12`, step 2205) on the Modal volume (`nanochat-vol`)
4. **nanochat repo** cloned into `nanochat-modal/nanochat/`:
   ```bash
   cd a4/nanochat-modal
   git clone https://github.com/karpathy/nanochat.git
   ```

## Repository Structure

```
a4/
├── A4_report.md                              # Main report
├── nanochat_chat_model_a4.py                 # Shared config (model tag, W&B names, GPU)
├── nanochat-modal/
│   ├── nanochat_modal.py                     # Modal entry point (stage_sft, stage_rl, etc.)
│   ├── pyproject.toml                        # Modal dependencies
│   ├── README.md                             # Modal setup instructions
│   └── nanochat/                             # ← clone nanochat repo here (gitignored)
│       └── scripts/
│           ├── chat_sft.py                   # SFT training script
│           ├── chat_rl.py                    # RL (GRPO) training script
│           ├── chat_eval.py                  # Benchmark evaluation script
│           ├── chat_rl_combined2rwd.py       # ★ Custom: RL with additional rewards (Part 4)
│           └── gsm8k_detailed_eval.py        # ★ Custom: Per-problem GSM8K evaluation
└── part2/
    ├── README.md                             # This file
    └── task1_results.md                      # Task 1 results and analysis
```

Files marked with ★ are our custom additions; everything else in `nanochat/` is upstream.

## Configuration

All A4 config lives in `a4/nanochat_chat_model_a4.py`:

| Variable | Value | Description |
|----------|-------|-------------|
| `A4_MODEL_TAG` | `d12` | Pretrained baseline model from A3 |
| `A4_MODEL_STEP` | `2205` | Pretrain checkpoint step |
| `WANDB_RUN_TASK1_SFT` | `a4_task1_sft` | W&B run name for SFT |
| `GPU_FINETUNE` | `H100:4` | 4× H100 for fine-tuning |

## How to Run

### Task 1: SFT with original configuration

```bash
cd a4/nanochat-modal

# Loads d12 baseline pretrained checkpoint (step 2205)
# Trains on default data mixture: SmolTalk + MMLU + GSM8K + SpellingBee + identity
# Evaluates on all ChatCORE benchmarks after training
# Logs to W&B project "nanochat-sft"
uv run modal run --detach nanochat_modal.py::stage_sft
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
