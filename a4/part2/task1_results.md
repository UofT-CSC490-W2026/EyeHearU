# Part 2 Task 1: SFT with Original Configuration

## Overview

We ran the nanochat **SFT** (Supervised Fine-Tuning) script on our pretrained **baseline `d12`** model
(step 2205) using the **original nanochat configuration** — no hyperparameter changes,
default data mixture and training schedule. The run was logged to Weights & Biases, and benchmark
evaluations (ChatCORE: ARC, MMLU, GSM8K, HumanEval, SpellingBee) ran automatically after training.

## Note on Midtraining

Karpathy removed the separate `mid_train.py` script from nanochat on January 31, 2025 (commit
`1ddaad1`), but the midtraining data (MMLU, GSM8K, SpellingBee, SimpleSpelling) was folded into
`chat_sft.py` as a unified mixture. We use this current version, so our single SFT stage is
functionally equivalent to running both midtraining and SFT.

## Model Choice Justification

We chose the **baseline GPT d12** architecture over the SwiGLU variant from A3 for the following reasons:

1. **Direct comparability to Karpathy's reference run.** The assignment asks us to "replicate the run
   in github where Karpathy trains the model on GSM8k." Karpathy's original nanochat uses the standard
   GPT architecture, so using the same architecture ensures our results are directly comparable without
   confounding variables from architectural differences.

2. **No monkey-patching complexity.** The SwiGLU variant requires custom wrapper scripts
   (`chat_sft_swiglu.py`, `chat_rl_swiglu.py`, etc.) that monkey-patch the model class at import time.
   Using the baseline architecture lets us use the upstream nanochat scripts directly, reducing the risk
   of subtle bugs.

3. **Similar pretraining quality.** Both d12 variants achieved comparable pretraining quality
   (Val BPB 0.8899 baseline vs. 0.9064 SwiGLU, CORE 0.1186 vs. 0.1334), so this choice does not
   sacrifice meaningful model capability.

The professor confirmed that any variant or baseline is acceptable as long as the choice is justified.

## Data Sources

| Data point | Source |
|------------|--------|
| SFT training metrics (loss, BPB, steps) | Modal `stage_sft` terminal logs + W&B |
| SFT benchmark accuracy (ARC, MMLU, GSM8K, etc.) | `chat_eval -i sft --model-tag=d12` output |
| Pretrained Val BPB (0.8899) and CORE (0.1186) | A3 pretrain run |
| Pretrained params and tokens | A3 pretrain training header |
| Pretrained benchmark baselines (~25%, ~0%) | Theoretical random-guess values |

## Training Summary

| Metric | Value |
|--------|-------|
| Model | d12 (GPT baseline, 12 layers, n_embd=768) |
| Pretrain checkpoint | step 2205 |
| Total SFT steps | 969 |
| GPU | 4× H100 80GB |
| Final val BPB | 0.3688 |
| Final ChatCORE | 0.2375 |

## Results Comparison: Pretrained vs After SFT

### Benchmark Accuracy

| Task | Pretrained (d12) | After SFT | Change |
|------|------------------|-----------|--------|
| ARC-Easy (↑) | ~25% (random) | 36.20% | +11.20% |
| ARC-Challenge (↑) | ~25% (random) | 32.85% | +7.85% |
| MMLU (↑) | ~25% (random) | 30.71% | +5.71% |
| GSM8K (↑) | ~0% | 3.56% | +3.56% |
| HumanEval (↑) | ~0% | 6.71% | +6.71% |
| SpellingBee (↑) | ~0% | 99.22% | +99.22% |
| **ChatCORE** (↑) | N/A | **0.2375** | — |

### Loss

| Metric | Pretrained | After SFT | Change |
|--------|-----------|-----------|--------|
| Val BPB (↓) | 0.8899 | 0.3688 | −58.6% |
| CORE | 0.1186 | N/A (ChatCORE = 0.2375) | — |

### Pretrained Baseline (from A3)

| Metric | Value | Source |
|--------|-------|--------|
| Val BPB | 0.8899 | A3 pretrain training report |
| CORE | 0.1186 | A3 `stage_post_pretrain_eval` output |
| Total Params | 286,262,424 | A3 pretrain training header |
| Train Tokens | 1,156,055,040 | A3 pretrain training header |

Categorical benchmark baselines (ARC ~25%, MMLU ~25%) are theoretical random-guess
values for 1-of-4 multiple choice. Generative baselines (GSM8K ~0%, HumanEval ~0%,
SpellingBee ~0%) reflect that the pretrained model has not learned chat format or
tool-use tokens, so it cannot produce valid answers for these tasks.

## Analysis

### Impact of SFT

- **Val BPB dropped dramatically** (0.8899 → 0.3688, −58.6%), showing the model learned
  conversational and task-specific patterns much better than raw pretraining.

- **Categorical benchmarks improved beyond random baseline**: ARC-Easy rose from ~25% to 36.20%,
  ARC-Challenge from ~25% to 32.85%, and MMLU from ~25% to 30.71%. SFT teaches the model the
  multiple-choice answer format, which is why these tasks see clear gains.

- **SpellingBee reached near-perfect accuracy** (99.22%), since the SFT data mixture explicitly
  includes 200K SimpleSpelling and 80K SpellingBee examples.

- **GSM8K improved modestly** (0% → 3.56%). The SFT mixture includes GSM8K examples with
  calculator tool use, but without RL-based optimization, the model struggles with multi-step
  math reasoning.

- **HumanEval showed early signs of coding ability** (0% → 6.71%), enabled by exposure to
  structured code generation during SFT.

### Key Takeaways

SFT is essential for converting a raw pretrained model into a useful chat model. The largest
gains come from format learning (multiple choice, tool use, spelling) rather than deep reasoning.
Tasks requiring multi-step reasoning like GSM8K show only modest improvement from SFT alone —
further gains require RL-based training (Part 3).
