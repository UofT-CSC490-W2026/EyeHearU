# Part 2 Task 1: SFT with Original Configuration

## Overview

We ran the nanochat **SFT** (Supervised Fine-Tuning) script on our pretrained `d12_swiglu` model
(from a3, step 2205) using the **original nanochat configuration** — no hyperparameter changes,
default data mixture and training schedule. The run was logged to Weights & Biases, and benchmark
evaluations (ChatCORE: ARC, MMLU, GSM8K, HumanEval, SpellingBee) ran automatically after training.

## W&B Run

| W&B Project | Run Name | Link |
|-------------|----------|------|
| `nanochat-sft` | `a4_task1_sft` | https://wandb.ai/ysj15265673506-university-of-toronto/nanochat-sft/runs/pb7f6eur |

## Data Sources

| Data point | Source |
|------------|--------|
| SFT training metrics (loss, BPB, ChatCORE, steps) | W&B run `a4_task1_sft` + Modal terminal logs from `stage_sft_task1` |
| SFT benchmark accuracy (ARC, MMLU, GSM8K, etc.) | `chat_eval_swiglu -i sft` output at end of `stage_sft_task1` |
| Pretrained Val BPB (0.9064) and CORE (0.1334) | a3 pretrain run, recorded in `a4/part4/compair_results.md` table row "picochat (SwiGLU)" |
| Pretrained params and tokens | a3 pretrain training header |
| Pretrained benchmark baselines (~25%, ~0%) | Theoretical random-guess values (pretrained model has no chat format knowledge) |

## Training Summary

| Metric | Value |
|--------|-------|
| Model | d12_swiglu (SwiGLU, 12 layers, n_embd=768) |
| Pretrain checkpoint | step 2205 |
| Total SFT steps | 969 |
| Training time | 4.25 min |
| GPU | 4x H100 80GB |
| Peak memory | 16,559.95 MiB |
| Final val BPB | 0.3683 |
| Final ChatCORE | 0.2380 |

### Val BPB over training

| Step | Val BPB |
|------|---------|
| 0    | 0.6424  |
| 200  | 0.4432  |
| 400  | 0.4244  |
| 600  | 0.4031  |
| 800  | 0.3798  |
| 969  | 0.3683  |

### ChatCORE over training

| Step | ChatCORE | ChatCORE_cat |
|------|----------|--------------|
| 200  | 0.1834   | 0.0751       |
| 400  | 0.2034   | 0.0734       |
| 600  | 0.2125   | 0.0777       |
| 800  | 0.2234   | 0.0856       |
| 969  | 0.2380   | 0.1009       |

## Results Comparison: Pretrained vs After SFT

### Benchmark Accuracy

| Task | Pretrained (d12_swiglu) | After SFT | Change |
|------|------------------------|-----------|--------|
| ARC-Easy (↑) | ~25% (random) | 36.15% | +11.15% |
| ARC-Challenge (↑) | ~25% (random) | 30.12% | +5.12% |
| MMLU (↑) | ~25% (random) | 31.39% | +6.39% |
| GSM8K (↑) | ~0% | 3.11% | +3.11% |
| HumanEval (↑) | ~0% | 8.54% | +8.54% |
| SpellingBee (↑) | ~0% | 98.44% | +98.44% |
| **ChatCORE** (↑) | N/A | **0.2380** | — |

### Loss

| Metric | Pretrained | After SFT | Change |
|--------|-----------|-----------|--------|
| Val BPB (↓) | 0.9064 | 0.3683 | -59.4% |
| CORE | 0.1334 | N/A (ChatCORE = 0.2380) | — |

### Pretrained Baseline (from a3)

| Metric | Value | Source |
|--------|-------|--------|
| Val BPB | 0.9064 | a3 pretrain training header, also in `a4/part4/compair_results.md` |
| CORE | 0.1334 | a3 `stage_post_pretrain_eval` output, also in `a4/part4/compair_results.md` |
| Total Params | 286,262,424 | a3 pretrain training header |
| Train Tokens | 1,156,055,040 | a3 pretrain training header |

Categorical benchmark baselines (ARC ~25%, MMLU ~25%) are theoretical random-guess
values for 1-of-4 multiple choice. Generative baselines (GSM8K ~0%, HumanEval ~0%,
SpellingBee ~0%) reflect that the pretrained model has not learned chat format or
tool-use tokens, so it cannot produce valid answers for these tasks.

## Analysis

### Impact of SFT

- **Val BPB dropped dramatically** (0.9064 → 0.3683, -59.4%), showing the model learned
  conversational and task-specific patterns much better than raw pretraining.

- **Categorical benchmarks improved beyond random baseline**: ARC-Easy rose from ~25% to 36.15%,
  ARC-Challenge from ~25% to 30.12%, and MMLU from ~25% to 31.39%. SFT teaches the model the
  multiple-choice answer format, which is why these tasks see clear gains.

- **SpellingBee reached near-perfect accuracy** (98.44%), since the SFT data mixture explicitly
  includes 200K SimpleSpelling and 80K SpellingBee examples.

- **GSM8K improved modestly** (0% → 3.11%). The SFT mixture includes GSM8K examples with
  calculator tool use, but without RL-based optimization, the model struggles with multi-step
  math reasoning.

- **HumanEval showed early signs of coding ability** (0% → 8.54%), enabled by exposure to
  structured code generation during SFT.

- **ChatCORE steadily improved** throughout training (0.1834 at step 200 → 0.2380 at step 969),
  indicating broad capability gains across all evaluated tasks.

### Key Takeaways

SFT is essential for converting a raw pretrained model into a useful chat model. The largest
gains come from format learning (multiple choice, tool use, spelling) rather than deep reasoning.
Tasks requiring multi-step reasoning like GSM8K show only modest improvement from SFT alone —
further gains would require RL-based training (Part 3).
