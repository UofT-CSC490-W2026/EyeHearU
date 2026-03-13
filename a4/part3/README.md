# Part 3: Replicating RL Run with Additional Analysis

## Overview

Replicate Karpathy's RL run (GRPO on GSM8K) from the original nanochat repo,
compare training curves and eval results, and analyze problem correctness patterns.

## What was run

```bash
cd a4/nanochat-modal

# RL training (GRPO on GSM8K, 467 steps)
uv run modal run --detach nanochat_modal.py::stage_rl_task1

# Eval-only (re-run if eval crashed)
uv run modal run --detach nanochat_modal.py::stage_eval_rl_task1
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Base checkpoint | SFT d12_swiglu (step 969) |
| RL method | GRPO (Group Relative Policy Optimization) |
| Dataset | GSM8K train set (7473 problems) |
| Reward | Binary 0/1 (correct/incorrect) |
| Samples per question | 16 |
| Total RL steps | 467 |
| GPU | 4× H100 80GB |
| W&B run | [`a4_task1_rl`](https://wandb.ai/ysj15265673506-university-of-toronto/nanochat-rl/runs/l12kd4ni) |

## File Structure

```
part3/
├── README.md                      # This file
├── reports/
│   ├── chat-rl-training.pdf       # RL training process report (from Modal volume)
│   ├── chat-evaluation-rl.pdf     # RL benchmark evaluation report
│   └── rl-eval-raw.txt            # Raw eval data with all benchmark numbers
└── rl_results.md                  # Results comparison and analysis
```

## Key Results

### RL Training Summary

| Metric | Value |
|--------|-------|
| Total steps | 467 |
| Final pass@1 | 1.25% |
| Final pass@8 | 4.25% |
| Final reward | 0.059 |

### Benchmark: Pretrained → SFT → RL

| Task | Pretrained | After SFT | After RL | SFT→RL Change |
|------|-----------|-----------|----------|---------------|
| ARC-Easy | ~25% | 36.15% | 32.07% | −4.08% |
| ARC-Challenge | ~25% | 30.12% | 31.48% | +1.36% |
| MMLU | ~25% | 31.39% | 28.79% | −2.60% |
| GSM8K | ~0% | 3.11% | 4.32% | +1.21% |
| HumanEval | ~0% | 8.54% | 0.00% | −8.54% |
| SpellingBee | ~0% | 98.44% | 0.00% | −98.44% |
| ChatCORE | 0.1334 | 0.2380 | 0.0457 | −0.1923 |

## Data Sources

| Data point | Source |
|------------|--------|
| RL training metrics (reward, pass@k) | W&B run `a4_task1_rl` |
| RL benchmark accuracy | `stage_eval_rl_task1` output → `rl-eval-raw.txt` |
| SFT benchmark accuracy | Part 2 `task1_results.md` |
| Pretrained baseline | A3 pretrain run |
| Karpathy's original run | https://github.com/karpathy/nanochat/discussions/481 |

## TODO for Report

- [ ] Export reward curve and pass@1 curve plots from W&B
- [ ] Find Karpathy's original RL run data and overlay for comparison
- [ ] Comment on differences between our run and original
- [ ] Analyze which GSM8K problems were correct/incorrect
- [ ] Cluster problems into categories
- [ ] Conduct EDA on correctness patterns
