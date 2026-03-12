# Part 2 Task 2: SFT with Extra Dataset (OpenHermes-2.5)

## Overview

We ran the nanochat **SFT** script on the same pretrained `d12_swiglu` model (a3, step 2205) using the **original configuration plus OpenHermes-2.5** as the extra dataset. All other hyperparameters match Task 1 (same data mixture: SmolTalk, MMLU, GSM8K, SpellingBee, identity; same LR schedule, eval-every, etc.). The run was logged to Weights & Biases, and ChatCORE benchmarks ran automatically after training.

## W&B Run

| W&B Project | Run Name |
|-------------|----------|
| `nanochat-sft` | `a4_task2_sft` |

(Link: W&B project nanochat-sft, run `a4_task2_sft`.)

## Training Summary

| Metric | Value |
|--------|-------|
| Model | d12_swiglu (SwiGLU, 12 layers, n_embd=768) |
| Pretrain checkpoint | step 2205 |
| Extra dataset | OpenHermes-2.5 (JSONL, ~547K conversations) |
| Total SFT steps | 1,002 |
| Training time | ~4.38 min |
| GPU | 4× H100 80GB |
| Peak memory | 16,561.70 MiB |
| Final val BPB | 0.3664 |
| Final ChatCORE | 0.2400 |

### Key Config (same as Task 1 except data)

| Parameter | Value |
|-----------|-------|
| load_optimizer | 0 |
| num_iterations | -1 (full epoch) |
| init_lr_frac | 0.8 |
| warmdown_ratio | 0.5 |
| eval_every | 200 |
| chatcore_every | 200 |
| mmlu_epochs | 3 |
| gsm8k_epochs | 4 |
| DDP world size | 4 |

## Chat Evaluation (SFT checkpoint)

Evaluation run: 2026-03-12 20:13:20, source `sft`, model_tag `d12_swiglu`.

| Task | Accuracy | Notes |
|------|----------|--------|
| ARC-Easy (↑) | 35.40% | |
| ARC-Challenge (↑) | 33.02% | |
| MMLU (↑) | 31.21% | |
| GSM8K (↑) | 3.56% | |
| HumanEval (↑) | 9.15% | |
| SpellingBee (↑) | 98.44% | |
| **ChatCORE** (↑) | **0.2400** | Composite metric |

(Eval settings: temperature 0, max_new_tokens 512, batch_size 8.)

## Comparison: Task 1 vs Task 2

Same pretrained base (d12_swiglu, step 2205); Task 2 adds OpenHermes-2.5 to the SFT mixture.

### Benchmark Accuracy

| Task | Task 1 (original) | Task 2 (+ OpenHermes-2.5) | Δ |
|------|-------------------|---------------------------|---|
| ARC-Easy (↑) | 36.15% | 35.40% | -0.75% |
| ARC-Challenge (↑) | 30.12% | 33.02% | +2.90% |
| MMLU (↑) | 31.39% | 31.21% | -0.18% |
| GSM8K (↑) | 3.11% | 3.56% | +0.45% |
| HumanEval (↑) | 8.54% | 9.15% | +0.61% |
| SpellingBee (↑) | 98.44% | 98.44% | 0% |
| **ChatCORE** (↑) | **0.2380** | **0.2400** | **+0.0020** |

### Training

| Metric | Task 1 | Task 2 | Δ |
|--------|--------|--------|---|
| SFT steps | 969 | 1,002 | +33 |
| Final val BPB (↓) | 0.3683 | 0.3664 | -0.0019 (slightly better) |
| Training time | ~4.25 min | ~4.38 min | +~0.13 min |

## Analysis

### Effect of Adding OpenHermes-2.5

- **ChatCORE** improved slightly (0.2380 → 0.2400). Val BPB also improved slightly (0.3683 → 0.3664), so the extra instruction-style data did not hurt and may have helped overall calibration.
- **ARC-Challenge** improved by about 2.9 points (30.12% → 33.02%). OpenHermes-2.5 includes diverse reasoning and instruction-following data, which can help on harder multiple-choice reasoning.
- **HumanEval** improved slightly (8.54% → 9.15%). OpenHermes contains code-related data, which is consistent with a small gain in code generation.
- **GSM8K** improved slightly (3.11% → 3.56%). The mixture already had GSM8K; the extra generalist data may give a small boost without dedicated math RL.
- **ARC-Easy** and **MMLU** are essentially unchanged (small fluctuations within run variance). SpellingBee stays at 98.44%, as the original mixture already heavily covers that skill.

### Takeaways

1. **No regression**: Adding OpenHermes-2.5 did not degrade overall performance; ChatCORE and val BPB improved slightly.
2. **Largest gain on ARC-Challenge**: The main visible gain is on the harder ARC set, suggesting better instruction/reasoning diversity from OpenHermes.
3. **Code and math**: Small gains on HumanEval and GSM8K are consistent with more diverse instruction and possibly some code/math in OpenHermes, but SFT-only gains on these are still modest compared to what RL (e.g. Task 2 RL) could bring.
4. **Same compute footprint**: Training time and step count are very close to Task 1, so the extra data did not blow up cost while giving a modest improvement in composite and some tasks.

### Conclusion

Task 2 (SFT with OpenHermes-2.5) matches or slightly exceeds Task 1 on ChatCORE and val BPB, with the clearest gain on ARC-Challenge. The extra dataset is a low-risk way to add instruction diversity and a small boost on reasoning and code without changing hyperparameters or compute budget.
