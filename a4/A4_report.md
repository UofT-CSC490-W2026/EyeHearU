# CSC490 Assignment A4 — RL-ing Nanochat

**Team: EyeHearU**

| Name | Student ID |
|------|------------|
| TODO | TODO |
| TODO | TODO |
| TODO | TODO |

---

## 1. Part One: GRPO and RL Review (10 marks)

<!-- TODO: Write a short paragraph comparing nanochat's RL implementation to standard GRPO -->

[TODO — Placeholder]

Compare nanochat's RL implementation (`scripts/chat_rl.py`) to the standard GRPO formulation from Shao et al. (2024). Key differences to discuss:

- How nanochat samples and scores completions within a group
- Advantage computation (group-relative vs. baseline-subtracted)
- KL penalty handling
- Why Karpathy may have simplified or diverged from the paper

---

## 2. Part Two: SFT & Midtraining (20 marks)

### 2.1 Original Configuration SFT (Bullet 1)

We ran the nanochat SFT script on our pretrained `d12_swiglu` model (from A3, step 2205) using the **original nanochat configuration** — no hyperparameter changes, default data mixture and training schedule. The run was logged to Weights & Biases.

**W&B Run:** [`a4_task1_sft`](https://wandb.ai/ysj15265673506-university-of-toronto/nanochat-sft/runs/pb7f6eur)

#### Model & Training Setup

| Parameter | Value |
|-----------|-------|
| Architecture | GPTSwiGLU (12 layers, n_embd=768) |
| Pretrain checkpoint | step 2205 (from A3) |
| Total SFT steps | 969 |
| Training time | 4.25 min |
| GPU | 4× H100 80GB |
| Peak memory | 16,559.95 MiB |

#### SFT Training Curves

**Val BPB over training:**

| Step | Val BPB |
|------|---------|
| 0    | 0.6424  |
| 200  | 0.4432  |
| 400  | 0.4244  |
| 600  | 0.4031  |
| 800  | 0.3798  |
| 969  | 0.3683  |

**ChatCORE over training:**

| Step | ChatCORE | ChatCORE_cat |
|------|----------|--------------|
| 200  | 0.1834   | 0.0751       |
| 400  | 0.2034   | 0.0734       |
| 600  | 0.2125   | 0.0777       |
| 800  | 0.2234   | 0.0856       |
| 969  | 0.2380   | 0.1009       |

#### Comparison: Pretrained vs. After SFT

**Benchmark Accuracy:**

| Task | Pretrained (d12\_swiglu) | After SFT | Change |
|------|--------------------------|-----------|--------|
| ARC-Easy (↑) | ~25% (random) | 36.15% | +11.15% |
| ARC-Challenge (↑) | ~25% (random) | 30.12% | +5.12% |
| MMLU (↑) | ~25% (random) | 31.39% | +6.39% |
| GSM8K (↑) | ~0% | 3.11% | +3.11% |
| HumanEval (↑) | ~0% | 8.54% | +8.54% |
| SpellingBee (↑) | ~0% | 98.44% | +98.44% |
| **ChatCORE** (↑) | N/A | **0.2380** | — |

**Loss:**

| Metric | Pretrained | After SFT | Change |
|--------|-----------|-----------|--------|
| Val BPB (↓) | 0.9064 | 0.3683 | −59.4% |
| CORE | 0.1334 | N/A (ChatCORE = 0.2380) | — |

The pretrained baseline metrics (Val BPB 0.9064, CORE 0.1334) come from our A3 pretraining run. Categorical benchmarks assume ~25% random-guess baselines for 4-choice tasks (ARC, MMLU). Generative tasks (GSM8K, HumanEval, SpellingBee) start near 0% because the pretrained model has no knowledge of chat format or tool-use tokens.

#### Analysis

**Val BPB dropped dramatically** (0.9064 → 0.3683, −59.4%), confirming the model learned conversational and task-specific patterns far beyond what raw pretraining provides.

**Categorical benchmarks improved beyond random baseline.** ARC-Easy rose from ~25% to 36.15%, ARC-Challenge from ~25% to 30.12%, and MMLU from ~25% to 31.39%. SFT teaches the model the multiple-choice answer format, which accounts for these gains.

**SpellingBee reached near-perfect accuracy** (98.44%). The SFT data mixture explicitly includes 200K SimpleSpelling and 80K SpellingBee examples, so this is expected.

**GSM8K improved modestly** (0% → 3.11%). While the SFT mixture includes GSM8K examples with calculator tool use, multi-step math reasoning requires RL-based optimization to improve significantly.

**HumanEval showed early coding ability** (0% → 8.54%), enabled by exposure to structured code generation during SFT.

**ChatCORE improved steadily** throughout training (0.1834 at step 200 → 0.2380 at step 969), indicating broad capability gains across all evaluated tasks.

**Key takeaway:** SFT converts a raw pretrained model into a functional chat model. The largest gains come from format learning (multiple choice, tool use, spelling) rather than deep reasoning. Tasks like GSM8K that require multi-step reasoning show only modest gains from SFT alone — further improvement requires RL (Part 3).

#### Data Sources

| Data point | Source |
|------------|--------|
| SFT training metrics | W&B run `a4_task1_sft` + Modal terminal logs |
| SFT benchmark accuracy | `chat_eval_swiglu -i sft` output |
| Pretrained Val BPB (0.9064), CORE (0.1334) | A3 pretrain run |
| Pretrained benchmark baselines | Theoretical random-guess values |

### 2.2 Additional Datasets for SFT (Bullet 2)

<!-- TODO: Find additional datasets, justify choices, run SFT, compare results -->

[TODO — Placeholder]

- Dataset selection and justification
- Training with same configuration
- Results comparison to Section 2.1

---

## 3. Part Three: Replicating RL Run (30 marks)

<!-- TODO: Run RL, compare to Karpathy's original, plot reward/eval curves, cluster problems -->

### 3.1 RL Training Replication

[TODO — Placeholder]

- Replicate nanochat RL run on GSM8K
- Compare training runs to original
- Reward curves and eval curves plots

### 3.2 Problem Analysis and Clustering

[TODO — Placeholder]

- Review correct vs. incorrect problems
- Cluster into categories
- Exploratory data analysis

---

## 4. Part Four: Complex Reward System (40 marks)

<!-- TODO: Design additional rewards, run experiments, compare, visualize -->

### 4.1 Additional Reward Design

[TODO — Placeholder]

- Describe 2+ additional reward systems
- Motivation from Part 3 analysis

### 4.2 Combined Reward Training

[TODO — Placeholder]

- Run with combined rewards
- Compare to original RL run

### 4.3 Separate Environment Training

[TODO — Placeholder]

- Run each reward system in separate environments
- Compare to combined runs

### 4.4 Error Analysis

[TODO — Placeholder]

- Compare mistake types: Original RL vs. RL with additional rewards
- Visualizations

### 4.5 Summary Table

[TODO — Placeholder]

| Run | Config | GSM8K Acc | ChatCORE | Notes |
|-----|--------|-----------|----------|-------|
| Pretrained | — | ~0% | N/A | Baseline |
| After SFT | Original | 3.11% | 0.2380 | Part 2.1 |
| After RL | Original | TODO | TODO | Part 3 |
| RL + Reward A | TODO | TODO | TODO | Part 4 |
| RL + Reward B | TODO | TODO | TODO | Part 4 |
| RL + Reward A (separate) | TODO | TODO | TODO | Part 4 |
| RL + Reward B (separate) | TODO | TODO | TODO | Part 4 |

---

## References

- Karpathy, A. (2025). nanochat: A tiny chatbot arena and training harness. https://github.com/karpathy/nanochat/discussions/481
- Shao, Z., Wang, P., Zhu, Q., et al. (2024). GRPO: Group Relative Policy Optimization for Language Model Alignment. arXiv preprint arXiv:2402.05191.
- Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168.
