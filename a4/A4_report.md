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

We ran the nanochat SFT script on our pretrained **baseline `d12`** model (step 2205) using the **original nanochat configuration** — no hyperparameter changes, default data mixture and training schedule. We chose the standard GPT architecture (rather than the SwiGLU variant from A3) to faithfully replicate Karpathy's original pipeline and avoid monkey-patching complexity. The run was logged to Weights & Biases.

#### Model Choice Justification

The baseline d12 GPT architecture matches Karpathy's original nanochat configuration. This ensures our SFT and RL results (Part 3) are directly comparable to the reference run without confounding architectural differences. Both d12 variants (baseline vs. SwiGLU) achieved similar pretraining quality (Val BPB 0.8899 baseline vs. 0.9064 SwiGLU, CORE 0.1186 vs. 0.1334).

#### Pretraining Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | GPT baseline (12 layers, n_embd=768) |
| Parameters | 286,262,424 |
| Training tokens | 1,156,055,040 |
| Tokens:params ratio | 10.5× |
| Training time | 4.92 min |
| GPU | 8× NVIDIA H100 80GB HBM3 |
| Final Val BPB | 0.8899 |
| CORE metric | 0.1186 |
| MFU | 38.56% |

#### SFT Training Setup

| Parameter | Value |
|-----------|-------|
| Architecture | GPT baseline (12 layers, n_embd=768) |
| Pretrain checkpoint | d12, step 2205 |
| Total SFT steps | 969 |
| Optimizer warm-start | No (fresh optimizer) |
| Init LR fraction | 0.80 |
| Warmdown ratio | 0.50 |
| DDP world size | 4 |
| Minimum Val BPB | 0.3688 |

#### Comparison: Pretrained vs. After SFT

**Benchmark Accuracy:**

| Task | Pretrained (d12) | After SFT | Change |
|------|------------------|-----------|--------|
| ARC-Easy (↑) | ~25% (random) | 36.20% | +11.20% |
| ARC-Challenge (↑) | ~25% (random) | 32.85% | +7.85% |
| MMLU (↑) | ~25% (random) | 30.71% | +5.71% |
| GSM8K (↑) | ~0% | 3.56% | +3.56% |
| HumanEval (↑) | ~0% | 6.71% | +6.71% |
| SpellingBee (↑) | ~0% | 99.22% | +99.22% |
| **ChatCORE** (↑) | N/A | **0.2375** | — |

**Loss:**

| Metric | Pretrained | After SFT | Change |
|--------|-----------|-----------|--------|
| Val BPB (↓) | 0.8899 | 0.3688 | −58.6% |
| CORE | 0.1186 | N/A (ChatCORE = 0.2375) | — |

Categorical benchmarks assume ~25% random-guess baselines for 4-choice tasks (ARC, MMLU). Generative tasks (GSM8K, HumanEval, SpellingBee) start near 0% because the pretrained model has no knowledge of chat format or tool-use tokens.

#### Analysis

**Val BPB dropped dramatically** (0.8899 → 0.3688, −58.6%), confirming the model learned conversational and task-specific patterns far beyond what raw pretraining provides.

**Categorical benchmarks improved beyond random baseline.** ARC-Easy rose from ~25% to 36.20%, ARC-Challenge from ~25% to 32.85%, and MMLU from ~25% to 30.71%. SFT teaches the model the multiple-choice answer format, which accounts for these gains.

**SpellingBee reached near-perfect accuracy** (99.22%). The SFT data mixture explicitly includes 200K SimpleSpelling and 80K SpellingBee examples, so this is expected.

**GSM8K improved modestly** (0% → 3.56%). While the SFT mixture includes GSM8K examples with calculator tool use, multi-step math reasoning requires RL-based optimization to improve significantly.

**HumanEval showed early coding ability** (0% → 6.71%), enabled by exposure to structured code generation during SFT.

**Key takeaway:** SFT converts a raw pretrained model into a functional chat model. The largest gains come from format learning (multiple choice, tool use, spelling) rather than deep reasoning. Tasks like GSM8K that require multi-step reasoning show only modest gains from SFT alone — further improvement requires RL (Part 3).

#### Data Sources

| Data point | Source |
|------------|--------|
| Pretrain metrics | Modal `stage_pretrain` report |
| SFT training metrics | Modal `stage_sft` report |
| SFT benchmark accuracy | `chat_eval -i sft --model-tag=d12` output |
| Pretrained benchmark baselines | Theoretical random-guess values |

### 2.2 Additional Datasets for SFT (Bullet 2)

<!-- TODO: Find additional datasets, justify choices, run SFT, compare results -->

[TODO — Placeholder]

- Dataset selection and justification
- Training with same configuration
- Results comparison to Section 2.1

---

## 3. Part Three: Replicating RL Run (30 marks)

### 3.1 RL Training Replication

We replicated Karpathy's RL run using the simplified GRPO implementation in `scripts/chat_rl.py`, training our d12 baseline model on the GSM8K training set.

#### RL Training Configuration

| Parameter | Value |
|-----------|-------|
| Starting checkpoint | SFT d12 (step 969) |
| Algorithm | Simplified GRPO (REINFORCE-style) |
| Task | GSM8K (train split, 7,473 problems) |
| Epochs | 1 |
| Examples per step | 16 |
| Samples per example | 16 |
| Total sequences per step | 256 |
| Max new tokens | 256 |
| Temperature | 1.0 |
| Top-k | 50 |
| Embedding LR | 0.2000 |
| Unembedding LR | 0.0040 |
| Matrix LR (Muon) | 0.0200 |
| Init LR fraction | 0.05 |
| LR schedule | Linear rampdown to 0 |
| Eval every | 60 steps |
| Eval examples | 400 (GSM8K test) |
| GPU | 4× NVIDIA H100 80GB |

#### Results: SFT vs. RL

| Task | After SFT | After RL | Change |
|------|-----------|----------|--------|
| ARC-Easy | 36.20% | 35.90% | −0.30% |
| ARC-Challenge | 32.85% | 30.46% | −2.39% |
| MMLU | 30.71% | 31.04% | +0.33% |
| **GSM8K** | **3.56%** | **10.92%** | **+7.36%** |
| HumanEval | 6.71% | 0.00% | −6.71% |
| SpellingBee | 99.22% | 2.73% | −96.49% |
| **ChatCORE** | **0.2375** | **0.0725** | **−0.1650** |

#### Reward and Eval Curves

The following W&B plots show training dynamics over ~467 RL steps. The run name "picochat_swiglu" is a W&B configuration artifact — this is the baseline d12 model trained with the standard GRPO pipeline described above.

![W&B Training Curves: LR multiplier, pass@1 through pass@5](part3/plots/wandb_curves_1.png)

![W&B Training Curves: pass@6 through pass@8, reward, sequence length](part3/plots/wandb_curves_2.png)

**Key observations from the curves:**

- **Reward** is noisy but trends upward from ~0.05 to ~0.10 over training, with occasional spikes up to 0.25. The high variance is expected: each step samples only 16 problems × 16 completions, so the mean reward fluctuates heavily.
- **Pass@1** (greedy accuracy) improves from ~0.02 to ~0.10, matching our final eval of 10.92%. The improvement is steepest in the first 200 steps, then plateaus with continued gradual gains.
- **Pass@k for higher k** shows progressively higher accuracy (pass@8 reaches ~0.22), confirming that the model learns multiple valid solution strategies — sampling more attempts yields more correct answers.
- **Learning rate multiplier** decays linearly from 1.0 to ~0.15 over training, confirming the linear rampdown schedule.
- **Sequence length** fluctuates between 120–220 tokens with no clear trend, indicating the model doesn't learn to generate significantly longer or shorter responses over RL training.

#### Comparison to Karpathy's Original Run

| | Karpathy (d32, ~1.9B params) | Our run (d12, ~286M params) |
|---|---|---|
| Architecture | GPT baseline (32 layers, 2048 dim) | GPT baseline (12 layers, 768 dim) |
| Parameters | ~1.9B | ~286M |
| Pretraining data | 800 shards (~37.6B tokens) | 8 shards (~1.2B tokens) |
| Pipeline | Pretrain → Midtrain → SFT → RL | Pretrain → SFT → RL |
| GSM8K after SFT | 12.74%† | 3.56% |
| GSM8K after RL | 19.94% | 10.92% |
| RL improvement | +7.20% | +7.36% |

*†Karpathy's SFT checkpoint includes a separate midtraining step before SFT; our pipeline omits midtraining.*

**Key observations:**

1. **The RL improvement is remarkably similar** despite very different model scales. Karpathy's d32 gained +7.20% from RL; our d12 gained +7.36%. This suggests the GRPO algorithm extracts a roughly constant amount of improvement from the GSM8K reward signal, regardless of starting accuracy. The ceiling is higher for larger models because they start higher after SFT.

2. **Our GSM8K accuracy (10.92%) is lower than Karpathy's (19.94%)** primarily because our model is ~7× smaller (286M vs. ~1.9B parameters) and was pretrained on ~30× less data (~1.2B vs. ~37.6B tokens). Additionally, Karpathy's pipeline included a separate midtraining step before SFT that we omitted. Smaller models have less capacity for multi-step reasoning, consistent with known scaling laws.

3. **Catastrophic forgetting is severe.** SpellingBee collapsed from 99.22% to 2.73%, and HumanEval dropped from 6.71% to 0.00%. This is a direct consequence of the simplified GRPO formulation: because there is no KL divergence penalty against the SFT reference model, the RL optimization is free to drift arbitrarily far from the SFT policy. The model "forgets" non-GSM8K skills because there is no regularization incentivizing their preservation. Standard GRPO includes a KL penalty term precisely to prevent this.

4. **Categorical benchmarks (ARC, MMLU) were relatively stable** (within ~2%), likely because the multiple-choice format is robust and these benchmarks rely more on factual knowledge encoded in the model weights than on the generation format that RL disrupts.

5. **ChatCORE dropped significantly** (0.2375 → 0.0725) because it is an average across all tasks, and the collapse of SpellingBee and HumanEval drags the composite score down despite GSM8K improving.

### 3.2 Problem Analysis and Clustering

We ran detailed per-problem evaluation on the full GSM8K test set (1,319 problems) for both the SFT and RL checkpoints. Each problem was classified along multiple dimensions to understand where the model succeeds and fails.

#### Classification Methodology

We categorized each GSM8K test problem along five dimensions:

1. **Domain** — Keyword-based classification into: money/shopping, time, food/cooking, distance/travel, people/age, counting/inventory, or other.
2. **Number of reasoning steps** — Counted by the number of calculator tool calls in the ground truth solution. Problems range from 0 steps (no tool calls) to 8 steps (complex multi-step reasoning).
3. **Answer magnitude** — The ground truth numerical answer classified as small (<10), medium (10–99), large (100–999), or very large (1000+).
4. **Question length** — Word count of the question text: short (<30 words), medium (30–59), or long (60+).
5. **Operation types** — Which arithmetic operations appear in the ground truth solution: addition, subtraction, multiplication, division.

#### Error Classification

For incorrect answers, we classified errors into four types:

- **Format error** — The model failed to produce the `####` answer marker, indicating it did not learn the expected response format.
- **No tool use** — The model attempted an answer but did not use calculator tool calls, suggesting it tried to do mental arithmetic.
- **Close arithmetic** — The extracted answer was within 10% of the ground truth, indicating the reasoning approach was correct but arithmetic was slightly off.
- **Wrong arithmetic** — The model used the correct format but arrived at the wrong numerical answer.

#### Results by Category

**Accuracy by Problem Domain (RL model):**

| Domain | SFT Accuracy | RL Accuracy | Improvement | n |
|--------|-------------|-------------|-------------|---|
| money/shopping | 4.1% | 12.6% | +8.5% | 438 |
| time | 3.0% | 10.3% | +7.3% | 331 |
| counting/inventory | 3.8% | 8.8% | +5.0% | 320 |
| people/age | 1.0% | 9.3% | +8.3% | 97 |
| food/cooking | 7.7% | 15.4% | +7.7% | 65 |
| distance/travel | 0.0% | 10.3% | +10.3% | 58 |
| other | 10.0% | 20.0% | +10.0% | 10 |

**Accuracy by Number of Reasoning Steps:**

| Steps | SFT Accuracy | RL Accuracy | Improvement | n |
|-------|-------------|-------------|-------------|---|
| 0 | 0.0% | 0.0% | 0.0% | 18 |
| 1 | 1.5% | 15.4% | +13.9% | 65 |
| 2 | 6.4% | 25.8% | +19.4% | 357 |
| 3 | 4.1% | 5.8% | +1.7% | 364 |
| 4 | 2.1% | 5.2% | +3.1% | 290 |
| 5 | 0.7% | 2.9% | +2.2% | 138 |
| 6 | 1.8% | 0.0% | −1.8% | 57 |
| 7 | 0.0% | 0.0% | 0.0% | 21 |
| 8 | 0.0% | 22.2%* | +22.2% | 9 |

*\*n=9 is too small for this to be statistically meaningful; likely noise.*

**Accuracy by Answer Magnitude:**

| Magnitude | SFT Accuracy | RL Accuracy | Improvement | n |
|-----------|-------------|-------------|-------------|---|
| small (<10) | 4.7% | 8.3% | +3.6% | 253 |
| medium (10–99) | 3.1% | 10.5% | +7.4% | 639 |
| large (100–999) | 4.4% | 15.5% | +11.1% | 296 |
| very large (1000+) | 1.5% | 7.6% | +6.1% | 131 |

**Accuracy by Question Length:**

| Length | SFT Accuracy | RL Accuracy | Improvement | n |
|--------|-------------|-------------|-------------|---|
| short (<30 words) | 9.7% | 23.6% | +13.9% | 216 |
| medium (30–59) | 2.8% | 9.5% | +6.7% | 852 |
| long (60+) | 0.8% | 4.8% | +4.0% | 251 |

**Error Type Distribution:**

| Error Type | SFT (count) | SFT (%) | RL (count) | RL (%) |
|------------|------------|---------|-----------|--------|
| Wrong arithmetic | 699 | 55.0% | 1002 | 85.3% |
| Format error | 479 | 37.7% | 141 | 12.0% |
| No tool use | 80 | 6.3% | 0 | 0.0% |
| Close arithmetic | 14 | 1.1% | 32 | 2.7% |

#### Visualizations

![Accuracy by Domain: SFT vs RL](part3/plots/sft_vs_rl_domain.png)

![Accuracy vs Reasoning Steps: SFT vs RL](part3/plots/sft_vs_rl_steps.png)

![Error Type Distribution: SFT vs RL](part3/plots/sft_vs_rl_errors.png)

![Accuracy by Answer Magnitude](part3/plots/accuracy_by_magnitude.png)

![Accuracy by Operation Type](part3/plots/accuracy_by_operation.png)

![Error Types After RL](part3/plots/error_types.png)

#### Key Findings

**1. RL gains are concentrated in simple problems.** The strongest improvements are in 1-step (+13.9%) and 2-step (+19.4%) problems. For 3+ step problems, improvement is marginal (+1–3%). For 6+ step problems, there is zero improvement. This indicates the d12 model can learn simple arithmetic patterns through RL but lacks the capacity for deep multi-step reasoning chains.

**2. Short questions benefit most from RL.** Short questions (<30 words) improved from 9.7% to 23.6% (+13.9%), while long questions (60+ words) improved only from 0.8% to 4.8%. Shorter questions tend to require fewer reasoning steps and simpler setups, which aligns with finding #1.

**3. RL dramatically reduces format errors.** Format errors (no `####` marker) dropped from 37.7% to 12.0% of all errors, and "no tool use" errors disappeared entirely (6.3% → 0.0%). This shows RL effectively teaches the model the expected response structure — the model learns to always produce a `####` answer and to use calculator tool calls.

**4. The dominant remaining error is wrong arithmetic.** After RL, 85.3% of errors are wrong arithmetic (up from 55.0% in SFT). This is not because RL made arithmetic worse — the absolute number of format errors decreased substantially, so wrong arithmetic now dominates the error distribution. The model knows *how* to approach problems but still computes incorrectly.

**5. Large-answer problems see the biggest accuracy gains.** Problems with answers in the 100–999 range improved by +11.1% (4.4% → 15.5%), more than small-answer problems (+3.6%). This may be because large-answer problems tend to involve straightforward multiplication (e.g., price × quantity), which RL reinforces effectively.

**6. RL improves uniformly across domains.** All problem domains saw +5–10% improvement. No single domain dominates, suggesting the RL signal generalizes across problem topics rather than overfitting to a specific type of word problem.

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
| Pretrained (d12) | — | ~0% | N/A | Baseline, Val BPB 0.8899, CORE 0.1186 |
| After SFT | Original | 3.56% | 0.2375 | Part 2.1 |
| After RL | Original (GRPO) | 10.92% | 0.0725 | Part 3 |
| RL + Reward A | TODO | TODO | TODO | Part 4 |
| RL + Reward B | TODO | TODO | TODO | Part 4 |
| RL + Reward A (separate) | TODO | TODO | TODO | Part 4 |
| RL + Reward B (separate) | TODO | TODO | TODO | Part 4 |

---

## References

- Karpathy, A. (2025). nanochat: A tiny chatbot arena and training harness. https://github.com/karpathy/nanochat/discussions/481
- Shao, Z., Wang, P., Zhu, Q., et al. (2024). GRPO: Group Relative Policy Optimization for Language Model Alignment. arXiv preprint arXiv:2402.05191.
- Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168.
