# Checkpoint Comparison: 512-Context vs 2048-Context Picochat

## Setup

| Checkpoint | Description | Step | Context length | Training |
|------------|-------------|------|----------------|----------|
| **Checkpoint 1** | `base_model_002000` | 2,000 | 512 | Trained from scratch on a portion of FineWeb-EDU at 512 tokens |
| **Checkpoint 2** | `base_model_004000` | 4,000 | 2048 | Resumed from Checkpoint 1; continued training for 2,000 steps at 2048 tokens |

Both models are picochat (depth 6, ~11M params). Evaluation uses the CORE benchmark (DCLM-style) with 100 examples per task and bits-per-byte (BPB) on the validation split.

---

## Aggregate Results

| Metric | Checkpoint 1 (step 2k, ctx 512) | Checkpoint 2 (step 4k, ctx 2048) |
|--------|----------------------------------|-----------------------------------|
| **CORE (centered)** | **0.0453** | 0.0386 |
| Val BPB (from training logs) | 1.264 | 1.183 |

- **CORE**: Checkpoint 1 is slightly higher (0.0453 vs 0.0386). The 2048-continuation did not improve the aggregate CORE score on this benchmark.
- **Val BPB**: Checkpoint 2 has lower validation BPB (1.183 vs 1.264), indicating better next-token prediction on the pretraining distribution after the 2048 phase.

---

## Task-Level Comparison (Accuracy)

Tasks where **Checkpoint 2 (2048)** does better:

| Task | Ckpt 1 (512) | Ckpt 2 (2048) | Δ |
|------|--------------|---------------|-----|
| hellaswag_zeroshot | 0.26 | 0.27 | +0.01 |
| copa | 0.50 | 0.51 | +0.01 |
| commonsense_qa | 0.29 | 0.33 | +0.04 |
| piqa | 0.45 | 0.47 | +0.02 |
| hellaswag | 0.27 | 0.28 | +0.01 |
| winograd | 0.45 | 0.49 | +0.04 |
| winogrande | 0.45 | 0.49 | +0.04 |
| bigbench_dyck_languages | 0.00 | 0.02 | +0.02 |
| bigbench_cs_algorithms | 0.13 | **0.41** | **+0.28** |
| bigbench_operators | 0.12 | 0.15 | +0.03 |
| coqa | 0.00 | 0.01 | +0.01 |

Tasks where **Checkpoint 1 (512)** does better:

| Task | Ckpt 1 (512) | Ckpt 2 (2048) | Δ |
|------|--------------|---------------|-----|
| openbook_qa | 0.25 | 0.24 | −0.01 |
| lambada_openai | 0.20 | 0.18 | −0.02 |
| agi_eval_lsat_ar | 0.28 | 0.19 | −0.09 |
| boolq | **0.75** | 0.52 | **−0.23** |
| bigbench_language_identification | 0.26 | 0.23 | −0.03 |

No change: arc_easy (0.34), arc_challenge (0.21), and several tasks at 0.00 for both.

---

## Summary and Interpretation

1. **CORE vs BPB**  
   Checkpoint 2 has better validation BPB but slightly lower CORE. So extending context and continuing training helped language modeling (BPB) more than it helped this mix of in-context evaluation tasks on average.

2. **Where 2048 helps**  
   The 2048 checkpoint improves on several tasks that can benefit from longer context (e.g. commonsense_qa, winograd, winogrande) and shows a large gain on **bigbench_cs_algorithms** (+0.28). That suggests the longer-context phase helped on some reasoning and multi-turn style tasks.

3. **Where 512 wins**  
   The 512 checkpoint is much stronger on **boolq** (0.75 vs 0.52) and better on **agi_eval_lsat_ar** and **lambada_openai**. Possible factors: different effective capacity allocation after continuation, or CORE prompt length / truncation (e.g. 512 ckpt sees truncated prompts, which might sometimes help on these formats).

4. **Conclusion**  
   Continuing from 512 to 2048 context improved validation BPB and several CORE tasks (especially bigbench_cs_algorithms, winograd, winogrande, commonsense_qa), but hurt a few others (notably boolq), so the overall CORE score was slightly lower. For a write-up, you can summarize: the 2048 continuation improves language modeling and some longer-context tasks, while the 512 checkpoint remains stronger on a few specific benchmarks (boolq, agi_eval_lsat_ar).

---

*Results from `base_eval/base_model_002000.csv` and `base_eval/base_model_004000.csv`.*
