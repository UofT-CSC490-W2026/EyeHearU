# Sentence Quality Benchmark (`rule` vs `t5` vs `bedrock`)

This benchmark evaluates how well each gloss-to-English mode converts the **same gloss sequence** into natural English.

## What to compare

- `rule`: beamsearch + LM + join/polish
- `t5`: beamsearch + FLAN-T5 rewrite
- `bedrock`: beamsearch + Bedrock (Claude Haiku) rewrite

## Metrics (automatic)

For each system against your reference sentence:

- `token_precision`, `token_recall`, `token_f1` (class-style precision/recall/F1 on words)
- `sentence_bleu` (1-4 gram BLEU with brevity penalty)
- `rougeL_recall` (sequence overlap recall)
- `exact_match` (strict normalized equality)
- `capitalization_ok`, `punctuation_ok` (surface quality)

## 1) Prepare data

Create CSV like `data_template.csv` with:

- `case_id`
- `glosses` (use `|` separator, e.g. `nurse|yes|table`)
- `reference` (human target English sentence)

Use 30-100 cases for a meaningful comparison.

## 2) Generate model outputs

Run from repo root:

```bash
cd backend
python ..\benchmark\sentence_quality\evaluate.py generate ^
  --input ..\benchmark\sentence_quality\data_template.csv ^
  --output ..\benchmark\sentence_quality\results\predictions.csv
```

This calls:

- `app.services.gloss_to_english.gloss_sequence_to_english`
- `app.services.gloss_to_english_t5.gloss_sequence_to_english_t5`
- `app.services.gloss_to_english_bedrock.gloss_sequence_to_english_bedrock`

Bedrock uses your backend `.env` settings (`BEDROCK_REGION`, `BEDROCK_MODEL_ID`).

## 3) Score outputs

```bash
cd backend
python ..\benchmark\sentence_quality\evaluate.py score ^
  --predictions ..\benchmark\sentence_quality\results\predictions.csv ^
  --out ..\benchmark\sentence_quality\results\metrics.json
```

## 4) Suggested reporting categories

Use three sections in your report:

1. **Faithfulness**: token recall/F1, ROUGE-L (did it keep meaning)
2. **Fluency**: punctuation/capitalization + manual 1-5 rating
3. **Overall quality**: sentence BLEU + side-by-side examples

Add 5-10 qualitative examples where systems differ the most.

## Beam vs No-Beam ablation

To test your intended question ("does beam search over top-k candidate matrix help?"),
use `evaluate_ablation.py`.

### Input format

Use JSON like `data_ablation_template.json`:

- `case_id`
- `reference`
- `candidates`: list of clips; each clip is list of top-k candidates with
  `{"sign": "...", "confidence": ...}`.

This directly matches your matrix intuition:

- rows = clip positions
- columns = top-k candidates for that clip

### Generate ablation predictions

```bash
cd /mnt/c/26winter/csc490/EyeHearU/backend
PYTHONPATH=. python ../benchmark/sentence_quality/evaluate_ablation.py generate \
  --input ../benchmark/sentence_quality/data_ablation_template.json \
  --output ../benchmark/sentence_quality/results/ablation_predictions.json \
  --beam-size 8 \
  --lm-weight 1.0
```

### Score ablation

```bash
cd /mnt/c/26winter/csc490/EyeHearU/backend
PYTHONPATH=. python ../benchmark/sentence_quality/evaluate_ablation.py score \
  --predictions ../benchmark/sentence_quality/results/ablation_predictions.json \
  --out ../benchmark/sentence_quality/results/ablation_metrics.json
```

### Compared systems

- `greedy_rule`, `greedy_t5`, `greedy_bedrock`
- `beam_rule`, `beam_t5`, `beam_bedrock`

So you can isolate:

1. **decoding effect**: greedy vs beam (same rewriter)
2. **rewriter effect**: rule vs t5 vs bedrock (same decoding)



Our result: (gloss to natural sentence)- across different last model

rule     F1=0.819 BLEU=0.060 ROUGE-L=0.683 Cap=1.000 Punct=1.000

t5       F1=0.732 BLEU=0.093 ROUGE-L=0.696 Cap=1.000 Punct=1.000

bedrock  F1=0.839 BLEU=0.248 ROUGE-L=0.786 Cap=1.000 Punct=1.000



