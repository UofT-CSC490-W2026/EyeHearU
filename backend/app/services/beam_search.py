"""
Beam search over per-clip top-k gloss hypotheses with a gloss bigram LM.

Score(sequence) = sum_t log P_model(g_t | clip_t) + lm_weight * sum_t log P_LM(g_t | g_{t-1})
with g_{-1} = START_TOKEN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.services.gloss_lm import START_TOKEN, GlossBigramLM

LOG_EPS = 1e-12


def _model_logp(confidence: float) -> float:
    c = max(float(confidence), LOG_EPS)
    return math.log(min(c, 1.0))


@dataclass(frozen=True)
class BeamPath:
    score: float
    glosses: tuple[str, ...]


def beam_search(
    candidates_per_clip: list[list[dict]],
    lm: GlossBigramLM,
    *,
    beam_size: int = 8,
    lm_weight: float = 1.0,
    top_sequences: int = 5,
) -> list[BeamPath]:
    """
    Args:
        candidates_per_clip: For each clip, a list of dicts with keys "sign", "confidence"
        lm: Gloss bigram language model
        beam_size: Max partial hypotheses to keep after each clip
        lm_weight: Strength of LM vs model (typical 0.5–2.0)
        top_sequences: How many full sequences to return (sorted by score desc)

    Returns:
        Best full sequences with total log scores (higher is better).
    """
    if not candidates_per_clip:
        return []

    # Step 0: start from START_TOKEN
    beam: list[tuple[float, tuple[str, ...], str]] = []
    # (total_score, path_glosses, last_gloss_for_lm)
    for cand in candidates_per_clip[0]:
        g = cand["sign"]
        mlp = _model_logp(float(cand["confidence"]))
        lmp = lm.log_p(START_TOKEN, g)
        score = mlp + lm_weight * lmp
        beam.append((score, (g,), g))

    beam.sort(key=lambda x: -x[0])
    beam = beam[: max(beam_size, 1)]

    for step in range(1, len(candidates_per_clip)):
        next_beam: list[tuple[float, tuple[str, ...], str]] = []
        for prev_score, path, prev_g in beam:
            for cand in candidates_per_clip[step]:
                g = cand["sign"]
                mlp = _model_logp(float(cand["confidence"]))
                lmp = lm.log_p(prev_g, g)
                score = prev_score + mlp + lm_weight * lmp
                next_beam.append((score, path + (g,), g))
        next_beam.sort(key=lambda x: -x[0])
        beam = next_beam[: max(beam_size, 1)]

    final = [BeamPath(score=s, glosses=p) for s, p, _ in beam]
    # Dedupe paths, keep best score per unique gloss tuple
    best: dict[tuple[str, ...], float] = {}
    for bp in final:
        best[bp.glosses] = max(best.get(bp.glosses, float("-inf")), bp.score)
    out = [BeamPath(score=v, glosses=k) for k, v in best.items()]
    out.sort(key=lambda x: -x.score)
    return out[: max(top_sequences, 1)]
