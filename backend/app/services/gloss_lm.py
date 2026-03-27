"""
Gloss bigram language model with add-one (Laplace) smoothing.

Scores transitions P(g_j | g_{i-1}) with a special start token "<s>" for the
first gloss in a sequence. Used with beam search to prefer plausible multi-sign
sequences over independent per-clip argmax.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

START_TOKEN = "<s>"


class GlossBigramLM:
    """Log-probability estimates from (sparse) bigram counts + smoothing."""

    __slots__ = (
        "_uniform",
        "_vocab_size_uniform",
        "_bigram",
        "_uni",
        "_alpha",
        "_vocab_size",
    )

    def __init__(
        self,
        bigram_counts: dict[tuple[str, str], int] | None = None,
        unigram_counts: dict[str, int] | None = None,
        *,
        alpha: float = 1.0,
        uniform: bool = False,
        vocab_size_uniform: int = 2,
    ):
        self._uniform = bool(uniform)
        self._vocab_size_uniform = max(int(vocab_size_uniform), 1)
        self._alpha = float(alpha)
        if self._uniform:
            self._bigram = {}
            self._uni = {}
            self._vocab_size = self._vocab_size_uniform
            return

        self._bigram = dict(bigram_counts or {})
        keys: set[str] = set()
        for (a, b) in self._bigram:
            keys.add(a)
            keys.add(b)
        if unigram_counts:
            keys |= set(unigram_counts.keys())
        self._vocab_size = max(len(keys), 1)

        self._uni = {g: 0 for g in keys}
        if unigram_counts:
            for g, c in unigram_counts.items():
                self._uni[g] = self._uni.get(g, 0) + int(c)
        for (prev, _), c in self._bigram.items():
            self._uni[prev] = self._uni.get(prev, 0) + int(c)

    def log_p(self, prev_gloss: str, next_gloss: str) -> float:
        """Log P(next | prev) with Laplace smoothing over the observed vocab."""
        if self._uniform:
            return -math.log(self._vocab_size_uniform)

        v = float(self._vocab_size)
        a = self._alpha
        c_bg = self._bigram.get((prev_gloss, next_gloss), 0)
        denom = self._uni.get(prev_gloss, 0) + a * v
        if denom <= 0:
            return -math.log(v)
        num = c_bg + a
        return math.log(num / denom)

    @classmethod
    def uniform_over_vocab(cls, vocab: set[str]) -> GlossBigramLM:
        """O(1) storage — all transitions get the same log-prob (beam uses model scores)."""
        vs = len(vocab | {START_TOKEN})
        return cls(uniform=True, vocab_size_uniform=max(vs, 1))

    @classmethod
    def from_json_file(cls, path: Path, *, vocab_hint: set[str] | None = None) -> GlossBigramLM:
        """
        Load LM from JSON:
          { "bigram_counts": { "prev|||next": int, ... }, "unigram_counts": { "g": int, ... } }
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        bg_raw: dict[str, int] = raw.get("bigram_counts") or {}
        uni_raw: dict[str, int] = raw.get("unigram_counts") or {}
        bigram: dict[tuple[str, str], int] = {}
        for k, c in bg_raw.items():
            if "|||" in k:
                a, b = k.split("|||", 1)
                bigram[(a, b)] = int(c)
        uni: dict[str, int] = {str(g): int(c) for g, c in uni_raw.items()}
        if vocab_hint:
            for g in vocab_hint:
                uni.setdefault(g, 0)
        return cls(bigram, uni if uni else None, alpha=float(raw.get("alpha", 1.0)))


def load_gloss_lm(path: Path | None, index_to_gloss: dict[int, str]) -> GlossBigramLM:
    """Load bigram LM from disk, or fall back to uniform over label-map glosses."""
    vocab = {g for g in index_to_gloss.values() if g}
    if path and path.is_file():
        try:
            return GlossBigramLM.from_json_file(path, vocab_hint=vocab)
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return GlossBigramLM.uniform_over_vocab(vocab)
