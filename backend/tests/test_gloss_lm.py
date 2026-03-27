"""Tests for gloss bigram LM."""

import json

from app.services.gloss_lm import GlossBigramLM, START_TOKEN, load_gloss_lm


def test_uniform_log_p_constant():
    lm = GlossBigramLM.uniform_over_vocab({"a", "b"})
    p1 = lm.log_p(START_TOKEN, "a")
    p2 = lm.log_p("a", "b")
    assert p1 == p2


def test_load_gloss_lm_fallback_corrupt_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    lm = load_gloss_lm(bad, {0: "x", 1: "y"})
    assert isinstance(lm, GlossBigramLM)


def test_from_json_skips_keys_without_separator(tmp_path):
    path = tmp_path / "lm.json"
    path.write_text(
        json.dumps(
            {
                "bigram_counts": {"badkey": 9, "a|||b": 3},
                "unigram_counts": {"a": 3, "b": 3},
            }
        ),
        encoding="utf-8",
    )
    lm = GlossBigramLM.from_json_file(path, vocab_hint=None)
    assert lm.log_p("a", "b") <= 0


def test_from_json_file(tmp_path):
    path = tmp_path / "lm.json"
    path.write_text(
        json.dumps(
            {
                "alpha": 1.0,
                "bigram_counts": {"<s>|||hello": 10, "hello|||thanks": 5},
                "unigram_counts": {"<s>": 10, "hello": 15, "thanks": 5},
            }
        ),
        encoding="utf-8",
    )
    lm = GlossBigramLM.from_json_file(path, vocab_hint={"extra"})
    assert lm.log_p(START_TOKEN, "hello") < 0


def test_log_p_denom_zero_returns_neg_log_v():
    """Empty LM with alpha=0 yields denom 0 for unseen prev — hits fallback branch."""
    lm = GlossBigramLM({}, {}, alpha=0.0)
    # vocab_size 1, uni empty, bigram empty — denom for any prev is 0 + 0 = 0
    assert lm.log_p("unknown_prev", "y") == -__import__("math").log(1.0)


def test_load_missing_file_uniform():
    lm = load_gloss_lm(None, {0: "a", 1: "b"})
    assert lm.log_p(START_TOKEN, "a") == lm.log_p(START_TOKEN, "b")
