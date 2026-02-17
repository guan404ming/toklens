"""Tests for metrics module."""

import pytest

from tests.conftest import TEST_TEXTS
from toklens import metrics


def test_fertility_positive(tiny_tokenizer):
    result = metrics.compute_fertility(tiny_tokenizer, TEST_TEXTS["en"])
    assert result > 0


def test_fertility_empty(tiny_tokenizer):
    result = metrics.compute_fertility(tiny_tokenizer, "")
    assert result == 0.0


def test_cpt_positive(tiny_tokenizer):
    result = metrics.compute_cpt(tiny_tokenizer, TEST_TEXTS["en"])
    assert result > 0


def test_cpt_empty(tiny_tokenizer):
    result = metrics.compute_cpt(tiny_tokenizer, "")
    assert result == 0.0


def test_compression_ratio_positive(tiny_tokenizer):
    result = metrics.compute_compression_ratio(tiny_tokenizer, TEST_TEXTS["en"])
    assert result > 0


def test_strr_range(tiny_tokenizer):
    result = metrics.compute_strr(tiny_tokenizer, TEST_TEXTS["en"])
    assert 0.0 <= result <= 1.0


def test_strr_empty(tiny_tokenizer):
    result = metrics.compute_strr(tiny_tokenizer, "")
    assert result == 0.0


def test_nsl_positive(tiny_tokenizer):
    result = metrics.compute_nsl(tiny_tokenizer, TEST_TEXTS["en"])
    assert result > 0


def test_nsl_with_ref(tiny_tokenizer):
    result = metrics.compute_nsl(tiny_tokenizer, TEST_TEXTS["en"], ref_length=100)
    assert result > 0


def test_parity_self_is_one(tiny_tokenizer):
    text = TEST_TEXTS["en"]
    result = metrics.compute_parity(tiny_tokenizer, text, text)
    assert result == pytest.approx(1.0)


def test_parity_different_lengths(tiny_tokenizer):
    short = "the cat sat"
    long = "the cat sat on the mat in the park"
    result = metrics.compute_parity(tiny_tokenizer, long, short)
    assert result > 1.0  # longer text should have higher parity ratio


def test_vocab_overlap_same(tiny_tokenizer):
    result = metrics.compute_vocab_overlap(tiny_tokenizer, tiny_tokenizer)
    assert result["overlap"] == result["total_a"]
    assert result["only_a"] == 0
    assert result["only_b"] == 0


def test_compute_all_keys(tiny_tokenizer):
    result = metrics.compute_all(tiny_tokenizer, TEST_TEXTS["en"])
    assert "fertility" in result
    assert "cpt" in result
    assert "compression_ratio" in result
    assert "strr" in result
    assert "nsl" in result
    assert "parity" not in result  # no ref_text


def test_compute_all_with_ref(tiny_tokenizer):
    result = metrics.compute_all(tiny_tokenizer, TEST_TEXTS["zh"], ref_text=TEST_TEXTS["en"])
    assert "parity" in result
