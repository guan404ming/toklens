"""Tests for analyzer module."""

from tests.conftest import TEST_TEXTS
from toklens.analyzer import Analyzer


def test_analyzer_init(tiny_tokenizer):
    analyzer = Analyzer(tiny_tokenizer, name="test")
    assert analyzer.name == "test"


def test_evaluate_text(tiny_tokenizer):
    analyzer = Analyzer(tiny_tokenizer, name="test")
    result = analyzer.evaluate_text(TEST_TEXTS["en"])
    assert "fertility" in result
    assert "cpt" in result
    assert "strr" in result


def test_evaluate_text_with_ref(tiny_tokenizer):
    analyzer = Analyzer(tiny_tokenizer, name="test")
    result = analyzer.evaluate_text(TEST_TEXTS["zh"], ref_text=TEST_TEXTS["en"])
    assert "parity" in result
