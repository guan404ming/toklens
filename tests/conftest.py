"""Shared test fixtures."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


@pytest.fixture
def tiny_tokenizer() -> Tokenizer:
    """Create a tiny BPE tokenizer for testing (no network needed)."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=256,
        special_tokens=["[UNK]", "[PAD]"],
    )
    # Train on minimal corpus
    tokenizer.train_from_iterator(
        [
            "the cat sat on the mat",
            "the dog ran in the park",
            "hello world this is a test",
            "testing one two three four five",
        ],
        trainer=trainer,
    )
    return tokenizer


# Inline test texts (no network needed)
TEST_TEXTS = {
    "en": "The quick brown fox jumps over the lazy dog. This is a simple test sentence.",
    "zh": "快速的棕色狐狸跳过了懒惰的狗。这是一个简单的测试句子。",
    "ja": "素早い茶色の狐が怠惰な犬を飛び越えた。これは簡単なテスト文です。",
    "de": "Der schnelle braune Fuchs springt ueber den faulen Hund. Ein einfacher Testsatz.",
}
