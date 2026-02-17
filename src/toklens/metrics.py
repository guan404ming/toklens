"""Tokenizer quality metrics."""

from __future__ import annotations

import numpy as np
from tokenizers import Tokenizer


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode text without special tokens for fair comparison.

    Some tokenizers (Llama, Gemma, Mistral, etc.) add BOS/EOS tokens by default.
    We strip them to measure raw tokenization quality.
    """
    try:
        return tokenizer.encode(text, add_special_tokens=False).ids
    except TypeError:
        # Shim or tokenizer that doesn't accept add_special_tokens
        return tokenizer.encode(text).ids


def fertility(tokens_per_word: list[list[int]], words: list[list[str]]) -> dict[str, float]:
    """Compute fertility (average tokens per word) per language.

    Args:
        tokens_per_word: For each language, list of token counts per word.
        words: For each language, list of words.

    Returns:
        Dict mapping language code to fertility score.
    """
    results = {}
    for lang_tokens, lang_words in zip(tokens_per_word, words):
        if lang_words:
            results[lang_words[0]] = np.mean(lang_tokens).item()
    return results


def _tokenize_words(
    tokenizer: Tokenizer, text: str
) -> tuple[list[str], list[int]]:
    """Split text into words and count tokens per word."""
    words = text.split()
    token_counts = []
    for word in words:
        ids = _encode(tokenizer, word)
        token_counts.append(len(ids))
    return words, token_counts


def compute_fertility(tokenizer: Tokenizer, text: str) -> float:
    """Compute fertility for a single text.

    Fertility = average number of tokens per word. Lower is better.
    """
    words, token_counts = _tokenize_words(tokenizer, text)
    if not words:
        return 0.0
    return float(np.mean(token_counts))


def compute_cpt(tokenizer: Tokenizer, text: str) -> float:
    """Compute characters per token.

    CPT = total characters / total tokens. Higher means coarser granularity.
    """
    ids = _encode(tokenizer, text)
    if not ids:
        return 0.0
    return len(text) / len(ids)


def compute_compression_ratio(tokenizer: Tokenizer, text: str) -> float:
    """Compute compression ratio.

    Compression ratio = bytes / tokens. Higher means better compression.
    """
    ids = _encode(tokenizer, text)
    if not ids:
        return 0.0
    n_bytes = len(text.encode("utf-8"))
    return n_bytes / len(ids)


def compute_strr(tokenizer: Tokenizer, text: str) -> float:
    """Compute Single Token Retention Rate.

    STRR = proportion of words that are kept as a single token.
    """
    words, token_counts = _tokenize_words(tokenizer, text)
    if not words:
        return 0.0
    single_token_words = sum(1 for c in token_counts if c == 1)
    return single_token_words / len(words)


def compute_nsl(tokenizer: Tokenizer, text: str, ref_length: int | None = None) -> float:
    """Compute Normalized Sequence Length.

    NSL = tokenized length / reference length.
    If ref_length is not provided, uses character count as reference.
    """
    ids = _encode(tokenizer, text)
    if ref_length is None:
        ref_length = len(text)
    if ref_length == 0:
        return 0.0
    return len(ids) / ref_length


def compute_parity(
    tokenizer: Tokenizer,
    text: str,
    ref_text: str,
) -> float:
    """Compute parity ratio between a target text and reference text.

    Parity = len(tokenize(text)) / len(tokenize(ref_text)).
    A value close to 1.0 means the tokenizer treats both languages similarly.
    """
    ids = _encode(tokenizer, text)
    ref_ids = _encode(tokenizer, ref_text)
    if not ref_ids:
        return 0.0
    return len(ids) / len(ref_ids)


def compute_vocab_overlap(tokenizer_a: Tokenizer, tokenizer_b: Tokenizer) -> dict[str, int]:
    """Compute vocabulary overlap between two tokenizers.

    Returns:
        Dict with keys: overlap, only_a, only_b, total_a, total_b.
    """
    vocab_a = set(tokenizer_a.get_vocab().keys())
    vocab_b = set(tokenizer_b.get_vocab().keys())
    overlap = vocab_a & vocab_b
    return {
        "overlap": len(overlap),
        "only_a": len(vocab_a - vocab_b),
        "only_b": len(vocab_b - vocab_a),
        "total_a": len(vocab_a),
        "total_b": len(vocab_b),
    }


def compute_all(
    tokenizer: Tokenizer, text: str, ref_text: str | None = None
) -> dict[str, float]:
    """Compute all single-tokenizer metrics on a text.

    Args:
        tokenizer: The tokenizer to evaluate.
        text: The text to evaluate on.
        ref_text: Reference text for parity computation (optional).

    Returns:
        Dict of metric name to value.
    """
    results: dict[str, float] = {
        "fertility": compute_fertility(tokenizer, text),
        "cpt": compute_cpt(tokenizer, text),
        "compression_ratio": compute_compression_ratio(tokenizer, text),
        "strr": compute_strr(tokenizer, text),
        "nsl": compute_nsl(tokenizer, text),
    }
    if ref_text is not None:
        results["parity"] = compute_parity(tokenizer, text, ref_text)
    return results
