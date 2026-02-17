"""Corpora loading for multilingual evaluation."""

from __future__ import annotations

from datasets import load_dataset

# Wikipedia language codes
WIKI_CODES: dict[str, str] = {
    "en": "20231101.en",
    "zh": "20231101.zh",
    "ja": "20231101.ja",
    "ar": "20231101.ar",
    "hi": "20231101.hi",
    "de": "20231101.de",
    "tr": "20231101.tr",
    "ko": "20231101.ko",
    "th": "20231101.th",
    "ru": "20231101.ru",
    "fr": "20231101.fr",
    "es": "20231101.es",
    "pt": "20231101.pt",
    "vi": "20231101.vi",
    "id": "20231101.id",
}

# Number of articles to sample per language
DEFAULT_N_ARTICLES = 100

_cache: dict[str, list[str]] = {}


def get_texts(
    lang: str,
    n_articles: int = DEFAULT_N_ARTICLES,
    max_chars: int = 50000,
) -> list[str]:
    """Load texts for a language from Wikipedia.

    Args:
        lang: Language code (e.g., "en", "zh").
        n_articles: Number of articles to load.
        max_chars: Approximate total character budget.

    Returns:
        List of text chunks.
    """
    cache_key = f"wiki_{lang}_{n_articles}"
    if cache_key in _cache:
        return _cache[cache_key]

    wiki_code = WIKI_CODES.get(lang)
    if wiki_code is None:
        raise ValueError(f"Unsupported language: {lang}. Available: {available_languages()}")

    ds = load_dataset("wikimedia/wikipedia", wiki_code, split="train", streaming=True)

    texts = []
    total_chars = 0
    for i, row in enumerate(ds):
        if i >= n_articles or total_chars >= max_chars:
            break
        text = row["text"].strip()
        if len(text) > 100:  # skip very short articles
            texts.append(text)
            total_chars += len(text)

    _cache[cache_key] = texts
    return texts


def get_parallel_texts(
    langs: list[str],
    n_articles: int = DEFAULT_N_ARTICLES,
    max_chars: int = 50000,
) -> dict[str, list[str]]:
    """Load texts for multiple languages.

    Note: Wikipedia texts are NOT parallel (not translations of each other).
    Parity is computed by comparing total tokenized lengths.

    Args:
        langs: List of language codes.
        n_articles: Number of articles per language.
        max_chars: Character budget per language.

    Returns:
        Dict mapping language code to list of texts.
    """
    return {
        lang: get_texts(lang, n_articles=n_articles, max_chars=max_chars)
        for lang in langs
    }


def available_languages() -> list[str]:
    """Return list of supported short language codes."""
    return sorted(WIKI_CODES.keys())
