"""Corpora loading for multilingual evaluation."""

from __future__ import annotations

from datasets import load_dataset

# FLORES-200 language codes for common languages
LANG_CODES: dict[str, str] = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "de": "deu_Latn",
    "tr": "tur_Latn",
    "ko": "kor_Hang",
    "th": "tha_Thai",
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "vi": "vie_Latn",
    "id": "ind_Latn",
}

_cache: dict[str, list[str]] = {}


def get_flores_code(lang: str) -> str:
    """Convert a short language code to FLORES-200 code."""
    if lang in LANG_CODES:
        return LANG_CODES[lang]
    # Assume it's already a FLORES-200 code
    return lang


def get_texts(lang: str, split: str = "devtest") -> list[str]:
    """Load texts for a language from FLORES-200.

    Args:
        lang: Language code (e.g., "en", "zh") or FLORES-200 code.
        split: Dataset split. Default "devtest" (1012 sentences).

    Returns:
        List of sentences.
    """
    flores_code = get_flores_code(lang)
    cache_key = f"{flores_code}_{split}"

    if cache_key in _cache:
        return _cache[cache_key]

    ds = load_dataset(
        "openlanguagedata/flores_plus",
        name=flores_code,
        split=split,
        trust_remote_code=True,
    )
    texts = [row["text"] for row in ds]
    _cache[cache_key] = texts
    return texts


def get_parallel_texts(
    langs: list[str], split: str = "devtest"
) -> dict[str, list[str]]:
    """Load parallel texts for multiple languages.

    Sentences at the same index are translations of each other.

    Args:
        langs: List of language codes.
        split: Dataset split.

    Returns:
        Dict mapping language code to list of sentences.
    """
    return {lang: get_texts(lang, split) for lang in langs}


def available_languages() -> list[str]:
    """Return list of supported short language codes."""
    return sorted(LANG_CODES.keys())
