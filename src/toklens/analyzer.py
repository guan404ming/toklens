"""Main Analyzer class for tokenizer evaluation."""

from __future__ import annotations

from tokenizers import Tokenizer

from toklens import corpora, metrics
from toklens.report import Report


class Analyzer:
    """Evaluate a tokenizer across languages and metrics."""

    def __init__(self, tokenizer: Tokenizer, name: str = "unknown"):
        self.tokenizer = tokenizer
        self.name = name

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> Analyzer:
        """Load a tokenizer from HuggingFace Hub or local path.

        Args:
            name_or_path: HuggingFace model name or path to tokenizer file.
        """
        tokenizer = Tokenizer.from_pretrained(name_or_path)
        return cls(tokenizer, name=name_or_path)

    def evaluate(
        self,
        langs: list[str] | None = None,
        ref_lang: str = "en",
        split: str = "devtest",
    ) -> Report:
        """Evaluate tokenizer on FLORES-200 corpora.

        Args:
            langs: Language codes to evaluate. Defaults to all available.
            ref_lang: Reference language for parity computation.
            split: FLORES-200 split to use.

        Returns:
            A Report with per-language, per-metric results.
        """
        if langs is None:
            langs = corpora.available_languages()

        # Ensure ref_lang is included
        if ref_lang not in langs:
            langs = [ref_lang, *langs]

        parallel = corpora.get_parallel_texts(langs, split=split)
        ref_texts = parallel[ref_lang]

        results: dict[str, dict[str, float]] = {}
        for lang in langs:
            texts = parallel[lang]
            joined = " ".join(texts)
            ref_joined = " ".join(ref_texts)
            results[lang] = metrics.compute_all(
                self.tokenizer, joined, ref_text=ref_joined
            )

        return Report(self.name, results)

    def evaluate_text(self, text: str, ref_text: str | None = None) -> dict[str, float]:
        """Evaluate tokenizer on user-provided text.

        Args:
            text: Text to evaluate.
            ref_text: Reference text for parity (optional).

        Returns:
            Dict of metric name to value.
        """
        return metrics.compute_all(self.tokenizer, text, ref_text=ref_text)
