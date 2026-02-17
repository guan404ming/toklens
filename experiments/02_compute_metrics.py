"""Step 2: Compute TokLens metrics on all tokenizers across all languages.

Input: None (downloads tokenizers from HF, Wikipedia from HF datasets)
Output: experiments/toklens_results.json, experiments/toklens_results.csv
"""

from __future__ import annotations

import csv
import json
import traceback

from tokenizers import Tokenizer

from experiments.config import EXTRA_TOKENIZERS, LANGS, TOKENIZERS
from toklens import metrics
from toklens.corpora import get_parallel_texts


class _EncodingShim:
    """Wraps a list of token IDs to match tokenizers.Encoding interface."""

    def __init__(self, ids: list[int]):
        self.ids = ids


class _SlowTokenizerShim:
    """Wraps a transformers slow tokenizer to match tokenizers.Tokenizer interface."""

    def __init__(self, auto_tok):
        self._tok = auto_tok

    def encode(self, text: str) -> _EncodingShim:
        # add_special_tokens=False to match tokenizers lib behavior
        return _EncodingShim(self._tok.encode(text, add_special_tokens=False))

    def get_vocab_size(self) -> int:
        return self._tok.vocab_size

    def get_vocab(self) -> dict:
        return self._tok.get_vocab()


def _load_tokenizer(hf_name: str, tokenizer_source: str | None) -> Tokenizer:
    """Load tokenizer, trying tokenizers lib first, then AutoTokenizer fallback."""
    source = tokenizer_source or hf_name
    try:
        return Tokenizer.from_pretrained(source)
    except Exception:
        pass

    # Fallback: use transformers AutoTokenizer and convert
    print(f"  Falling back to AutoTokenizer for {source}...")
    from transformers import AutoTokenizer
    auto_tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    # Fast tokenizers wrap a Rust backend
    if hasattr(auto_tok, "backend_tokenizer"):
        return auto_tok.backend_tokenizer
    # Slow tokenizers (e.g. tiktoken-based ChatGLM) need a shim
    print(f"  Using slow tokenizer shim for {source}")
    return _SlowTokenizerShim(auto_tok)


def compute_all_metrics(
    output_json: str = "experiments/toklens_results.json",
    output_csv: str = "experiments/toklens_results.csv",
) -> dict:
    """Compute all TokLens metrics for every tokenizer x language pair."""
    print(f"Loading Wikipedia texts for {len(LANGS)} languages...")
    parallel = get_parallel_texts(LANGS)
    ref_texts = parallel["en"]
    ref_joined = " ".join(ref_texts)

    results = {}
    for hf_name, display_name, _params, tokenizer_source in TOKENIZERS:
        print(f"\nEvaluating: {display_name} ({hf_name})")
        try:
            tokenizer = _load_tokenizer(hf_name, tokenizer_source)
        except Exception:
            print("  FAILED to load tokenizer:")
            traceback.print_exc()
            results[hf_name] = {"error": "tokenizer_load_failed"}
            continue

        lang_results = {}
        for lang in LANGS:
            texts = parallel[lang]
            joined = " ".join(texts)
            m = metrics.compute_all(tokenizer, joined, ref_text=ref_joined)
            lang_results[lang] = m

        # Compute vocab size
        vocab_size = tokenizer.get_vocab_size()

        results[hf_name] = {
            "display_name": display_name,
            "vocab_size": vocab_size,
            "metrics": lang_results,
        }
        # Print summary for en
        en = lang_results.get("en", {})
        print(f"  vocab={vocab_size}, en fertility={en.get('fertility', 'N/A'):.3f}")

    # Extra tokenizers (no leaderboard scores, metric-only analysis)
    for tok_source, display_name, _desc in EXTRA_TOKENIZERS:
        print(f"\nEvaluating (extra): {display_name} ({tok_source})")
        try:
            tokenizer = _load_tokenizer(tok_source, None)
        except Exception:
            print("  FAILED to load tokenizer:")
            traceback.print_exc()
            results[tok_source] = {"error": "tokenizer_load_failed"}
            continue

        lang_results = {}
        for lang in LANGS:
            texts = parallel[lang]
            joined = " ".join(texts)
            m = metrics.compute_all(tokenizer, joined, ref_text=ref_joined)
            lang_results[lang] = m

        vocab_size = tokenizer.get_vocab_size()
        results[tok_source] = {
            "display_name": display_name,
            "vocab_size": vocab_size,
            "metrics": lang_results,
            "extra": True,
        }
        en = lang_results.get("en", {})
        print(f"  vocab={vocab_size}, en fertility={en.get('fertility', 'N/A'):.3f}")

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to {output_json}")

    # Save flat CSV
    rows = []
    for hf_name, data in results.items():
        if "error" in data:
            continue
        for lang, m in data["metrics"].items():
            row = {
                "tokenizer": hf_name,
                "display_name": data["display_name"],
                "vocab_size": data["vocab_size"],
                "lang": lang,
            }
            row.update(m)
            rows.append(row)

    if rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to {output_csv}")

    return results


if __name__ == "__main__":
    compute_all_metrics()
