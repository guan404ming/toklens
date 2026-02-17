"""Metric validation experiment.

Runs TokLens metrics on multiple tokenizers and collects results
for correlation analysis with downstream benchmarks.
"""

from __future__ import annotations

import csv
import json
import sys

from toklens.analyzer import Analyzer

# Tokenizers to evaluate (HuggingFace model names)
TOKENIZERS = [
    "openai-community/gpt2",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-7B",
    "google/gemma-2-9b",
    "mistralai/Mistral-7B-v0.3",
    "deepseek-ai/DeepSeek-V2-Lite",
    "microsoft/Phi-3-mini-4k-instruct",
    "tiiuae/falcon-7b",
]

# Languages to test
LANGS = ["en", "zh", "ja", "ar", "hi", "de", "tr", "ko", "th", "ru", "fr", "es"]


def run_experiment(
    tokenizers: list[str] | None = None,
    langs: list[str] | None = None,
    output_path: str = "experiments/results.json",
) -> dict:
    """Run metrics on all tokenizers and save results.

    Args:
        tokenizers: List of HF model names. Defaults to TOKENIZERS.
        langs: Language codes. Defaults to LANGS.
        output_path: Where to save JSON results.

    Returns:
        Dict with all results.
    """
    tokenizers = tokenizers or TOKENIZERS
    langs = langs or LANGS

    all_results = {}
    for name in tokenizers:
        print(f"Evaluating: {name}")
        try:
            analyzer = Analyzer.from_pretrained(name)
            report = analyzer.evaluate(langs=langs)
            all_results[name] = report.to_dict()
            print(f"  Done: {len(report.langs)} languages")
        except Exception as e:
            print(f"  Failed: {e}")
            all_results[name] = {"error": str(e)}

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


def results_to_csv(
    results_path: str = "experiments/results.json",
    output_path: str = "experiments/results.csv",
) -> None:
    """Convert JSON results to a flat CSV for analysis."""
    with open(results_path) as f:
        data = json.load(f)

    rows = []
    for tokenizer_name, result in data.items():
        if "error" in result:
            continue
        for lang, metrics_dict in result.get("results", {}).items():
            row = {"tokenizer": tokenizer_name, "lang": lang}
            row.update(metrics_dict)
            rows.append(row)

    if not rows:
        print("No valid results to convert")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "csv":
        results_to_csv()
    else:
        run_experiment()
