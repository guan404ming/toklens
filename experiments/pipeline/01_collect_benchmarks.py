"""Step 1: Collect benchmark scores from Open LLM Leaderboard v2.

Downloads leaderboard data and extracts scores for our target models.

Input: None (downloads from HuggingFace)
Output: experiments/data/benchmark_scores.json
Source: https://huggingface.co/datasets/open-llm-leaderboard/contents
"""

from __future__ import annotations

import json

from datasets import load_dataset

from experiments.config import BENCHMARK_COLUMNS, TOKENIZERS


def collect_benchmarks(output_path: str = "experiments/data/benchmark_scores.json") -> dict:
    """Download leaderboard and extract scores for our tokenizers."""
    print("Loading Open LLM Leaderboard v2...")
    ds = load_dataset("open-llm-leaderboard/contents", split="train")

    # Build lookup by fullname
    leaderboard = {}
    for row in ds:
        leaderboard[row["fullname"]] = row

    results = {}
    for hf_name, display_name, params, _tok_src in TOKENIZERS:
        # Try exact match first
        row = leaderboard.get(hf_name)
        if row is None:
            print(f"  MISSING: {hf_name}")
            results[hf_name] = {
                "display_name": display_name,
                "params_b": params,
                "source": "not_found",
                "benchmarks": {},
            }
            continue

        benchmarks = {}
        for bench_name, col_name in BENCHMARK_COLUMNS.items():
            val = row.get(col_name)
            if val is not None:
                benchmarks[bench_name] = float(val)

        results[hf_name] = {
            "display_name": display_name,
            "params_b": params,
            "source": "https://huggingface.co/datasets/open-llm-leaderboard/contents",
            "benchmarks": benchmarks,
        }
        avg = benchmarks.get("Average", "N/A")
        print(f"  OK: {display_name} (avg={avg})")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    print(f"Found: {sum(1 for v in results.values() if v['benchmarks'])}/{len(TOKENIZERS)}")
    return results


if __name__ == "__main__":
    collect_benchmarks()
