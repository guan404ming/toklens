"""Step 3: Correlation analysis between TokLens metrics and benchmarks.

Input: experiments/toklens_results.json, experiments/benchmark_scores.json
Output: experiments/correlations.csv, experiments/correlations.json
"""

from __future__ import annotations

import csv
import json

import numpy as np
from scipy import stats


def load_data() -> tuple[dict, dict]:
    with open("experiments/toklens_results.json") as f:
        toklens = json.load(f)
    with open("experiments/benchmark_scores.json") as f:
        benchmarks = json.load(f)
    return toklens, benchmarks


def build_paired_data(
    toklens: dict, benchmarks: dict
) -> tuple[list[str], dict[str, list[float]], dict[str, list[float]], list[float], list[float]]:
    """Build aligned arrays of metrics and benchmarks for models that have both."""
    models = []
    metric_arrays: dict[str, list[float]] = {}
    bench_arrays: dict[str, list[float]] = {}
    params_list: list[float] = []
    vocab_list: list[float] = []

    for hf_name in toklens:
        if "error" in toklens[hf_name]:
            continue
        if hf_name not in benchmarks or not benchmarks[hf_name].get("benchmarks"):
            continue

        models.append(hf_name)
        params_list.append(benchmarks[hf_name]["params_b"])
        vocab_list.append(toklens[hf_name]["vocab_size"])

        # Average TokLens metrics across languages (English only for simplicity first)
        en_metrics = toklens[hf_name]["metrics"].get("en", {})
        for metric_name, val in en_metrics.items():
            metric_arrays.setdefault(metric_name, []).append(val)

        # Average across all languages
        all_langs = toklens[hf_name]["metrics"]
        for metric_name in en_metrics:
            vals = [all_langs[lang].get(metric_name, 0) for lang in all_langs]
            key = f"{metric_name}_avg"
            metric_arrays.setdefault(key, []).append(np.mean(vals))

        # Benchmark scores
        for bench_name, val in benchmarks[hf_name]["benchmarks"].items():
            bench_arrays.setdefault(bench_name, []).append(val)

    return models, metric_arrays, bench_arrays, params_list, vocab_list


def _partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Compute partial Spearman correlation of x and y controlling for z.

    Uses the standard formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    P-value from t-test with n-3 degrees of freedom.
    """
    r_xy = stats.spearmanr(x, y).statistic
    r_xz = stats.spearmanr(x, z).statistic
    r_yz = stats.spearmanr(y, z).statistic

    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return 0.0, 1.0

    r_partial = (r_xy - r_xz * r_yz) / denom
    # t-test for significance
    n = len(x)
    t_stat = r_partial * np.sqrt((n - 3) / (1 - r_partial**2 + 1e-10))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 3))
    return float(r_partial), float(p_val)


def compute_correlations(
    output_csv: str = "experiments/correlations.csv",
    output_json: str = "experiments/correlations.json",
) -> dict:
    """Compute Spearman and Pearson correlations with p-values."""
    toklens, benchmarks = load_data()
    models, metric_arrays, bench_arrays, params_list, _vocab_list = build_paired_data(
        toklens, benchmarks
    )

    print(f"Models with both metrics and benchmarks: {len(models)}")
    for m in models:
        print(f"  {m}")

    params_arr = np.array(params_list)
    results = []
    n_comparisons = 0

    for metric_name, metric_vals in metric_arrays.items():
        for bench_name, bench_vals in bench_arrays.items():
            if len(metric_vals) != len(bench_vals):
                continue

            x = np.array(metric_vals)
            y = np.array(bench_vals)

            # Skip if no variance
            if np.std(x) == 0 or np.std(y) == 0:
                continue

            spearman_r, spearman_p = stats.spearmanr(x, y)
            pearson_r, pearson_p = stats.pearsonr(x, y)

            # Partial correlation controlling for model size
            partial_r, partial_p = _partial_correlation(x, y, params_arr)

            n_comparisons += 1

            results.append({
                "metric": metric_name,
                "benchmark": bench_name,
                "spearman_r": round(float(spearman_r), 4),
                "spearman_p": round(float(spearman_p), 6),
                "pearson_r": round(float(pearson_r), 4),
                "pearson_p": round(float(pearson_p), 6),
                "partial_r": round(partial_r, 4),
                "partial_p": round(partial_p, 6),
                "n": len(models),
            })

    # Bonferroni correction
    alpha = 0.05
    corrected_alpha = alpha / max(n_comparisons, 1)
    for r in results:
        r["bonferroni_alpha"] = round(corrected_alpha, 6)
        r["spearman_sig"] = r["spearman_p"] < corrected_alpha
        r["pearson_sig"] = r["pearson_p"] < corrected_alpha
        r["partial_sig"] = r["partial_p"] < corrected_alpha

    # Sort by absolute spearman correlation
    results.sort(key=lambda r: -abs(r["spearman_r"]))

    # Save CSV
    if results:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved to {output_csv}")

    # Save JSON
    summary = {
        "n_models": len(models),
        "models": models,
        "n_comparisons": n_comparisons,
        "bonferroni_alpha": corrected_alpha,
        "correlations": results,
    }
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved to {output_json}")

    # Print top correlations
    print(f"\nTop 10 correlations (Spearman, n={len(models)}):")
    for r in results[:10]:
        sig = "*" if r["spearman_sig"] else ""
        print(
            f"  {r['metric']:25s} vs {r['benchmark']:12s} "
            f"rho={r['spearman_r']:+.3f} p={r['spearman_p']:.4f} {sig}"
        )

    print("\nTop 10 partial correlations (controlling for model size):")
    partial_sorted = sorted(results, key=lambda r: -abs(r["partial_r"]))
    for r in partial_sorted[:10]:
        sig = "*" if r["partial_sig"] else ""
        print(
            f"  {r['metric']:25s} vs {r['benchmark']:12s} "
            f"r_partial={r['partial_r']:+.3f} p={r['partial_p']:.4f} {sig}"
        )

    return summary


if __name__ == "__main__":
    compute_correlations()
