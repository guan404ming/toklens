"""Step 4: Generate publication-quality figures.

Input: experiments/toklens_results.json, experiments/benchmark_scores.json,
       experiments/correlations.json
Output: experiments/figures/*.pdf
"""

from __future__ import annotations

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

# Colorblind-safe palette (Wong 2011)
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]

# Display names for metrics
METRIC_LABELS = {
    "fertility": "Fertility",
    "cpt": "CPT",
    "compression_ratio": "Compression",
    "strr": "STRR",
    "nsl": "NSL",
    "parity": "Parity",
    "fertility_avg": "Fertility (avg)",
    "cpt_avg": "CPT (avg)",
    "compression_ratio_avg": "Compression (avg)",
    "strr_avg": "STRR (avg)",
    "nsl_avg": "NSL (avg)",
    "parity_avg": "Parity (avg)",
}


def load_all():
    with open("experiments/toklens_results.json") as f:
        toklens = json.load(f)
    with open("experiments/benchmark_scores.json") as f:
        benchmarks = json.load(f)
    with open("experiments/correlations.json") as f:
        correlations = json.load(f)
    return toklens, benchmarks, correlations


def _get_valid_tokenizers(toklens: dict) -> list[tuple[str, dict]]:
    """Return tokenizers with valid results, sorted by display name."""
    items = [(k, v) for k, v in toklens.items() if "error" not in v]
    return sorted(items, key=lambda x: x[1]["display_name"])


def fig1_correlation_heatmap(
    correlations: dict, save_path: str = "experiments/figures/fig1_heatmap.pdf"
):
    """Figure 1: Correlation heatmap (metrics x benchmarks), English only."""
    data = correlations["correlations"]

    # English-only metrics (not _avg), ordered logically
    metric_order = ["fertility", "cpt", "compression_ratio", "strr", "nsl", "parity"]
    en_metrics = [m for m in metric_order if any(r["metric"] == m for r in data)]
    bench_order = ["IFEval", "BBH", "MATH_Lvl5", "GPQA", "MUSR", "MMLU_PRO", "Average"]
    benchmarks_list = [b for b in bench_order if any(r["benchmark"] == b for r in data)]

    matrix = np.full((len(en_metrics), len(benchmarks_list)), np.nan)
    pvals = np.full((len(en_metrics), len(benchmarks_list)), 1.0)
    for r in data:
        if r["metric"] in en_metrics and r["benchmark"] in benchmarks_list:
            i = en_metrics.index(r["metric"])
            j = benchmarks_list.index(r["benchmark"])
            matrix[i, j] = r["spearman_r"]
            pvals[i, j] = r["spearman_p"]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    metric_labels = [METRIC_LABELS.get(m, m) for m in en_metrics]
    ax.set_xticks(range(len(benchmarks_list)))
    ax.set_xticklabels(benchmarks_list, rotation=45, ha="right")
    ax.set_yticks(range(len(en_metrics)))
    ax.set_yticklabels(metric_labels)

    # Annotate cells with value and significance
    for i in range(len(en_metrics)):
        for j in range(len(benchmarks_list)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            sig = ""
            if pvals[i, j] < 0.001:
                sig = "***"
            elif pvals[i, j] < 0.01:
                sig = "**"
            elif pvals[i, j] < 0.05:
                sig = "*"
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, label="Spearman $\\rho$", shrink=0.8)
    ax.set_title("Tokenizer Metric vs Benchmark Correlation (English)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def fig1b_partial_heatmap(
    correlations: dict, save_path: str = "experiments/figures/fig1b_partial_heatmap.pdf"
):
    """Figure 1b: Partial correlation heatmap controlling for model size."""
    data = correlations["correlations"]

    metric_order = ["fertility", "cpt", "compression_ratio", "strr", "nsl", "parity"]
    en_metrics = [m for m in metric_order if any(r["metric"] == m for r in data)]
    bench_order = ["IFEval", "BBH", "MATH_Lvl5", "GPQA", "MUSR", "MMLU_PRO", "Average"]
    benchmarks_list = [b for b in bench_order if any(r["benchmark"] == b for r in data)]

    matrix = np.full((len(en_metrics), len(benchmarks_list)), np.nan)
    pvals = np.full((len(en_metrics), len(benchmarks_list)), 1.0)
    for r in data:
        if r["metric"] in en_metrics and r["benchmark"] in benchmarks_list:
            i = en_metrics.index(r["metric"])
            j = benchmarks_list.index(r["benchmark"])
            matrix[i, j] = r.get("partial_r", 0)
            pvals[i, j] = r.get("partial_p", 1.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    metric_labels = [METRIC_LABELS.get(m, m) for m in en_metrics]
    ax.set_xticks(range(len(benchmarks_list)))
    ax.set_xticklabels(benchmarks_list, rotation=45, ha="right")
    ax.set_yticks(range(len(en_metrics)))
    ax.set_yticklabels(metric_labels)

    for i in range(len(en_metrics)):
        for j in range(len(benchmarks_list)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            sig = ""
            if pvals[i, j] < 0.001:
                sig = "***"
            elif pvals[i, j] < 0.01:
                sig = "**"
            elif pvals[i, j] < 0.05:
                sig = "*"
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, label="Partial $\\rho$ (controlling for params)", shrink=0.8)
    ax.set_title("Partial Correlation (Controlling for Model Size)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def fig2_fertility_vs_benchmark(
    toklens: dict,
    benchmarks: dict,
    save_path: str = "experiments/figures/fig2_scatter.pdf",
):
    """Figure 2: Scatter plot of fertility vs Average benchmark score."""
    names, ferts, avgs, params = [], [], [], []
    for hf_name, data in toklens.items():
        if "error" in data:
            continue
        if hf_name not in benchmarks or not benchmarks[hf_name].get("benchmarks"):
            continue
        avg = benchmarks[hf_name]["benchmarks"].get("Average")
        if avg is None:
            continue
        en_fert = data["metrics"].get("en", {}).get("fertility")
        if en_fert is None:
            continue
        names.append(data["display_name"])
        ferts.append(en_fert)
        avgs.append(avg)
        params.append(benchmarks[hf_name]["params_b"])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        ferts, avgs, c=params, cmap="viridis", s=80, edgecolors="black", linewidth=0.5, zorder=3
    )

    # Add model labels with smart offset to reduce overlap
    for name, x, y in zip(names, ferts, avgs):
        ax.annotate(
            name, (x, y), fontsize=6, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
            arrowprops={"arrowstyle": "-", "alpha": 0.3, "lw": 0.5},
        )

    fig.colorbar(scatter, label="Model size (B params)")
    ax.set_xlabel("Fertility (English, lower = better)")
    ax.set_ylabel("Average Benchmark Score")
    ax.set_title("Fertility vs Downstream Performance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def _lang_heatmap(
    toklens: dict,
    metric: str,
    title: str,
    cmap: str,
    cbar_label: str,
    save_path: str,
    annotate: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Generic heatmap: tokenizers (rows) x languages (cols) for a given metric."""
    items = _get_valid_tokenizers(toklens)
    if not items:
        return

    tokenizer_names = [v["display_name"] for _, v in items]
    langs = sorted(items[0][1]["metrics"].keys())

    matrix = np.array([
        [data["metrics"].get(lang, {}).get(metric, np.nan) for lang in langs]
        for _, data in items
    ])

    fig, ax = plt.subplots(figsize=(12, max(6, len(tokenizer_names) * 0.4)))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(langs, rotation=45, ha="right")
    ax.set_yticks(range(len(tokenizer_names)))
    ax.set_yticklabels(tokenizer_names, fontsize=7)

    if annotate:
        for i in range(len(tokenizer_names)):
            for j in range(len(langs)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5)

    fig.colorbar(im, label=cbar_label, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def fig3_parity_heatmap(toklens: dict, save_path: str = "experiments/figures/fig3_parity.pdf"):
    """Figure 3: Parity ratio heatmap."""
    _lang_heatmap(
        toklens, "parity",
        title="Cross-lingual Parity Ratio (vs English)",
        cmap="RdYlGn_r", cbar_label="Parity (1.0 = equal to English)",
        save_path=save_path, vmin=0.5, vmax=3.0,
    )


def fig4_fertility_heatmap(
    toklens: dict, save_path: str = "experiments/figures/fig4_fertility_by_lang.pdf"
):
    """Figure 4: Fertility heatmap across languages."""
    _lang_heatmap(
        toklens, "fertility",
        title="Fertility Across Languages (lower = better)",
        cmap="YlOrRd", cbar_label="Fertility (tokens/word)",
        save_path=save_path,
    )


def fig5_strr_heatmap(
    toklens: dict, save_path: str = "experiments/figures/fig5_strr_by_lang.pdf"
):
    """Figure 5: STRR heatmap across languages."""
    _lang_heatmap(
        toklens, "strr",
        title="Single Token Retention Rate (higher = better)",
        cmap="YlGn", cbar_label="STRR",
        save_path=save_path, vmin=0, vmax=1,
    )


def fig6_cpt_heatmap(
    toklens: dict, save_path: str = "experiments/figures/fig6_cpt_by_lang.pdf"
):
    """Figure 6: Characters per token heatmap."""
    _lang_heatmap(
        toklens, "cpt",
        title="Characters Per Token (higher = more compact)",
        cmap="YlGnBu", cbar_label="CPT (chars/token)",
        save_path=save_path,
    )


def fig7_vocab_size_vs_avg(
    toklens: dict,
    benchmarks: dict,
    save_path: str = "experiments/figures/fig7_vocab_vs_benchmark.pdf",
):
    """Figure 7: Vocab size vs average benchmark score."""
    names, vocabs, avgs, params = [], [], [], []
    for hf_name, data in toklens.items():
        if "error" in data:
            continue
        if hf_name not in benchmarks or not benchmarks[hf_name].get("benchmarks"):
            continue
        avg = benchmarks[hf_name]["benchmarks"].get("Average")
        if avg is None:
            continue
        names.append(data["display_name"])
        vocabs.append(data["vocab_size"])
        avgs.append(avg)
        params.append(benchmarks[hf_name]["params_b"])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        vocabs, avgs, c=params, cmap="viridis", s=80, edgecolors="black", linewidth=0.5, zorder=3
    )
    for name, x, y in zip(names, vocabs, avgs):
        ax.annotate(
            name, (x, y), fontsize=6, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
            arrowprops={"arrowstyle": "-", "alpha": 0.3, "lw": 0.5},
        )

    fig.colorbar(scatter, label="Model size (B params)")
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Average Benchmark Score")
    ax.set_title("Vocabulary Size vs Downstream Performance")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def generate_all():
    toklens, benchmarks, correlations = load_all()
    fig1_correlation_heatmap(correlations)
    fig1b_partial_heatmap(correlations)
    fig2_fertility_vs_benchmark(toklens, benchmarks)
    fig3_parity_heatmap(toklens)
    fig4_fertility_heatmap(toklens)
    fig5_strr_heatmap(toklens)
    fig6_cpt_heatmap(toklens)
    fig7_vocab_size_vs_avg(toklens, benchmarks)
    print("\nAll figures generated.")


if __name__ == "__main__":
    generate_all()
