"""Step 5: Multilingual correlation analysis.

Correlates per-language TokLens metrics with per-language MMLU-ProX scores.
This addresses the limitation of English-only benchmarks by testing whether
tokenizer quality predicts multilingual downstream performance.

Source: MMLU-ProX (Li et al., 2025), 5-shot CoT prompting results.
"""

from __future__ import annotations

import csv
import json

import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import pandas as pd

# MMLU-ProX 5-shot CoT scores per model per language.
# Source: Tables 1 in https://arxiv.org/abs/2503.10497
# Languages: en, zh, ja, ko, fr, de, es, pt, ar, th, hi
# We include only models that overlap with our TokLens evaluation.
# Note: Mistral-7B-v0.2 is close to Mistral-7B-v0.3 (same tokenizer family).
#       InternLM3-8B is close to InternLM2.5-7B (similar tokenizer).
#       Phi4-14B matches our Phi-4.
MMMLU_SCORES = {
    "Qwen/Qwen2.5-3B": {
        "en": 44.7, "zh": 37.5, "ja": 32.0, "ko": 27.8,
        "fr": 37.7, "de": 32.4, "es": 37.1, "pt": 35.9,
        "ar": 28.3, "th": 26.7, "hi": 16.3,
    },
    "Qwen/Qwen2.5-7B": {
        "en": 56.9, "zh": 49.7, "ja": 45.4, "ko": 41.8,
        "fr": 49.1, "de": 47.6, "es": 50.0, "pt": 49.6,
        "ar": 41.1, "th": 40.1, "hi": 32.2,
    },
    "Qwen/Qwen2.5-14B": {
        "en": 64.1, "zh": 57.4, "ja": 53.5, "ko": 52.1,
        "fr": 58.6, "de": 56.0, "es": 58.1, "pt": 57.7,
        "ar": 50.5, "th": 49.2, "hi": 40.2,
    },
    "meta-llama/Llama-3.1-8B": {
        "en": 43.5, "zh": 33.4, "ja": 28.7, "ko": 22.4,
        "fr": 35.4, "de": 33.9, "es": 35.7, "pt": 33.2,
        "ar": 21.5, "th": 27.3, "hi": 23.5,
    },
    "microsoft/phi-4": {
        "en": 63.7, "zh": 58.8, "ja": 54.7, "ko": 54.5,
        "fr": 62.9, "de": 62.2, "es": 63.0, "pt": 62.5,
        "ar": 54.6, "th": 49.9, "hi": 49.4,
    },
    "google/gemma-2-9b": {
        "en": 51.8, "zh": 43.3, "ja": 41.1, "ko": 40.1,
        "fr": 47.0, "de": 45.3, "es": 47.7, "pt": 47.7,
        "ar": 38.5, "th": 40.3, "hi": 39.3,
    },
    # Mistral-7B-v0.2 uses same tokenizer as v0.3
    "mistralai/Mistral-7B-v0.3": {
        "en": 31.7, "zh": 22.4, "ja": 19.3, "ko": 17.4,
        "fr": 27.1, "de": 26.4, "es": 26.8, "pt": 26.1,
        "ar": 13.4, "th": 11.8, "hi": 10.8,
    },
}

# Languages available in both TokLens and MMLU-ProX
SHARED_LANGS = ["en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ar", "th", "hi"]


def load_toklens():
    with open("experiments/toklens_results.json") as f:
        return json.load(f)


def load_benchmarks():
    with open("experiments/benchmark_scores.json") as f:
        return json.load(f)


def compute_per_language_correlations(
    output_csv="experiments/multilingual_correlations.csv",
    output_json="experiments/multilingual_correlations.json",
):
    """Correlate per-language TokLens metrics with per-language MMLU-ProX scores."""
    toklens = load_toklens()
    benchmarks = load_benchmarks()

    # Metrics to correlate
    metric_names = ["fertility", "cpt", "compression_ratio", "strr", "nsl", "parity"]

    # --- Analysis 1: Per-language pooled correlation ---
    # Pool all (model, language) pairs: for each pair, we have TokLens metric and MMLU-ProX score.
    print("=" * 60)
    print("Analysis 1: Pooled per-language correlation")
    print("  Each data point = (model, language) pair")
    print("=" * 60)

    results_pooled = []
    for metric in metric_names:
        x_vals, y_vals, params_vals = [], [], []
        for hf_name in MMMLU_SCORES:
            if hf_name not in toklens or "error" in toklens[hf_name]:
                continue
            param_b = benchmarks.get(hf_name, {}).get("params_b", 0)
            for lang in SHARED_LANGS:
                tok_metric = toklens[hf_name]["metrics"].get(lang, {}).get(metric)
                mmmlu_score = MMMLU_SCORES[hf_name].get(lang)
                if tok_metric is not None and mmmlu_score is not None:
                    x_vals.append(tok_metric)
                    y_vals.append(mmmlu_score)
                    params_vals.append(param_b)

        if len(x_vals) < 5:
            continue

        x = np.array(x_vals)
        y = np.array(y_vals)
        z = np.array(params_vals)

        if np.std(x) == 0 or np.std(y) == 0:
            continue

        rho, p = stats.spearmanr(x, y)

        # Partial correlation controlling for model size
        r_xy = stats.spearmanr(x, y).statistic
        r_xz = stats.spearmanr(x, z).statistic
        r_yz = stats.spearmanr(y, z).statistic
        denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        if denom > 1e-10:
            r_partial = (r_xy - r_xz * r_yz) / denom
            n = len(x)
            t_stat = r_partial * np.sqrt((n - 3) / (1 - r_partial**2 + 1e-10))
            p_partial = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 3))
        else:
            r_partial, p_partial = 0.0, 1.0

        results_pooled.append({
            "analysis": "pooled",
            "metric": metric,
            "benchmark": "MMLU-ProX",
            "spearman_r": round(float(rho), 4),
            "spearman_p": round(float(p), 6),
            "partial_r": round(float(r_partial), 4),
            "partial_p": round(float(p_partial), 6),
            "n": len(x_vals),
            "n_models": len(MMMLU_SCORES),
            "n_langs": len(SHARED_LANGS),
        })

    results_pooled.sort(key=lambda r: -abs(r["spearman_r"]))
    for r in results_pooled:
        sig = "***" if r["spearman_p"] < 0.001 else "**" if r["spearman_p"] < 0.01 else "*" if r["spearman_p"] < 0.05 else ""
        psig = "***" if r["partial_p"] < 0.001 else "**" if r["partial_p"] < 0.01 else "*" if r["partial_p"] < 0.05 else ""
        print(
            f"  {r['metric']:20s} rho={r['spearman_r']:+.3f} p={r['spearman_p']:.4f} {sig:4s}"
            f"  partial={r['partial_r']:+.3f} p={r['partial_p']:.4f} {psig}"
            f"  (n={r['n']})"
        )

    # --- Analysis 2: Non-English only (exclude en) ---
    print()
    print("=" * 60)
    print("Analysis 2: Non-English languages only")
    print("=" * 60)

    non_en_langs = [l for l in SHARED_LANGS if l != "en"]
    results_nonenglish = []
    for metric in metric_names:
        x_vals, y_vals, params_vals = [], [], []
        for hf_name in MMMLU_SCORES:
            if hf_name not in toklens or "error" in toklens[hf_name]:
                continue
            param_b = benchmarks.get(hf_name, {}).get("params_b", 0)
            for lang in non_en_langs:
                tok_metric = toklens[hf_name]["metrics"].get(lang, {}).get(metric)
                mmmlu_score = MMMLU_SCORES[hf_name].get(lang)
                if tok_metric is not None and mmmlu_score is not None:
                    x_vals.append(tok_metric)
                    y_vals.append(mmmlu_score)
                    params_vals.append(param_b)

        if len(x_vals) < 5:
            continue

        x = np.array(x_vals)
        y = np.array(y_vals)
        z = np.array(params_vals)

        if np.std(x) == 0 or np.std(y) == 0:
            continue

        rho, p = stats.spearmanr(x, y)

        r_xy = stats.spearmanr(x, y).statistic
        r_xz = stats.spearmanr(x, z).statistic
        r_yz = stats.spearmanr(y, z).statistic
        denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        if denom > 1e-10:
            r_partial = (r_xy - r_xz * r_yz) / denom
            n = len(x)
            t_stat = r_partial * np.sqrt((n - 3) / (1 - r_partial**2 + 1e-10))
            p_partial = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 3))
        else:
            r_partial, p_partial = 0.0, 1.0

        results_nonenglish.append({
            "analysis": "non_english",
            "metric": metric,
            "benchmark": "MMLU-ProX",
            "spearman_r": round(float(rho), 4),
            "spearman_p": round(float(p), 6),
            "partial_r": round(float(r_partial), 4),
            "partial_p": round(float(p_partial), 6),
            "n": len(x_vals),
            "n_models": len(MMMLU_SCORES),
            "n_langs": len(non_en_langs),
        })

    results_nonenglish.sort(key=lambda r: -abs(r["spearman_r"]))
    for r in results_nonenglish:
        sig = "***" if r["spearman_p"] < 0.001 else "**" if r["spearman_p"] < 0.01 else "*" if r["spearman_p"] < 0.05 else ""
        psig = "***" if r["partial_p"] < 0.001 else "**" if r["partial_p"] < 0.01 else "*" if r["partial_p"] < 0.05 else ""
        print(
            f"  {r['metric']:20s} rho={r['spearman_r']:+.3f} p={r['spearman_p']:.4f} {sig:4s}"
            f"  partial={r['partial_r']:+.3f} p={r['partial_p']:.4f} {psig}"
            f"  (n={r['n']})"
        )

    # --- Analysis 3: Per-language correlation (model-level) ---
    print()
    print("=" * 60)
    print("Analysis 3: Per-language model-level correlation")
    print("  For each language, correlate metric across models")
    print("=" * 60)

    results_perlang = []
    for lang in SHARED_LANGS:
        for metric in metric_names:
            x_vals, y_vals = [], []
            for hf_name in MMMLU_SCORES:
                if hf_name not in toklens or "error" in toklens[hf_name]:
                    continue
                tok_metric = toklens[hf_name]["metrics"].get(lang, {}).get(metric)
                mmmlu_score = MMMLU_SCORES[hf_name].get(lang)
                if tok_metric is not None and mmmlu_score is not None:
                    x_vals.append(tok_metric)
                    y_vals.append(mmmlu_score)

            if len(x_vals) < 4:
                continue

            x = np.array(x_vals)
            y = np.array(y_vals)

            if np.std(x) == 0 or np.std(y) == 0:
                continue

            rho, p = stats.spearmanr(x, y)
            results_perlang.append({
                "analysis": "per_language",
                "lang": lang,
                "metric": metric,
                "benchmark": "MMLU-ProX",
                "spearman_r": round(float(rho), 4),
                "spearman_p": round(float(p), 6),
                "n": len(x_vals),
            })

    # Print per-language results
    for lang in SHARED_LANGS:
        lang_results = [r for r in results_perlang if r["lang"] == lang]
        if not lang_results:
            continue
        lang_results.sort(key=lambda r: -abs(r["spearman_r"]))
        top = lang_results[0]
        sig = "*" if top["spearman_p"] < 0.05 else ""
        print(
            f"  {lang}: best = {top['metric']:15s} rho={top['spearman_r']:+.3f} "
            f"p={top['spearman_p']:.4f} {sig} (n={top['n']})"
        )

    # Save all results
    all_results = results_pooled + results_nonenglish + results_perlang
    if all_results:
        all_keys = list(dict.fromkeys(k for r in all_results for k in r))
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nCSV saved to {output_csv}")

    summary = {
        "models": list(MMMLU_SCORES.keys()),
        "n_models": len(MMMLU_SCORES),
        "languages": SHARED_LANGS,
        "n_languages": len(SHARED_LANGS),
        "source": "MMLU-ProX (Li et al., 2025), 5-shot CoT",
        "pooled": results_pooled,
        "non_english": results_nonenglish,
        "per_language": results_perlang,
    }
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved to {output_json}")

    return summary


MODEL_PARAMS_B = {
    "Qwen/Qwen2.5-3B": 3.1,
    "Qwen/Qwen2.5-7B": 7.6,
    "Qwen/Qwen2.5-14B": 14.8,
    "meta-llama/Llama-3.1-8B": 8.0,
    "microsoft/phi-4": 14.7,
    "google/gemma-2-9b": 9.0,
    "mistralai/Mistral-7B-v0.3": 7.2,
}


def build_lme_dataframe() -> pd.DataFrame:
    """Build a long-format DataFrame for mixed-effects modeling.

    Each row = one (model, language) pair with columns:
    model, lang, log_params, score, fertility, cpt, compression_ratio, strr, nsl, parity
    """
    toklens = load_toklens()
    metric_names = ["fertility", "cpt", "compression_ratio", "strr", "nsl", "parity"]

    rows = []
    for hf_name, lang_scores in MMMLU_SCORES.items():
        if hf_name not in toklens or "error" in toklens[hf_name]:
            continue
        params_b = MODEL_PARAMS_B[hf_name]
        for lang in SHARED_LANGS:
            score = lang_scores.get(lang)
            if score is None:
                continue
            metrics = toklens[hf_name]["metrics"].get(lang, {})
            row = {
                "model": hf_name,
                "lang": lang,
                "log_params": np.log(params_b),
                "score": score,
            }
            for m in metric_names:
                row[m] = metrics.get(m)
            rows.append(row)

    return pd.DataFrame(rows)


def run_lme_analysis(
    output_csv="experiments/multilingual_lme_results.csv",
    output_json="experiments/multilingual_lme_results.json",
):
    """Run linear mixed-effects models: score ~ metric + log_params + (1|model).

    For each TokLens metric, fits a separate model with:
    - Fixed effects: z-scored metric + log(params)
    - Random intercept: model identity
    This properly handles the non-independence of repeated measures per model.
    """
    df = build_lme_dataframe()
    metric_names = ["fertility", "cpt", "compression_ratio", "strr", "nsl", "parity"]

    print()
    print("=" * 70)
    print("Analysis 4: Linear Mixed-Effects Models")
    print("  score ~ z(metric) + log_params + (1 | model)")
    print(f"  n = {len(df)} observations, {df['model'].nunique()} models, "
          f"{df['lang'].nunique()} languages")
    print("=" * 70)

    results = []
    for metric in metric_names:
        sub = df[["model", "lang", "log_params", "score", metric]].dropna()
        if len(sub) < 10:
            continue

        # Z-score the metric for interpretable coefficients
        m_mean, m_std = sub[metric].mean(), sub[metric].std()
        if m_std < 1e-10:
            continue
        sub = sub.copy()
        sub["z_metric"] = (sub[metric] - m_mean) / m_std

        try:
            model = smf.mixedlm(
                "score ~ z_metric + log_params",
                data=sub,
                groups=sub["model"],
            )
            fit = model.fit(reml=True)

            coef = fit.params["z_metric"]
            se = fit.bse["z_metric"]
            z_val = fit.tvalues["z_metric"]
            p_val = fit.pvalues["z_metric"]

            # Also extract log_params effect
            coef_lp = fit.params["log_params"]
            p_lp = fit.pvalues["log_params"]

            # Random effect variance
            re_var = float(fit.cov_re.iloc[0, 0])

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(
                f"  {metric:20s} beta={coef:+.3f} SE={se:.3f} z={z_val:+.3f} "
                f"p={p_val:.4f} {sig:4s}"
                f"  log_params: beta={coef_lp:+.3f} p={p_lp:.4f}"
                f"  RE_var={re_var:.2f}"
            )

            results.append({
                "metric": metric,
                "beta": round(float(coef), 4),
                "se": round(float(se), 4),
                "z": round(float(z_val), 4),
                "p": round(float(p_val), 6),
                "log_params_beta": round(float(coef_lp), 4),
                "log_params_p": round(float(p_lp), 6),
                "re_variance": round(re_var, 4),
                "n_obs": len(sub),
                "n_models": sub["model"].nunique(),
                "n_langs": sub["lang"].nunique(),
                "aic": round(float(fit.aic), 2),
                "bic": round(float(fit.bic), 2),
            })
        except Exception as e:
            print(f"  {metric:20s} FAILED: {e}")

    if results:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  CSV saved to {output_csv}")

        summary = {
            "method": "Linear Mixed-Effects Model (REML)",
            "formula": "score ~ z(metric) + log(params_b) + (1 | model)",
            "n_obs": len(df),
            "n_models": df["model"].nunique(),
            "n_langs": df["lang"].nunique(),
            "results": results,
        }
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  JSON saved to {output_json}")

    return results


if __name__ == "__main__":
    compute_per_language_correlations()
    run_lme_analysis()
