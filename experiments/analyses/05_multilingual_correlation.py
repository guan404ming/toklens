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
    with open("experiments/data/toklens_results.json") as f:
        return json.load(f)


def load_benchmarks():
    with open("experiments/data/benchmark_scores.json") as f:
        return json.load(f)


def compute_per_language_correlations(
    output_csv="experiments/data/multilingual_correlations.csv",
    output_json="experiments/data/multilingual_correlations.json",
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


def load_training_tokens(path: str = "experiments/data/training_tokens.json"):
    """Load per-model training-token counts (in trillions) and imputation flags."""
    with open(path) as f:
        data = json.load(f)
    return data["models"]


# Map HF id -> Qtok label used in experiments/qtok_outputs/*.tsv
HF_TO_QTOK_LABEL = {
    "Qwen/Qwen2.5-3B": "qwen25",
    "Qwen/Qwen2.5-7B": "qwen25",
    "Qwen/Qwen2.5-14B": "qwen25",
    "meta-llama/Llama-3.1-8B": "llama31_8b",
    "microsoft/phi-4": "phi4",
    "google/gemma-2-9b": "gemma2_9b",
    "mistralai/Mistral-7B-v0.3": "mistral_7b_v03",
}


def load_qtok_allocations(qtok_dir: str = "experiments/qtok_outputs") -> dict:
    """Build per-(qtok_label, lang) vocabulary allocation % from Qtok's TSVs.

    Returns dict: {qtok_label: {lang: percent}} where percent estimates the
    fraction of the tokenizer's *vocabulary* devoted to that language's
    script. Used as a tokenizer-training-data proxy in the LME analysis.
    """
    latin = pd.read_csv(f"{qtok_dir}/latin_stats.tsv", sep="\t").set_index("Tokenizer")
    uni = pd.read_csv(f"{qtok_dir}/unicode_stats.tsv", sep="\t").set_index("Tokenizer")

    def _sum_cols(df: pd.DataFrame, label: str, cols: list[str]) -> float:
        if label not in df.index:
            return float("nan")
        present = [c for c in cols if c in df.columns]
        if not present:
            return float("nan")
        return float(df.loc[label, present].sum())

    def cjk_pct(label):
        return _sum_cols(uni, label, ["CJK (spaced)", "CJK (inner)", "CJK (char)"])

    def hangul_pct(label):
        return _sum_cols(uni, label, ["HANGUL (spaced)", "HANGUL (char)"])

    def arabic_pct(label):
        return _sum_cols(uni, label, ["ARABIC (inner)", "ARABIC (spaced)"])

    def thai_pct(label):
        return _sum_cols(uni, label, ["THAI (inner)"])

    def devanagari_pct(label):
        return _sum_cols(uni, label, ["DEVANAGARI (spaced)", "DEVANAGARI (inner)"])

    def latin_lang_pct(label, lang):
        if label in latin.index and lang in latin.columns:
            return float(latin.loc[label, lang])
        return float("nan")

    out: dict[str, dict[str, float]] = {}
    for label in set(latin.index) | set(uni.index):
        out[label] = {
            "en": latin_lang_pct(label, "en"),
            "fr": latin_lang_pct(label, "fr"),
            "de": latin_lang_pct(label, "de"),
            "es": latin_lang_pct(label, "es"),
            "pt": latin_lang_pct(label, "pt"),
            "zh": cjk_pct(label),
            "ja": cjk_pct(label),  # Qtok aggregates Hiragana/Katakana under CJK/Other
            "ko": hangul_pct(label),
            "ar": arabic_pct(label),
            "th": thai_pct(label),
            "hi": devanagari_pct(label),
        }
    return out


def build_lme_dataframe() -> pd.DataFrame:
    """Build a long-format DataFrame for mixed-effects modeling.

    Each row = one (model, language) pair with columns:
    model, lang, log_params, log_train_tokens, train_tokens_imputed,
    qtok_alloc, log1p_qtok_alloc,
    score, fertility, cpt, compression_ratio, strr, nsl, parity
    """
    toklens = load_toklens()
    train_tokens = load_training_tokens()
    qtok_alloc = load_qtok_allocations()
    metric_names = ["fertility", "cpt", "compression_ratio", "strr", "nsl", "parity"]

    rows = []
    for hf_name, lang_scores in MMMLU_SCORES.items():
        if hf_name not in toklens or "error" in toklens[hf_name]:
            continue
        params_b = MODEL_PARAMS_B[hf_name]
        tt = train_tokens.get(hf_name, {})
        tokens_T = tt.get("tokens_T")
        imputed = tt.get("imputed", False)
        qlabel = HF_TO_QTOK_LABEL.get(hf_name)
        for lang in SHARED_LANGS:
            score = lang_scores.get(lang)
            if score is None:
                continue
            metrics = toklens[hf_name]["metrics"].get(lang, {})
            qa = qtok_alloc.get(qlabel, {}).get(lang) if qlabel else None
            row = {
                "model": hf_name,
                "lang": lang,
                "log_params": np.log(params_b),
                "log_train_tokens": np.log(tokens_T) if tokens_T else np.nan,
                "train_tokens_imputed": bool(imputed),
                "qtok_alloc": qa,
                "log1p_qtok_alloc": np.log1p(qa) if qa is not None and not np.isnan(qa) else np.nan,
                "score": score,
            }
            for m in metric_names:
                row[m] = metrics.get(m)
            rows.append(row)

    return pd.DataFrame(rows)


def _fit_lme(df: pd.DataFrame, metric_names: list[str], formula: str, label: str) -> list[dict]:
    """Fit one LME per metric using the given formula. Returns one dict per metric."""
    base_cols = ["model", "lang", "log_params", "log_train_tokens", "log1p_qtok_alloc", "score"]
    results = []
    print()
    print("=" * 70)
    print(f"  {label}")
    print(f"  formula: {formula}")
    print(f"  n_obs={len(df)}  n_models={df['model'].nunique()}  "
          f"n_langs={df['lang'].nunique()}")
    print("=" * 70)
    for metric in metric_names:
        cols = [c for c in base_cols if c in df.columns] + [metric]
        sub = df[cols].dropna()
        if len(sub) < 10:
            continue

        m_mean, m_std = sub[metric].mean(), sub[metric].std()
        if m_std < 1e-10:
            continue
        sub = sub.copy()
        sub["z_metric"] = (sub[metric] - m_mean) / m_std

        try:
            fit = smf.mixedlm(formula, data=sub, groups=sub["model"]).fit(reml=True)
            coef = fit.params["z_metric"]
            se = fit.bse["z_metric"]
            z_val = fit.tvalues["z_metric"]
            p_val = fit.pvalues["z_metric"]
            re_var = float(fit.cov_re.iloc[0, 0])

            row = {
                "label": label,
                "metric": metric,
                "beta": round(float(coef), 4),
                "se": round(float(se), 4),
                "z": round(float(z_val), 4),
                "p": round(float(p_val), 6),
                "re_variance": round(re_var, 4),
                "n_obs": len(sub),
                "n_models": sub["model"].nunique(),
                "n_langs": sub["lang"].nunique(),
                "aic": round(float(fit.aic), 2),
                "bic": round(float(fit.bic), 2),
            }
            for cov in ("log_params", "log_train_tokens", "log1p_qtok_alloc"):
                if cov in fit.params.index:
                    row[f"{cov}_beta"] = round(float(fit.params[cov]), 4)
                    row[f"{cov}_p"] = round(float(fit.pvalues[cov]), 6)

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            extras = []
            if "log_params_beta" in row:
                extras.append(f"log_params: beta={row['log_params_beta']:+.3f} p={row['log_params_p']:.4f}")
            if "log_train_tokens_beta" in row:
                extras.append(f"log_tt: beta={row['log_train_tokens_beta']:+.3f} p={row['log_train_tokens_p']:.4f}")
            if "log1p_qtok_alloc_beta" in row:
                extras.append(f"log_qa: beta={row['log1p_qtok_alloc_beta']:+.3f} p={row['log1p_qtok_alloc_p']:.4f}")
            print(
                f"    {metric:20s} beta={coef:+.3f} SE={se:.3f} z={z_val:+.3f} "
                f"p={p_val:.4f} {sig:4s}  " + "  ".join(extras) +
                f"  RE_var={re_var:.2f}"
            )
            results.append(row)
        except Exception as e:
            print(f"    {metric:20s} FAILED: {e}")
    return results


def run_lme_analysis(
    output_csv="experiments/data/multilingual_lme_results.csv",
    output_json="experiments/data/multilingual_lme_results.json",
):
    """Run LME models with four specifications:
    M1: score ~ z(metric) + log_params + (1|model)
    M2: + log_train_tokens (full)
    M3: + log_train_tokens (drop imputed)
    M4: + log1p(qtok_alloc) as tokenizer-training-data proxy
    """
    df = build_lme_dataframe()
    metric_names = ["fertility", "cpt", "compression_ratio", "strr", "nsl", "parity"]

    m1 = _fit_lme(df, metric_names,
                  "score ~ z_metric + log_params",
                  "M1: original (params only)")
    m2 = _fit_lme(df, metric_names,
                  "score ~ z_metric + log_params + log_train_tokens",
                  "M2: + log(training_tokens)  [includes imputed]")
    df_clean = df[~df["train_tokens_imputed"]]
    m3 = _fit_lme(df_clean, metric_names,
                  "score ~ z_metric + log_params + log_train_tokens",
                  "M3: + log(training_tokens)  [excluding imputed]")
    m4 = _fit_lme(df, metric_names,
                  "score ~ z_metric + log_params + log1p_qtok_alloc",
                  "M4: + log1p(qtok_alloc)  [vocab-derived tokenizer-data proxy]")

    # Leave-one-language-out cross-validation on M1 spec, focused on STRR.
    print()
    print("=" * 70)
    print("  LOLO: leave-one-language-out CV on M1 (STRR)")
    print("=" * 70)
    lolo_rows = []
    for held_out in sorted(df["lang"].unique()):
        sub_df = df[df["lang"] != held_out]
        cols = ["model", "lang", "log_params", "score", "strr"]
        sub = sub_df[cols].dropna()
        if len(sub) < 10:
            continue
        m_mean, m_std = sub["strr"].mean(), sub["strr"].std()
        if m_std < 1e-10:
            continue
        sub = sub.copy()
        sub["z_metric"] = (sub["strr"] - m_mean) / m_std
        try:
            fit = smf.mixedlm(
                "score ~ z_metric + log_params",
                data=sub,
                groups=sub["model"],
            ).fit(reml=True)
            beta = float(fit.params["z_metric"])
            z_val = float(fit.tvalues["z_metric"])
            p_val = float(fit.pvalues["z_metric"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"    held_out={held_out:3s} n={len(sub):3d}  STRR beta={beta:+.3f} z={z_val:+.3f} p={p_val:.4f} {sig}")
            lolo_rows.append({
                "held_out_lang": held_out, "metric": "strr",
                "beta": round(beta, 4), "z": round(z_val, 4), "p": round(p_val, 6),
                "n_obs": int(len(sub)),
            })
        except Exception as e:
            print(f"    held_out={held_out}: FAILED {e}")

    all_rows = m1 + m2 + m3 + m4
    if all_rows:
        all_keys = list(dict.fromkeys(k for r in all_rows for k in r))
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  CSV saved to {output_csv}")

        summary = {
            "method": "Linear Mixed-Effects Model (REML)",
            "specifications": {
                "M1": "score ~ z(metric) + log(params_b) + (1 | model)",
                "M2": "score ~ z(metric) + log(params_b) + log(train_tokens_T) + (1 | model)  [all rows; mistral imputed]",
                "M3": "score ~ z(metric) + log(params_b) + log(train_tokens_T) + (1 | model)  [imputed rows removed]",
                "M4": "score ~ z(metric) + log(params_b) + log1p(qtok_alloc_pct) + (1 | model)  [Qtok per-(model,lang) vocab allocation as tokenizer-training-data proxy]",
            },
            "n_obs_full": len(df),
            "n_obs_clean": len(df_clean),
            "n_obs_qtok": int(df["log1p_qtok_alloc"].notna().sum()),
            "n_models_full": df["model"].nunique(),
            "n_models_clean": df_clean["model"].nunique(),
            "n_langs": df["lang"].nunique(),
            "M1": m1,
            "M2": m2,
            "M3": m3,
            "M4": m4,
            "LOLO_strr": lolo_rows,
        }
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  JSON saved to {output_json}")

    return all_rows


if __name__ == "__main__":
    compute_per_language_correlations()
    run_lme_analysis()
