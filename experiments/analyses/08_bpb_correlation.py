"""Step 8: correlate held-out English BPB with TokLens English metrics.

Input:
- experiments/perplexity_results.json (BPB per model from Modal run)
- experiments/toklens_results.json    (TokLens metrics per tokenizer)

Output:
- experiments/bpb_correlation.json    (Spearman correlations)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent

EN_METRICS = ["fertility", "cpt", "compression_ratio", "strr", "nsl"]


def main():
    with open(REPO_ROOT / "experiments/data/perplexity_results.json") as f:
        ppl = json.load(f)
    with open(REPO_ROOT / "experiments/data/toklens_results.json") as f:
        toklens = json.load(f)

    PARAMS_B = {
        "Qwen/Qwen2.5-3B": 3.1,
        "Qwen/Qwen2.5-7B": 7.6,
        "Qwen/Qwen2.5-14B": 14.8,
        "meta-llama/Llama-3.1-8B": 8.0,
        "microsoft/phi-4": 14.7,
        "google/gemma-2-9b": 9.0,
        "mistralai/Mistral-7B-v0.3": 7.2,
    }

    rows = []
    for r in ppl["results"]:
        m = r["model"]
        if m not in toklens or "error" in toklens[m]:
            continue
        en = toklens[m]["metrics"].get("en", {})
        rows.append({
            "model": m,
            "bpb": r["bpb"],
            "params_b": PARAMS_B.get(m),
            "log_params": np.log(PARAMS_B[m]) if m in PARAMS_B else None,
            **{k: en.get(k) for k in EN_METRICS},
        })

    if len(rows) < 4:
        raise SystemExit(f"Too few models: {len(rows)}")

    print(f"\n{'Model':32s}  {'BPB':>7s}  " + "  ".join(f"{k:>10s}" for k in EN_METRICS))
    for r in rows:
        print(f"{r['model']:32s}  {r['bpb']:7.4f}  " +
              "  ".join(f"{(r[k] if r[k] is not None else float('nan')):10.4f}" for k in EN_METRICS))

    def _corr(rows_subset, label):
        bpb = np.array([r["bpb"] for r in rows_subset])
        out = []
        print(f"\nSpearman BPB vs TokLens English metrics ({label}, n={len(rows_subset)}):")
        for k in EN_METRICS:
            x = np.array([r[k] for r in rows_subset], dtype=float)
            if np.isnan(x).all():
                continue
            rho, p = stats.spearmanr(x, bpb)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  BPB vs {k:25s}  rho={rho:+.3f}  p={p:.4f} {sig}")
            out.append({"metric": k, "spearman_r": round(float(rho), 4),
                        "spearman_p": round(float(p), 6), "n": len(rows_subset)})
        return out

    full_corrs = _corr(rows, "all 7 models")
    rows_no_gemma = [r for r in rows if "gemma" not in r["model"].lower()]
    nogemma_corrs = _corr(rows_no_gemma, "excluding Gemma-2 (BPB outlier)")

    # Spearman of BPB vs log(params) across the 7 models
    bpb = np.array([r["bpb"] for r in rows])
    lp = np.array([r["log_params"] for r in rows])
    rho_p, p_p = stats.spearmanr(lp, bpb)
    print(f"\nBPB vs log(params), n={len(rows)}: rho={rho_p:+.3f}  p={p_p:.4f}")
    bpb_n = np.array([r["bpb"] for r in rows_no_gemma])
    lp_n = np.array([r["log_params"] for r in rows_no_gemma])
    rho_pn, p_pn = stats.spearmanr(lp_n, bpb_n)
    print(f"BPB vs log(params), n={len(rows_no_gemma)} (no Gemma): rho={rho_pn:+.3f}  p={p_pn:.4f}")

    bpb_vs_size = {
        "all": {"spearman_r": round(float(rho_p), 4), "spearman_p": round(float(p_p), 6), "n": len(rows)},
        "no_gemma": {"spearman_r": round(float(rho_pn), 4), "spearman_p": round(float(p_pn), 6), "n": len(rows_no_gemma)},
    }

    summary = {
        "n_models": len(rows),
        "bpb_table": rows,
        "correlations_with_bpb_full": full_corrs,
        "correlations_with_bpb_no_gemma": nogemma_corrs,
        "bpb_vs_log_params": bpb_vs_size,
    }
    out_path = REPO_ROOT / "experiments/data/bpb_correlation.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
