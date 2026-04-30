"""Step 6: Cross-toolkit comparison with Qtok (Chelombitko et al., 2024).

Pipeline:
1. Download tokenizer.json for each TokLens tokenizer (fall back to
   AutoTokenizer.save_pretrained for non-fast tokenizers).
2. Invoke Qtok CLI on the 24-tokenizer set in one pass.
3. Parse Qtok TSV outputs (basic_stats, latin_stats, cyrillic_stats) into
   per-tokenizer scalar metrics.
4. Compute Spearman ranking correlation between TokLens metrics and the
   Qtok scalars on shared dimensions, and emit a feature matrix.

Outputs:
- experiments/qtok_outputs/        (raw Qtok TSVs / PNGs / HTML)
- experiments/qtok_compare.json    (parsed scalars + correlations)
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = REPO_ROOT / "experiments" / "qtok_tokenizers"
QTOK_OUT = REPO_ROOT / "experiments" / "qtok_outputs"
QTOK_OUT.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

# Map TokLens HF id -> short label for Qtok
LABELS = {
    "openai-community/gpt2": "gpt2",
    "meta-llama/Llama-2-7b-hf": "llama2_7b",
    "meta-llama/Llama-3.1-8B": "llama31_8b",
    "Qwen/Qwen2.5-0.5B": "qwen25",
    "google/gemma-2-9b": "gemma2_9b",
    "mistralai/Mistral-7B-v0.3": "mistral_7b_v03",
    "mistralai/Mistral-Nemo-Base-2407": "mistral_nemo",
    "microsoft/Phi-3-mini-4k-instruct": "phi3_mini",
    "microsoft/phi-4": "phi4",
    "01-ai/Yi-1.5-9B": "yi15_9b",
    "tiiuae/falcon-7b": "falcon_7b",
    "bigscience/bloom-7b1": "bloom_7b1",
    "CohereForAI/c4ai-command-r-v01": "command_r",
    "stabilityai/stablelm-2-12b": "stablelm2_12b",
    "THUDM/glm-4-9b": "glm4_9b",
    "allenai/OLMo-2-1124-7B-Instruct": "olmo2_7b",
    "internlm/internlm2_5-7b-chat": "internlm25_7b",
    "HuggingFaceTB/SmolLM2-1.7B": "smollm2_17b",
    "Qwen/Qwen3-8B": "qwen3_8b",
    "deepseek-ai/DeepSeek-V3": "deepseek_v3",
}


def download_tokenizer_jsons() -> dict[str, str]:
    """Ensure each tokenizer has a tokenizer.json. Return {label: file_path}."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    paths: dict[str, str] = {}
    for hf_id, label in LABELS.items():
        out_dir = TOKENIZER_DIR / label
        out_dir.mkdir(exist_ok=True)
        tok_json = out_dir / "tokenizer.json"
        if tok_json.exists() and tok_json.stat().st_size > 1000:
            paths[label] = str(tok_json)
            print(f"[cached] {label}")
            continue
        try:
            fp = hf_hub_download(hf_id, "tokenizer.json", local_dir=str(out_dir))
            paths[label] = fp
            print(f"[downloaded] {label}: {fp}")
        except Exception as e1:
            print(f"[fallback for {label}: {e1}]")
            try:
                tk = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                tk.save_pretrained(str(out_dir))
                if tok_json.exists():
                    paths[label] = str(tok_json)
                    print(f"[converted] {label}")
                else:
                    print(f"[failed: no tokenizer.json after save] {label}")
            except Exception as e2:
                print(f"[skip {label}: {e2}]")
    return paths


def run_qtok(paths: dict[str, str]):
    """Run Qtok pipeline directly (bypass CLI plotting which fails for >28 tokenizers).

    Replicates qtok.qtok.run_it but skips the PNG plotting step that has a
    hard-coded color budget; we only need the TSV outputs.
    """
    if not paths:
        raise SystemExit("No tokenizers available")
    from qtok.qtoklib.tokenizer import load_vocab
    from qtok.qtoklib.classification import get_classification
    from qtok.qtoklib.tables import get_stats_table, get_unicode_tables, get_language_table
    import qtok as _qtok_pkg

    QTOK_DATA = Path(_qtok_pkg.__file__).resolve().parent / "data"

    def _save_tsv(file_path, data):
        with open(file_path, "w", encoding="utf-8") as fw:
            for line in data:
                fw.write("\t".join(map(str, line)) + "\n")

    print("Loading our vocabs...")
    model2vocab_tok = {lbl: load_vocab(fp) for lbl, fp in paths.items()}

    with open(QTOK_DATA / "model2vocab_tok.json") as fh:
        model2vocab = json.load(fh)
    with open(QTOK_DATA / "token2hits_tok.json") as fh:
        token2hits = json.load(fh)
    for label, vocab in model2vocab_tok.items():
        model2vocab[label] = vocab
        for token, rank in vocab.items():
            if token not in token2hits:
                token2hits[token] = [0] * len(model2vocab) + [rank]
            else:
                token2hits[token].extend(
                    [0] * (len(model2vocab) - len(token2hits[token])) + [rank]
                )

    token2meta, _ = get_classification(token2hits)
    stats_table, stats_table_p = get_stats_table(model2vocab, token2hits, token2meta)
    unicode_table_p = get_unicode_tables(model2vocab, token2hits, token2meta)
    _save_tsv(QTOK_OUT / "basic_stats_abs.tsv", stats_table)
    _save_tsv(QTOK_OUT / "basic_stats.tsv", stats_table_p)
    _save_tsv(QTOK_OUT / "unicode_stats.tsv", unicode_table_p)

    with open(QTOK_DATA / "tokens2natural_lat.json") as fh:
        lat_data = json.load(fh)
    with open(QTOK_DATA / "tokens2natural_cyr.json") as fh:
        cyr_data = json.load(fh)

    final_table_lat, _ = get_language_table(model2vocab, token2hits, token2meta, lat_data)
    final_table_cyr, _ = get_language_table(model2vocab, token2hits, token2meta, cyr_data)
    _save_tsv(QTOK_OUT / "latin_stats.tsv", final_table_lat)
    _save_tsv(QTOK_OUT / "cyrillic_stats.tsv", final_table_cyr)
    print(f"TSVs written to {QTOK_OUT}")


def parse_tsv(path: Path, our_labels: set[str]) -> pd.DataFrame:
    """Read a Qtok TSV; keep only rows whose tokenizer label is in our_labels."""
    df = pd.read_csv(path, sep="\t")
    name_col = df.columns[0]
    df = df[df[name_col].isin(our_labels)].set_index(name_col)
    return df


def build_qtok_scalars(our_labels: set[str]) -> pd.DataFrame:
    """Pull per-tokenizer scalars we'll compare to TokLens metrics."""
    basic = parse_tsv(QTOK_OUT / "basic_stats.tsv", our_labels)
    latin = parse_tsv(QTOK_OUT / "latin_stats.tsv", our_labels)
    cyr = parse_tsv(QTOK_OUT / "cyrillic_stats.tsv", our_labels)

    out = pd.DataFrame(index=sorted(our_labels))
    # Latin allocation per language we care about
    for lang in ("en", "de", "fr", "es", "pt", "tr"):
        if lang in latin.columns:
            out[f"qtok_lat_{lang}"] = latin[lang]
    # Cyrillic
    for lang in ("ru",):
        if lang in cyr.columns:
            out[f"qtok_cyr_{lang}"] = cyr[lang]
    # spaced_alpha % is closest to "word-start tokens" -> proxy for STRR
    if "spaced_alpha" in basic.columns:
        out["qtok_spaced_alpha"] = basic["spaced_alpha"]
    if "char_alpha" in basic.columns:
        out["qtok_char_alpha"] = basic["char_alpha"]
    return out


def build_toklens_scalars(label_to_hf: dict[str, str]) -> pd.DataFrame:
    """Pull comparable scalars from TokLens results for ranking comparison."""
    with open(REPO_ROOT / "experiments/toklens_results.json") as f:
        data = json.load(f)

    rows = {}
    for label, hf in label_to_hf.items():
        if hf not in data or "error" in data[hf]:
            continue
        per_lang = data[hf]["metrics"]
        row = {}
        for lang in ("en", "de", "fr", "es", "pt", "ru", "tr"):
            if lang in per_lang:
                row[f"toklens_strr_{lang}"] = per_lang[lang].get("strr")
                row[f"toklens_fert_{lang}"] = per_lang[lang].get("fertility")
                row[f"toklens_par_{lang}"] = per_lang[lang].get("parity")
        rows[label] = row
    df = pd.DataFrame(rows).T.sort_index()
    return df


def spearman_pairs(qtok: pd.DataFrame, toklens: pd.DataFrame) -> list[dict]:
    """Hand-curated metric-pair correlations; report Spearman + n + p."""
    pairs = [
        ("qtok_lat_en", "toklens_strr_en"),
        ("qtok_lat_de", "toklens_strr_de"),
        ("qtok_lat_fr", "toklens_strr_fr"),
        ("qtok_lat_es", "toklens_strr_es"),
        ("qtok_lat_pt", "toklens_strr_pt"),
        ("qtok_lat_tr", "toklens_strr_tr"),
        ("qtok_cyr_ru", "toklens_strr_ru"),
        ("qtok_cyr_ru", "toklens_par_ru"),
        ("qtok_cyr_ru", "toklens_fert_ru"),
        ("qtok_lat_en", "toklens_fert_en"),
        ("qtok_spaced_alpha", "toklens_strr_en"),
    ]
    common = qtok.index.intersection(toklens.index)
    qq = qtok.loc[common]
    tt = toklens.loc[common]

    out = []
    for q, t in pairs:
        if q not in qq.columns or t not in tt.columns:
            continue
        x = pd.to_numeric(qq[q], errors="coerce")
        y = pd.to_numeric(tt[t], errors="coerce")
        df = pd.concat([x, y], axis=1).dropna()
        if len(df) < 5:
            continue
        rho, p = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
        out.append({
            "qtok": q, "toklens": t, "n": int(len(df)),
            "spearman_r": round(float(rho), 4),
            "spearman_p": round(float(p), 6),
        })
    return out


def main():
    paths = download_tokenizer_jsons()
    if not paths:
        raise SystemExit("nothing to compare")
    run_qtok(paths)

    label_to_hf = {lbl: hf for hf, lbl in LABELS.items() if lbl in paths}
    qtok = build_qtok_scalars(set(label_to_hf.keys()))
    tl = build_toklens_scalars(label_to_hf)
    print("\nQtok scalars head:\n", qtok.head())
    print("\nTokLens scalars head:\n", tl.head())

    pairs = spearman_pairs(qtok, tl)
    print("\nSpearman ranking correlation (Qtok vs TokLens):")
    for r in pairs:
        sig = "***" if r["spearman_p"] < 0.001 else "**" if r["spearman_p"] < 0.01 else "*" if r["spearman_p"] < 0.05 else ""
        print(f"  {r['qtok']:25s} vs {r['toklens']:25s} rho={r['spearman_r']:+.3f} p={r['spearman_p']:.4f} {sig} (n={r['n']})")

    summary = {
        "n_tokenizers": int(len(qtok)),
        "qtok_scalars": qtok.round(4).reset_index().rename(columns={"index": "tokenizer"}).to_dict(orient="records"),
        "toklens_scalars": tl.round(4).reset_index().rename(columns={"index": "tokenizer"}).to_dict(orient="records"),
        "ranking_correlations": pairs,
    }
    out_path = REPO_ROOT / "experiments/qtok_compare.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
