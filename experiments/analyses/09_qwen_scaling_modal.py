"""Step 9: Qwen2.5 fixed-tokenizer scaling on MMLU-ProX via Modal GPU.

Why: the body of the paper relies on `qwen_scaling_analysis.json` to claim
that languages with higher single-token retention rate (STRR) scale more
steeply on MMLU-ProX. Qwen2.5 0.5B and 1.5B are not in the public
MMLU-ProX paper or any leaderboard, so this script regenerates the
per-(size, language) MMLU-ProX scores ourselves to make the claim
reproducible.

Models: Qwen2.5-0.5B / 1.5B / 3B / 7B / 14B (all share one 152K tokenizer).

Languages: en, zh, ja, ko, fr, de, es, pt, ar, th, hi (intersection of
MMLU-ProX languages and the languages used elsewhere in the paper).

Eval: 5-shot chain-of-thought, exact-match on the final answer letter.
Uses the official MMLU-ProX validation/test splits via the
`li-lab/MMLU-ProX` HuggingFace dataset. Five-shot demonstrations are
drawn from the per-language validation split. The test split is
evaluated.

Run:
  modal run experiments/09_qwen_scaling_modal.py

Output:
  experiments/data/qwen_scaling_raw.json    -- per-(model, language) accuracy
  experiments/data/qwen_scaling_analysis.json (overwritten with new slopes)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
]
SIZES_B = [0.5, 1.5, 3.0, 7.0, 14.0]

LANGS = ["en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ar", "th", "hi"]

N_SHOTS = 5
MAX_NEW_TOKENS = 512  # CoT answers can be long
TEST_LIMIT = None  # None = full test split; set to e.g. 500 for a quick run

app = modal.App("toklens-qwen-scaling")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "huggingface_hub>=0.26",
        "datasets>=3.0",
        "accelerate>=1.0",
        "sentencepiece",
        "protobuf",
        "numpy",
        "vllm==0.6.4",
    )
)

cache_vol = modal.Volume.from_name("hf-cache-toklens", create_if_missing=True)


def build_prompt(question: str, options: list[str], shots: list[dict]) -> str:
    """5-shot CoT prompt; same format the MMLU-ProX paper uses."""
    parts = []
    for s in shots:
        opts = "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(s["options"]))
        parts.append(
            f"Question: {s['question']}\nOptions:\n{opts}\n"
            f"Answer: Let's think step by step. {s['cot_content']}"
        )
    opts = "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(options))
    parts.append(
        f"Question: {question}\nOptions:\n{opts}\nAnswer: Let's think step by step."
    )
    return "\n\n".join(parts)


_ANS_RE = re.compile(r"answer is\s*\(?([A-J])\)?", re.IGNORECASE)


def extract_letter(text: str) -> str | None:
    m = _ANS_RE.search(text)
    if m:
        return m.group(1).upper()
    # Fallback: last A-J letter that stands alone
    cands = re.findall(r"\b([A-J])\b", text)
    return cands[-1] if cands else None


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=6 * 3600,
    volumes={"/hf_cache": cache_vol},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def eval_model_all_langs(model_id: str, langs: list[str], n_shots: int,
                         max_new_tokens: int, test_limit: int | None) -> dict:
    """Evaluate one model on all languages. Reuses the loaded weights."""
    import os
    import time

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache"

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    print(f"[{model_id}] loading vLLM...", flush=True)
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop=["Question:"])

    out = {"model": model_id, "languages": {}}
    for lang in langs:
        t0 = time.time()
        print(f"[{model_id}/{lang}] loading dataset...", flush=True)
        ds = load_dataset("li-lab/MMLU-ProX", lang)
        val = list(ds["validation"])
        test = list(ds["test"])
        if test_limit:
            test = test[:test_limit]

        # Sample n_shots from validation. Stable across runs.
        shots = val[:n_shots]
        prompts = [build_prompt(ex["question"], ex["options"], shots) for ex in test]

        print(f"[{model_id}/{lang}] generating {len(prompts)} prompts...", flush=True)
        gens = llm.generate(prompts, sp)

        correct = 0
        n = len(test)
        for ex, g in zip(test, gens):
            pred = extract_letter(g.outputs[0].text)
            gold = ex.get("answer") or ex.get("label")
            if isinstance(gold, int):
                gold = chr(65 + gold)
            if pred is not None and pred == gold:
                correct += 1
        acc = 100.0 * correct / max(n, 1)
        out["languages"][lang] = {
            "accuracy": acc,
            "n_correct": correct,
            "n_total": n,
            "elapsed_s": time.time() - t0,
        }
        print(f"[{model_id}/{lang}] acc={acc:.1f}  ({n} examples, {time.time()-t0:.0f}s)",
              flush=True)
    return out


@app.local_entrypoint()
def main():
    """Drives all 5 models across 11 languages, then computes slopes."""
    import numpy as np
    from scipy import stats

    print(f"Evaluating {len(MODELS)} models on {len(LANGS)} languages...")
    raw_results = {}
    for model_id in MODELS:
        res = eval_model_all_langs.remote(
            model_id, LANGS, N_SHOTS, MAX_NEW_TOKENS, TEST_LIMIT
        )
        raw_results[model_id] = res
        # Save partial results after each model.
        out_path = REPO_ROOT / "experiments/data/qwen_scaling_raw.json"
        out_path.write_text(json.dumps(raw_results, indent=2))
        print(f"Saved partial results to {out_path}")

    # Compute slopes: regress accuracy on log10(params) per language.
    log_params = np.log10(SIZES_B)
    slopes = {}
    for lang in LANGS:
        ys = np.array([raw_results[m]["languages"][lang]["accuracy"] for m in MODELS])
        res = stats.linregress(log_params, ys)
        slopes[lang] = {"slope": float(res.slope), "r2": float(res.rvalue ** 2)}

    analysis = {
        "method": "Qwen2.5 fixed-tokenizer scaling analysis",
        "description": (
            "All Qwen2.5 models share the same 152K tokenizer. We regress "
            "MMLU-ProX 5-shot CoT accuracy on log10(params) per language, "
            "then correlate the scaling slope with tokenizer metrics."
        ),
        "tokenizer": "Qwen/Qwen2.5-1.5B",
        "sizes_b": SIZES_B,
        "n_shots": N_SHOTS,
        "source": "MMLU-ProX (li-lab/MMLU-ProX), evaluated locally via vLLM",
        "slopes": slopes,
    }
    out_path = REPO_ROOT / "experiments/data/qwen_scaling_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    print(f"Saved {out_path}")
    for lang, info in slopes.items():
        print(f"  {lang}: slope={info['slope']:.2f}  r2={info['r2']:.3f}")
