"""Step 7: English bits-per-byte (BPB) on held-out text via Modal GPU.

Why: the Open LLM Leaderboard v2 stresses reasoning, math, instruction
following, and knowledge --- none of which are tokenization-sensitive on
Wikipedia-style English text. To check whether TokLens English metrics
correlate with a tokenization-sensitive English signal, we compute BPB
on a held-out English corpus. BPB is tokenization-invariant (it
normalizes by UTF-8 byte count of the original text), so it is
comparable across tokenizers.

Models: the 7 LME models (Qwen2.5-3B/7B/14B, Llama-3.1-8B, Phi-4,
Gemma-2-9B, Mistral-7B-v0.3).

Strategy: a single container processes all 7 models sequentially. This
avoids re-downloading shards across containers and keeps the
HuggingFace cache volume warm.

Run:
  modal run experiments/07_perplexity_modal.py

Output:
  experiments/perplexity_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    # Smaller -> larger so we get partial results fast if anything fails late.
    "Qwen/Qwen2.5-3B",
    "mistralai/Mistral-7B-v0.3",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-7B",
    "google/gemma-2-9b",
    "Qwen/Qwen2.5-14B",
    "microsoft/phi-4",
]

app = modal.App("toklens-bpb")

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
    )
)

cache_vol = modal.Volume.from_name("hf-cache-toklens", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/hf_cache": cache_vol},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def compute_bpb_all(text: str, model_ids: list[str], max_seq_len: int = 4096) -> list[dict]:
    """Compute BPB for each model_id sequentially in one container.

    BPB = (sum of -log2 P(token_i | context)) / total_bytes_of_text
    Always prepends the tokenizer's BOS token if available, since some
    SentencePiece-style models (Gemma, Mistral) underperform without it.
    """
    import gc
    import math
    import os
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache"

    n_bytes_total = len(text.encode("utf-8"))
    print(f"corpus: {n_bytes_total} bytes, {len(text)} chars")

    results = []
    for model_id in model_ids:
        t0 = time.time()
        print(f"\n[{model_id}] loading tokenizer...", flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            print(f"[{model_id}] FAILED at tokenizer: {e}", flush=True)
            results.append({"model": model_id, "error": f"tokenizer: {e}"})
            continue
        print(f"[{model_id}] loading model...", flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            ).eval()
        except Exception as e:
            print(f"[{model_id}] FAILED at model load: {e}", flush=True)
            results.append({"model": model_id, "error": f"load: {e}"})
            continue
        load_s = time.time() - t0
        print(f"[{model_id}] loaded in {load_s:.1f}s", flush=True)

        # Always include BOS if the tokenizer has one. SentencePiece-based
        # models (Gemma 2, Mistral, Llama) train with a BOS prepended; not
        # adding it makes the first-context PPL spuriously high.
        ids = tok(text, add_special_tokens=True, return_tensors="pt").input_ids[0]
        n_tokens = ids.shape[0]

        nll_total = 0.0
        n_target = 0
        try:
            with torch.no_grad():
                for start in range(0, n_tokens, max_seq_len):
                    chunk = ids[start : start + max_seq_len].unsqueeze(0).cuda()
                    if chunk.shape[1] < 2:
                        continue
                    out = model(chunk, labels=chunk)
                    n_t = chunk.shape[1] - 1
                    nll_total += out.loss.item() * n_t
                    n_target += n_t
        except Exception as e:
            print(f"[{model_id}] FAILED at forward: {e}", flush=True)
            results.append({"model": model_id, "error": f"forward: {e}"})
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        bits = nll_total / math.log(2)
        bpb = bits / n_bytes_total
        nll_per_token = nll_total / n_target

        elapsed = time.time() - t0
        print(
            f"[{model_id}] n_bytes={n_bytes_total} n_tokens={n_tokens} "
            f"n_target={n_target} bpb={bpb:.4f} ({elapsed:.0f}s)",
            flush=True,
        )
        results.append({
            "model": model_id,
            "n_bytes": int(n_bytes_total),
            "n_tokens": int(n_tokens),
            "n_target": int(n_target),
            "nll_total_nats": float(nll_total),
            "nll_per_token_nats": float(nll_per_token),
            "bpb": float(bpb),
            "elapsed_s": elapsed,
        })

        # free GPU memory before loading the next model
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    return results


@app.function(image=image, timeout=300)
def fetch_held_out_english(target_chars: int = 60000) -> str:
    """Fetch wikitext-2-raw-v1 test split (held-out English, standard LM eval)."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    chunks = []
    n = 0
    for row in ds:
        t = row["text"].strip()
        if not t:
            continue
        chunks.append(t)
        n += len(t)
        if n >= target_chars:
            break
    text = "\n\n".join(chunks)[:target_chars]
    print(f"Fetched held-out English: {len(text)} chars, {len(text.encode('utf-8'))} bytes")
    return text


@app.local_entrypoint()
def main():
    print("Fetching held-out English corpus...")
    text = fetch_held_out_english.remote(60000)

    print(f"\nComputing BPB on {len(MODELS)} models sequentially in one container...")
    results = compute_bpb_all.remote(text, MODELS)

    print("\n=== summary ===")
    for r in results:
        if "error" in r:
            print(f"  {r['model']:32s}  FAILED: {r['error']}")
        else:
            print(f"  {r['model']:32s}  BPB={r['bpb']:.4f}  ({r['elapsed_s']:.0f}s)")

    out_path = REPO_ROOT / "experiments" / "perplexity_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "corpus": "wikitext-2-raw-v1 test (first ~60K chars)",
            "metric": "bits per byte (with tokenizer BOS prepended)",
            "n_models": len(results),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved {out_path}")
