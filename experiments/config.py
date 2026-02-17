"""Experiment configuration: tokenizers, languages, benchmarks."""

# Tokenizers to evaluate.
# Only models with scores on Open LLM Leaderboard v2.
# Format: (HF model name, display name, param count in billions, tokenizer_source)
# - hf_name: used for benchmark lookup on the leaderboard
# - tokenizer_source: HF repo to load tokenizer from (if different, e.g. non-gated mirror)
# Source: https://huggingface.co/datasets/open-llm-leaderboard/contents
TOKENIZERS = [
    # --- GPT family ---
    ("openai-community/gpt2", "GPT-2", 0.1, None),
    # --- Llama family (non-gated mirrors for tokenizer access) ---
    ("meta-llama/Llama-2-7b-hf", "Llama-2-7B", 6.7, "NousResearch/Llama-2-7b-hf"),
    ("meta-llama/Llama-3.1-8B", "Llama-3.1-8B", 8.0, "NousResearch/Meta-Llama-3.1-8B"),
    # --- Qwen family (multiple sizes, same tokenizer across sizes) ---
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B", 1.5, None),
    ("Qwen/Qwen2.5-3B", "Qwen2.5-3B", 3.1, None),
    ("Qwen/Qwen2.5-7B", "Qwen2.5-7B", 7.6, None),
    ("Qwen/Qwen2.5-14B", "Qwen2.5-14B", 14.8, None),
    # --- Gemma family (non-gated mirror) ---
    ("google/gemma-2-9b", "Gemma-2-9B", 9.0, "unsloth/gemma-2-9b"),
    # --- Mistral family (different tokenizers across versions) ---
    ("mistralai/Mistral-7B-v0.3", "Mistral-7B-v0.3", 7.2, None),
    ("mistralai/Mistral-Nemo-Base-2407", "Mistral-Nemo-12B", 11.6, None),
    # --- Phi family ---
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-mini", 3.8, None),
    ("microsoft/phi-4", "Phi-4", 14.7, None),
    # --- Yi ---
    ("01-ai/Yi-1.5-9B", "Yi-1.5-9B", 8.8, None),
    # --- Falcon ---
    ("tiiuae/falcon-7b", "Falcon-7B", 7.0, None),
    # --- BLOOM (multilingual, large vocab) ---
    ("bigscience/bloom-7b1", "BLOOM-7B", 7.1, None),
    # --- Command R (tokenizer-only repo, main repo is gated) ---
    ("CohereForAI/c4ai-command-r-v01", "Command-R-35B", 35.0,
     "Xenova/c4ai-command-r-v01-tokenizer"),
    # --- StableLM ---
    ("stabilityai/stablelm-2-12b", "StableLM-2-12B", 12.1, None),
    # --- GLM (sentencepiece only, needs AutoTokenizer fallback) ---
    ("THUDM/glm-4-9b", "GLM-4-9B", 9.0, "THUDM/glm-4-9b"),
    # --- OLMo ---
    ("allenai/OLMo-2-1124-7B-Instruct", "OLMo-2-7B", 7.3, None),
    # --- InternLM (base model has tokenizer.json, chat model does not) ---
    ("internlm/internlm2_5-7b-chat", "InternLM2.5-7B", 7.7, "internlm/internlm2_5-7b"),
    # --- SmolLM (small but competitive) ---
    ("HuggingFaceTB/SmolLM2-1.7B", "SmolLM2-1.7B", 1.7, None),
    # --- Qwen2.5-0.5B (smallest Qwen, same tokenizer family) ---
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B", 0.5, None),
]

# Additional tokenizers for metric-only analysis (no leaderboard scores).
# These are included in cross-lingual heatmaps but NOT in correlation analysis.
# Format: (HF tokenizer source, display name, vocab description)
EXTRA_TOKENIZERS = [
    ("Qwen/Qwen3-8B", "Qwen3-8B", "Latest Qwen generation"),
    ("deepseek-ai/DeepSeek-V3", "DeepSeek-V3", "DeepSeek V3 tokenizer"),
]

# Languages to evaluate (FLORES-200 short codes)
LANGS = [
    "en", "zh", "ja", "ar", "hi", "de", "tr",
    "ko", "th", "ru", "fr", "es", "pt", "vi", "id",
]

# Open LLM Leaderboard v2 benchmark columns
# Source: https://huggingface.co/datasets/open-llm-leaderboard/contents
BENCHMARK_COLUMNS = {
    "IFEval": "IFEval",
    "BBH": "BBH",
    "MATH_Lvl5": "MATH Lvl 5",
    "GPQA": "GPQA",
    "MUSR": "MUSR",
    "MMLU_PRO": "MMLU-PRO",
    "Average": "Average ⬆️",
}

# Language groupings for per-script analysis
SCRIPT_GROUPS = {
    "Latin": ["en", "de", "fr", "es", "pt", "tr", "vi", "id"],
    "CJK": ["zh", "ja", "ko"],
    "Arabic": ["ar"],
    "Devanagari": ["hi"],
    "Thai": ["th"],
    "Cyrillic": ["ru"],
}
