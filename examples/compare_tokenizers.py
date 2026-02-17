"""Compare two tokenizers side by side.

Usage:
    uv run python examples/compare_tokenizers.py
"""

from toklens import compare

result = compare(
    "openai-community/gpt2",
    "meta-llama/Llama-3.1-8B",
    langs=["en", "zh", "de"],
)

print(result.summary())
print()

overlap = result.vocab_overlap()
if overlap:
    print(f"Vocabulary overlap: {overlap['overlap']} tokens")
    print(f"Only in GPT-2: {overlap['only_a']}")
    print(f"Only in Llama 3: {overlap['only_b']}")
