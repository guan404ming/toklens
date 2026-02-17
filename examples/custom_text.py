"""Evaluate a tokenizer on custom text.

Usage:
    uv run python examples/custom_text.py
"""

from toklens.analyzer import Analyzer

analyzer = Analyzer.from_pretrained("openai-community/gpt2")

# Evaluate on your own text
result = analyzer.evaluate_text(
    text="The quick brown fox jumps over the lazy dog.",
    ref_text="Le rapide renard brun saute par-dessus le chien paresseux.",
)

for metric, value in result.items():
    print(f"{metric}: {value:.4f}")
