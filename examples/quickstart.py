"""Quickstart example for TokLens.

Usage:
    uv run python examples/quickstart.py
"""

from toklens.analyzer import Analyzer

# Analyze a single tokenizer
analyzer = Analyzer.from_pretrained("openai-community/gpt2")
report = analyzer.evaluate(langs=["en", "zh", "de"])

# Print full report
print(report.summary())
print()

# Access individual metrics
print("Fertility:", report.fertility())
print("STRR:", report.strr())
print("Compression Ratio:", report.compression_ratio())
