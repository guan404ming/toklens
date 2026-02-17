# TokLens: Looking Beyond Fertility in Tokenizer Evaluation

A diagnostic toolkit for multilingual tokenizer analysis. TokLens provides a unified, reusable library for computing tokenizer quality metrics that researchers currently reimplement from scratch in every paper.

## Motivation

Every multilingual tokenizer paper reimplements the same metrics (fertility, STRR, parity) as one-off scripts. No general-purpose, installable library exists. Meanwhile, recent work (NAACL 2024) shows that common metrics like fertility and parity don't always predict downstream performance, raising the question: **which tokenizer metrics actually matter?**

TokLens aims to:
1. Provide a standard toolkit so researchers stop reimplementing metrics
2. Investigate which metrics are predictive of downstream task performance
3. Propose new metrics that go beyond fertility

## Research Question

> Given a tokenizer, without training a model, can we predict how well it will perform on downstream tasks?

## Metrics

| Metric | Source | Description |
|---|---|---|
| Fertility | ACL 2021 | Average tokens per word. Lower = better compression. |
| CPT | Multiple | Characters per token. Measures granularity. |
| STRR | arXiv 2510.09947 | Single Token Retention Rate. Proportion of words kept as single tokens. |
| Parity Ratio | TokSuite | Cross-lingual fairness: ratio of tokenized lengths for parallel sentences. |
| NSL | Multiple | Normalized Sequence Length. Length-normalized comparison across tokenizers. |
| Compression Ratio | Multiple | Bytes per token. Raw efficiency measure. |
| Vocab Overlap | New | Intersection of two tokenizer vocabularies. |
| TBD: new metrics | This work | Metrics with stronger predictive power for downstream performance. |

## API Design

```python
from toklens import Analyzer

# Load any HuggingFace tokenizer
analyzer = Analyzer.from_pretrained("meta-llama/Llama-3.1-8B")

# Evaluate across languages
report = analyzer.evaluate(langs=["en", "zh", "ja", "ar", "hi", "de", "tr"])

# Access individual metrics
report.fertility()          # per-language fertility scores
report.strr()               # single token retention rate per language
report.parity(ref="en")     # parity ratio vs English
report.compression_ratio()  # bytes per token

# Compare two tokenizers
from toklens import compare

diff = compare(
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-7B",
    langs=["en", "zh", "ja"]
)
diff.summary()
diff.plot()

# Full report
report.summary()    # table of all metrics
report.plot()       # matplotlib visualization
report.to_latex()   # LaTeX table for papers
```

## Built-in Corpora

TokLens ships with curated multilingual test texts covering:

- **Languages**: English, Chinese, Japanese, Arabic, Hindi, German, Turkish, Korean, Thai, Russian (expandable)
- **Domains**: formal (Wikipedia/news) and informal (social media/conversational)
- **Parallel texts**: aligned sentence pairs for parity computation

Corpora are small (a few MB total) and bundled with the package.

## Project Structure

```
toklens/
├── src/toklens/
│   ├── __init__.py
│   ├── analyzer.py      # Load tokenizer, run evaluation
│   ├── metrics.py       # All metric implementations
│   ├── corpora.py       # Built-in multilingual test data
│   ├── compare.py       # Two-tokenizer comparison
│   └── report.py        # Output: tables, plots, LaTeX
├── corpora/
│   ├── en/
│   ├── zh/
│   └── ...
├── tests/
├── pyproject.toml
└── README.md
```

## Dependencies

- `tokenizers` (HuggingFace, Rust-backed, handles the heavy tokenization)
- `numpy` (metric computation)
- `matplotlib` (optional, visualization)

No GPU required. Runs on any machine.

## Research Plan

### Phase 1: Toolkit (the library)

- Implement all standard metrics (fertility, STRR, CPT, parity, NSL, compression ratio)
- Curate multilingual corpora (10+ languages, 2 domains)
- Build comparison and reporting API
- Publish to PyPI

### Phase 2: Metric Validation (the paper)

- Collect N tokenizers: GPT-4o, Llama 3, Qwen 2.5, Gemma 2, Mistral, DeepSeek, etc.
- Compute all TokLens metrics for each tokenizer
- Correlate with known downstream benchmarks (Open LLM Leaderboard, MMLU, etc.)
- Identify which metrics have predictive power and which don't
- Propose new metrics if existing ones are insufficient

### Phase 3: Publication

- Target: ACL/EMNLP System Demonstration track
- Paper: toolkit description + metric validation experiments
- Title: "TokLens: Looking Beyond Fertility in Tokenizer Evaluation"

## Related Work

- "How Good is Your Tokenizer?" (ACL 2021) - 400+ citations, foundational
- "Tokenizer Choice: Negligible or Crucial?" (NAACL 2024) - fertility/parity not always predictive
- "Beyond Fertility: Analyzing STRR" (arXiv 2510.09947) - proposes STRR metric
- TokSuite (arXiv 2512.20757) - controlled experiments on tokenizer impact
- tokviz - visualization only, 12 stars, inactive

## License

MIT
