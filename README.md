# TokLens: Looking Beyond Fertility in Tokenizer Evaluation

A diagnostic toolkit for multilingual tokenizer analysis. TokLens provides a unified, reusable library for computing tokenizer quality metrics that researchers currently reimplement from scratch in every paper.

## Motivation

Every multilingual tokenizer paper reimplements the same metrics (fertility, STRR, parity) as one-off scripts. Existing tools like Qtok focus on vocabulary-level introspection (Unicode coverage, script distribution) but do not compute text-level metrics. Meanwhile, recent work (NAACL 2024) shows that common metrics like fertility and parity don't always predict downstream performance, raising the question: **which tokenizer metrics actually matter?**

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
| IPS | SampoNLP 2025 | Integrated Performance Score. Balances morpheme coverage vs over-splitting. |
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

## Corpora

TokLens uses [FLORES-200](https://github.com/facebookresearch/flores) as its default multilingual corpus:

- **200+ languages** with parallel sentences (parity ratio ready)
- **Standard benchmark** in tokenizer research (CC-BY-SA 4.0)
- **Auto-downloaded** and cached locally via HuggingFace datasets

Users can also evaluate on custom text via `Analyzer.evaluate_text()`.

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
├── data/                    # cached corpora (gitignored)
├── tests/
├── pyproject.toml
└── README.md
```

## Dependencies

- `tokenizers` (HuggingFace, Rust-backed, handles the heavy tokenization)
- `datasets` (HuggingFace, for loading FLORES-200)
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

### Foundational

- [How Good is Your Tokenizer?](https://aclanthology.org/2021.acl-long.243/) (ACL 2021) - 400+ citations, defines fertility metric
- [Tokenizer Choice: Negligible or Crucial?](https://aclanthology.org/2024.findings-naacl.247/) (NAACL 2024) - shows fertility/parity not always predictive of downstream performance

### Metrics and Evaluation (2025-2026)

- [Beyond Fertility: Analyzing STRR](https://arxiv.org/abs/2510.09947) (arXiv, Oct 2025) - proposes Single Token Retention Rate metric
- [TokSuite](https://arxiv.org/abs/2512.20757) (arXiv, Dec 2025) - 14 controlled models to isolate tokenizer impact
- [Tokenization Disparities as Infrastructure Bias](https://arxiv.org/abs/2510.12389) (arXiv, Oct 2025) - fairness analysis across 200+ languages with FLORES-200
- [Evaluating Morphological Alignment of Tokenizers in 70 Languages](https://arxiv.org/abs/2507.06378) (arXiv, Jul 2025) - morphological alignment evaluation
- [Stop Taking Tokenizers for Granted](https://arxiv.org/abs/2601.13260) (arXiv, Jan 2026) - position paper arguing tokenizers are core design decisions

### Toolkits

- [Qtok](https://github.com/nup-csai/Qtok) (arXiv 2410.12989, Oct 2024) - vocabulary-level analysis (Unicode coverage, script distribution). Does **not** compute fertility, STRR, parity, CPT, or compression ratio. Complementary to TokLens.
- [SampoNLP](https://arxiv.org/abs/2601.04469) (arXiv, Jan 2025) - morphological lexicon toolkit for Uralic languages, proposes IPS metric
- tokviz - visualization only, 12 stars, inactive

### Cross-lingual Tokenizer Design

- [Parallel Tokenizers](https://arxiv.org/abs/2510.06128) (arXiv, Oct 2025) - cross-lingual vocabulary alignment via bilingual dictionaries
- [Comparative analysis of subword tokenization for Indian languages](https://arxiv.org/abs/2505.16868) (arXiv, May 2025)
- [Entropy-Driven Pre-Tokenization for BPE](https://arxiv.org/abs/2506.15889) (arXiv, Jun 2025)

## License

MIT
