# 05 - Compare

## Goal

Implement side-by-side comparison of two or more tokenizers.

## Tasks

- [ ] `compare(tokenizer_a, tokenizer_b, langs)` function
- [ ] Compute all metrics for both tokenizers on same corpora
- [ ] Return a `ComparisonReport` with diff view
- [ ] `ComparisonReport.summary()` prints a side-by-side table
- [ ] `ComparisonReport.vocab_overlap()` shows shared/unique tokens
- [ ] Unit tests comparing two known tokenizers

## Acceptance Criteria

- `compare("gpt2", "meta-llama/Llama-3.1-8B", langs=["en"])` returns valid comparison
- Summary clearly shows which tokenizer is better on each metric
