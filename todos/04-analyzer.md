# 04 - Analyzer

## Goal

Implement the main `Analyzer` class that loads a tokenizer and runs evaluation.

## Tasks

- [ ] `Analyzer.from_pretrained(name_or_path)` loads any HF tokenizer
- [ ] `Analyzer.evaluate(langs, domains)` runs all metrics on built-in corpora
- [ ] `Analyzer.evaluate_text(text)` runs metrics on user-provided text
- [ ] Returns a `Report` object with per-metric accessors
- [ ] Handle tokenizer loading errors gracefully
- [ ] Unit tests with a small/fast tokenizer (e.g., gpt2)

## Acceptance Criteria

- `Analyzer.from_pretrained("gpt2").evaluate(langs=["en"])` returns a valid Report
- Works with HF tokenizers, tiktoken-style, and sentencepiece-backed tokenizers
