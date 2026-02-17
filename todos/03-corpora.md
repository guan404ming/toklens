# 03 - Corpora Loader

## Goal

Load FLORES-200 (and optionally other datasets) for multilingual evaluation. No bundled data.

## Tasks

- [ ] Implement `corpora.py` with `load_flores(langs, split)` to download/cache FLORES-200 via HuggingFace datasets
- [ ] Support `get_texts(lang, dataset)` as unified API
- [ ] Cache downloaded data locally to avoid re-downloading
- [ ] Support user-provided text via `evaluate_text()` on Analyzer
- [ ] Add a small inline test fixture (a few sentences per language) for unit tests only
- [ ] Document supported datasets and how to add custom corpora

## Why FLORES-200

- 200+ languages with parallel sentences (parity ratio ready)
- Standard benchmark in tokenizer research (used by Tokenization Disparities paper)
- CC-BY-SA 4.0 license
- Available on HuggingFace Hub

## Acceptance Criteria

- `from toklens.corpora import get_texts` downloads and returns text for any FLORES-200 language
- Parity computation works directly on parallel sentences
- No data bundled in the package itself
- Offline unit tests use small inline fixtures, not network calls
