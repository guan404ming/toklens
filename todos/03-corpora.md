# 03 - Built-in Corpora

## Goal

Curate and bundle small multilingual test texts for out-of-the-box evaluation.

## Tasks

- [ ] Select 10 languages: en, zh, ja, ar, hi, de, tr, ko, th, ru
- [ ] Collect formal domain texts (Wikipedia/news excerpts, ~1000 words each)
- [ ] Collect informal domain texts (conversational/social, ~1000 words each)
- [ ] Collect parallel sentence pairs (for parity computation, ~200 pairs per language pair)
- [ ] Store as plain text files under `src/toklens/corpora/`
- [ ] Implement `corpora.py` loader with `get_texts(lang, domain)` API
- [ ] Document data sources and licensing

## Acceptance Criteria

- `from toklens.corpora import get_texts` returns text for any supported language
- Total bundled data under 5 MB
- All texts have clear provenance and compatible licenses
