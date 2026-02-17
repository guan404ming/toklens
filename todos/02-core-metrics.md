# 02 - Core Metrics

## Goal

Implement all standard tokenizer quality metrics in `src/toklens/metrics.py`.

## Tasks

- [ ] Fertility: average tokens per word
- [ ] CPT: characters per token
- [ ] Compression Ratio: bytes per token
- [ ] STRR: single token retention rate (proportion of words kept as one token)
- [ ] NSL: normalized sequence length
- [ ] Parity Ratio: cross-lingual tokenized length ratio vs reference language
- [ ] Vocab Overlap: intersection size between two tokenizer vocabularies
- [ ] IPS: integrated performance score (morpheme coverage vs over-splitting)
- [ ] Unit tests for each metric with known expected values

## Acceptance Criteria

- Each metric is a pure function taking token IDs / text / tokenizer and returning a float or dict
- All metrics tested against hand-computed examples
- No side effects, no state
