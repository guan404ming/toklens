# 09 - Metric Validation Experiment

## Goal

Run the core research experiment: which tokenizer metrics predict downstream performance?

## Tasks

- [ ] Collect 10+ tokenizers: GPT-4o, Llama 3, Qwen 2.5, Gemma 2, Mistral, DeepSeek, etc.
- [ ] Run TokLens on all tokenizers across all languages
- [ ] Collect downstream benchmark scores (Open LLM Leaderboard, MMLU, HumanEval, etc.)
- [ ] Compute correlation (Spearman/Pearson) between each metric and downstream scores
- [ ] Control for model size and training data
- [ ] Identify which metrics have statistical significance
- [ ] Investigate if metric combinations are more predictive than individual metrics
- [ ] Document results in experiment notebook

## Acceptance Criteria

- Correlation table: metrics x benchmarks with p-values
- Clear conclusion on which metrics matter and which don't
- Reproducible experiment with saved raw data
