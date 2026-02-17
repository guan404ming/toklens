# 09 - Metric Validation Experiment

## Goal

Full experimental pipeline for a paper-ready study: which tokenizer metrics predict downstream performance?

---

## 09a - Tokenizer Collection

### Tasks

- [ ] Collect 15+ tokenizers spanning different families and vocab sizes:
  - GPT-2 (50k), GPT-4o (200k)
  - Llama 2 (32k), Llama 3 / 3.1 (128k)
  - Qwen 2.5 (152k)
  - Gemma 2 (256k)
  - Mistral v0.3 (32k), Mistral Nemo (131k)
  - DeepSeek V2 Lite, DeepSeek V3
  - Phi-3 (32k)
  - Falcon (65k)
  - Yi (64k)
  - Command R (256k)
  - BLOOM (250k)
- [ ] Record metadata for each: vocab size, training data size (if known), BPE vs Unigram, byte-level or not
- [ ] Save tokenizer metadata to `experiments/tokenizer_metadata.json`

### Acceptance Criteria

- At least 15 tokenizers with diverse vocab sizes (32k-256k)
- Both old (GPT-2, BLOOM) and new (GPT-4o, Llama 3.1) generations represented

---

## 09b - TokLens Metric Computation

### Tasks

- [ ] Run all TokLens metrics on every tokenizer across all 15 FLORES-200 languages
- [ ] Languages: en, zh, ja, ar, hi, de, tr, ko, th, ru, fr, es, pt, vi, id
- [ ] Save raw results to `experiments/toklens_results.json`
- [ ] Export flat CSV to `experiments/toklens_results.csv`
- [ ] Verify results are reproducible (run twice, compare)

### Acceptance Criteria

- 15 tokenizers x 15 languages x 6 metrics = 1350 data points minimum
- Results match across two independent runs

---

## 09c - Downstream Benchmark Collection

### Tasks

- [ ] Collect benchmark scores from public leaderboards for each model:
  - MMLU (5-shot)
  - HumanEval (pass@1)
  - GSM8K (5-shot)
  - HellaSwag
  - ARC-Challenge
  - TruthfulQA
  - WinoGrande
- [ ] Collect multilingual benchmarks where available:
  - MGSM (multilingual GSM8K)
  - XCOPA
  - XWinograd
  - FLORES translation BLEU (if applicable)
- [ ] Record model size (params) for each model to control for scale
- [ ] Save to `experiments/benchmark_scores.json`
- [ ] Document sources for each score (leaderboard URL, paper, etc.)

### Acceptance Criteria

- At least 5 benchmarks per model
- All scores traceable to a public source
- Model size recorded for every entry

---

## 09d - Correlation Analysis

### Tasks

- [ ] Compute Spearman and Pearson correlation between each metric and each benchmark
- [ ] Compute partial correlation controlling for model size (params)
- [ ] Compute partial correlation controlling for vocab size
- [ ] Test statistical significance (p-values) for all correlations
- [ ] Apply Bonferroni correction for multiple comparisons
- [ ] Identify which metrics survive correction (alpha=0.05)
- [ ] Save correlation matrix to `experiments/correlations.csv`

### Acceptance Criteria

- Full correlation matrix: metrics x benchmarks with r, p, corrected-p
- Clear list of statistically significant predictors
- Partial correlations show effect beyond model size

---

## 09e - Metric Combination Analysis

### Tasks

- [ ] Fit simple linear regression: benchmark ~ single metric (for each pair)
- [ ] Fit multiple regression: benchmark ~ all metrics
- [ ] Compute R-squared and adjusted R-squared
- [ ] Run stepwise feature selection to find best metric subset
- [ ] Test if metric combinations outperform individual metrics
- [ ] Check for multicollinearity (VIF) between metrics
- [ ] Save regression results to `experiments/regression_results.json`

### Acceptance Criteria

- R-squared comparison: single metric vs combined
- Best subset identified for each benchmark
- Multicollinearity report (drop metrics with VIF > 10)

---

## 09f - Per-Language Analysis

### Tasks

- [ ] Compute correlations separately for each language group:
  - Latin-script (en, de, fr, es, pt, tr, vi, id)
  - CJK (zh, ja, ko)
  - Indic (hi)
  - Arabic script (ar)
  - Thai (th)
  - Cyrillic (ru)
- [ ] Test if metric predictiveness varies by script/language family
- [ ] Identify languages where metrics fail (poor correlation)
- [ ] Analyze fairness: does high fertility in language X correlate with lower benchmark scores for X-language tasks?

### Acceptance Criteria

- Per-language-group correlation tables
- Clear finding on whether metrics work equally across scripts

---

## 09g - New Metric Exploration

### Tasks

- [ ] If existing metrics are weak predictors, explore new candidates:
  - Entropy of token length distribution
  - Vocabulary utilization rate (% of vocab used on typical text)
  - Cross-lingual token sharing rate
  - Morphological alignment score (if morpheme data available)
  - Effective vocabulary size per language
- [ ] Implement promising new metrics in `metrics.py`
- [ ] Re-run correlation analysis with new metrics included
- [ ] Test if new metrics improve prediction over existing ones

### Acceptance Criteria

- At least 2 new metric candidates explored
- Comparison table: existing vs new metrics predictive power

---

## 09h - Visualization and Figures

### Tasks

- [ ] Correlation heatmap: metrics x benchmarks (Figure 1)
- [ ] Scatter plots: best metric vs benchmark score, colored by model family (Figure 2)
- [ ] Per-language metric comparison: bar chart across tokenizers (Figure 3)
- [ ] Parity ratio heatmap: languages x tokenizers (Figure 4)
- [ ] Regression fit plots: predicted vs actual benchmark scores (Figure 5)
- [ ] Vocabulary size vs metric values: control variable visualization (Figure 6)
- [ ] Save all figures as PDF/SVG to `experiments/figures/`

### Acceptance Criteria

- All figures publication-quality (300 DPI, clear labels, colorblind-safe palette)
- Figures tell a coherent story matching paper narrative

---

## 09i - Reproducibility Package

### Tasks

- [ ] Create `experiments/run_all.sh` that runs the full pipeline end-to-end
- [ ] Pin all dependency versions in a lockfile
- [ ] Save all intermediate data (raw results, correlations, figures)
- [ ] Write `experiments/README.md` explaining how to reproduce
- [ ] Verify a clean run from scratch produces identical results
- [ ] Estimate total runtime and document it

### Acceptance Criteria

- `bash experiments/run_all.sh` reproduces all results from scratch
- Total runtime documented
- All data and figures committed to repo
