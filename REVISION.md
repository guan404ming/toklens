# Camera-Ready Revision Plan

Decision: **Accept**. Below we enumerate every reviewer concern, mark its
location in the original review, and describe the planned revision.
Items are grouped by reviewer; within each reviewer they follow the order
in which the points appear in the review.

Legend for revision type:
- **[Text]** writing-only change (clarification, softening, pointer)
- **[Fig]** figure / table change
- **[Exp]** new or extended experiment / analysis
- **[Ref]** add reference / related work entry

## Progress

| ID | Status | Note |
| --- | --- | --- |
| R3Gib-W1 (parity definition) | Done | Section 3 / 4.2 / Fig 3 / Table 4 captions updated; uses character-budget normalization (matches code) |
| R3Gib-W2 (text part) | Done | New Discussion paragraph "Interpreting the English null result"; abstract softened |
| R3Gib-W4 (soften causal) | Done | Abstract / contributions / conclusion all softened |
| R3Gib-S1 (Beyond standard BPE) | Done | New Related Work paragraph; new bib entries `xue2022byt5`, `rust2023pixel`; Limitations updated |
| R3Gib-S2 (Practical Recommendations) | Done | Folded into rewritten Discussion |
| RVLSR-W2 (TokSuite ref) | Done | TokSuite (Altıntaş et al. 2025) added to `custom.bib` and cited in Related Work + Limitations |
| RVLSR-S1 (Figure 1 readability) | Done | Parity heatmap (`fig3_parity.pdf`) regenerated with one-decimal annotations + font 7; figsize trimmed so figure floats next to its reference |
| **RVLSR-W2 (LME training-token covariate)** | Done | M1/M2/M3 LME re-fit; STRR/CPT/parity stable across specs; sensitivity (drop imputed Mistral) confirms; new paragraph in 5.6 + 3-spec Appendix table; Limitations softened |
| **R3Gib-W3 (Qtok benchmarking)** | Done | New Appendix \"Comparison with Qtok\" with feature matrix + 11-pair Spearman ranking on 16 shared tokenizers; ρ=0.68--0.84 on low-resource Latin / Cyrillic, decoupled on en/de (saturation); Related Work updated |
| **RVLSR-W1 (Qwen confound)** | Done (vocabulary-derived proxy) | New M4 LME spec adds log1p(Qtok per-(model,lang) vocabulary allocation \%) as a tokenizer-training-data proxy; Qtok proxy itself p<0.001; STRR/CPT/parity/NSL stable; fertility becomes p=0.0005 (signal previously confounded with allocation); CR collapses to p=0.41 (its multilingual signal was script-byte width). Open-data LM replication (OLMo/Pythia) deferred — no public model family with both wide multilingual coverage and per-language pretraining mix; documented in Limitations. |
| **R3Gib-W2 (optional perplexity exp)** | Done | Modal A100 run on wikitext-2-test (60K bytes) for 7 LME models. Qwen scales cleanly (BPB 0.68→0.58→0.40 over 3B/7B/14B). BPB vs log(params): ρ=−0.94, p=0.005 (no Gemma) — strongly explained by size; BPB vs TokLens English metrics non-significant due to small English-side spread (5 of 7 tokenizers at STRR≈0.636). Result reframes the English null in Discussion + new Appendix B. Gemma-2 BPB outlier (1.34) flagged as bf16+softcap numerical artifact. |
| **RVLSR-W3 (additional benchmarks)** | Done (text + LOLO) | Surveyed Belebele, Global-MMLU, MMLU-ProX 0-shot, INCLUDE; none has public 7-model x 11-lang coverage. Substituted leave-one-language-out CV on MMLU-ProX in LME (STRR p<0.001 in all 11 fits, β ∈ [4.89, 6.00]) as benchmark-robustness proxy; new Appendix table; Limitations explicitly notes the survey + LOLO + future work. |

---

## Reviewer 3Gib (Rating 5, Soundness 2.5)

### R3Gib-W1. Parity metric on non-parallel corpora — **[Text]**
- **Source:** Summary of Weaknesses, paragraph 1.
- **Issue:** Section 3 defines parity as a ratio over parallel text, but
  Section 4.2 uses non-parallel native Wikipedia corpora. The reader
  cannot tell whether tokens are normalized by characters / bytes, or
  whether an aligned subset is used.
- **Plan:**
  1. In Section 3, restate the parity definition and explicitly add the
     normalized form actually used in the experiments (tokens per byte
     / per character of each monolingual corpus).
  2. In Section 4.2, add one sentence pointing back to that normalized
     form and clarify that no aligned subset is used.
  3. Update Figure 1 / Table caption to mention the normalization.

### R3Gib-W2. Open LLM Leaderboard v2 mismatch — **[Text]** + **[Exp, optional]**
- **Source:** Summary of Weaknesses, paragraph 2.
- **Issue:** OLM v2 measures reasoning / math / IF / knowledge, which
  do not stress tokenization, so the English null result is unsurprising
  and presented as overly informative.
- **Plan:**
  1. Reframe the English result in Section 5 and the abstract: it
     reflects a *metric / benchmark mismatch*, not evidence that
     tokenization is irrelevant in English.
  2. Add a short paragraph in the Discussion arguing for tokenization-
     sensitive English probes (held-out perplexity, long-context
     retrieval, cost-per-query / effective context).
  3. *Optional [Exp]:* report perplexity on a held-out English subset
     for the subset of open models we already evaluate, so the English
     side has at least one tokenization-sensitive signal.

### R3Gib-W3. No empirical comparison with Qtok — **[Exp]** + **[Text]**
- **Source:** Summary of Weaknesses, paragraph 3.
- **Issue:** Qtok is cited as the closest prior toolkit but never
  benchmarked against; reviewer expects ranking agreement / metric
  coverage.
- **Plan:**
  1. Add a new subsection (Related Work or Appendix) "Comparison with
     Qtok" containing:
     - a feature matrix (metrics implemented, languages, model coverage,
       statistical analysis, downstream linkage),
     - a quantitative comparison: Spearman correlation between Qtok's
       and TokLens's tokenizer rankings on the shared metrics, on the
       same set of tokenizers.
  2. State explicitly which TokLens metrics Qtok lacks and vice versa.

### R3Gib-W4. Causal language vs correlational evidence — **[Text]**
- **Source:** Summary of Weaknesses, paragraph 4.
- **Issue:** Abstract and Section 5.6 use causal phrasing
  ("measurable impact", "modulates scaling benefits") although the
  Limitations section already admits the evidence is correlational.
- **Plan:**
  1. Soften the abstract: replace "tokenizer quality has a measurable
     impact" with "tokenizer quality is significantly associated with
     multilingual LLM performance, after controlling for model size".
  2. Soften Section 5.6: replace "modulates scaling benefits" with
     "is correlated with the per-language scaling pattern".
  3. Mirror the same language in the Conclusion.
  4. Cross-reference the Limitations section earlier (end of Section 5)
     so readers see the caveat before the takeaway.

### R3Gib-S1. Mitigation methods and alternative tokenization paradigms — **[Ref]** + **[Text]**
- **Source:** Comments / Suggestions, paragraph 1.
- **Issue:** Reviewer asks for a pointer to parity-based BPE,
  character-level, and pixel-based tokenization, even without new
  experiments.
- **Plan:**
  1. Extend Related Work with a short paragraph "Beyond standard BPE"
     covering parity-based BPE, character-level, byte-level, and pixel
     tokenization, with citations.
  2. In Limitations / Future Work, add one sentence on how TokLens
     metrics could be applied to these alternatives.

### R3Gib-S2. Actionable recommendations beyond "prioritize STRR" — **[Text]**
- **Source:** Comments / Suggestions, paragraph 2.
- **Issue:** Discussion hints at recommendations but does not give
  concrete targets (e.g., vocabulary size or allocation for
  STRR ≥ 0.3 across scripts).
- **Plan:**
  1. Add a "Practical Recommendations" paragraph in Section 6 (or end
     of Section 5) summarizing, from our own data:
     - empirical vocabulary-size range associated with STRR ≥ 0.3 per
       script family,
     - the trade-off between vocabulary size and fertility for
       low-resource scripts,
     - guidance on minimum per-script byte allocation during BPE
       training.
  2. Reference the relevant rows of our results tables for each claim.

---

## Reviewer VLSR (Rating 9, Soundness 4)

### RVLSR-W1. Qwen scaling experiment is confounded — **[Exp]** + **[Text]**
- **Source:** Summary of Weaknesses, paragraph 1.
- **Issue:** Qwen tokenizer training mix likely matches Qwen LM training
  mix, so higher intrinsic quality may simply track training-data
  volume. Reviewer flags that proxying via Wikipedia size is too weak;
  suggests inferring the tokenizer's training distribution (ref [1])
  or using open-data LMs.
- **Plan:**
  1. **Tokenizer-data inference.** Add an experiment that estimates the
     language distribution of each tokenizer's training data using the
     method recommended by the reviewer (cited as [1]). Use this
     estimate, instead of Wikipedia size, as the per-language
     pretraining-data proxy in the LME model.
  2. **Open-data sanity check.** Re-run the Qwen scaling analysis on
     at least one open-data model family (e.g., OLMo / Pythia /
     similar) where the per-language pretraining mix is known, and
     report whether the tokenizer-quality effect persists after
     controlling for the *known* per-language token counts.
  3. Update Section 5 (scaling subsection) and Limitations to reflect
     the stronger evidence and what still remains confounded.

### RVLSR-W2. Model-size confound — control for training tokens — **[Exp]** + **[Ref]**
- **Source:** Summary of Weaknesses, paragraph 2.
- **Issue:** Controlling only for parameter count is insufficient;
  training-token count should also be a covariate. Reviewer points to
  TokSuite (ref [2]) as a more controlled comparison.
- **Plan:**
  1. Extend the LME model with a `log(training_tokens)` fixed effect
     where the value is publicly reported; for models without disclosed
     token counts, mark them and report a sensitivity analysis with
     and without imputed values.
  2. Re-report the significance of intrinsic metrics after this control.
  3. Add TokSuite to Related Work and discuss how our setup compares.

### RVLSR-W3. Strong reliance on MMLU-ProX — **[Exp]**
- **Source:** Summary of Weaknesses, paragraph 3.
- **Issue:** Multilingual evaluation depends on a single benchmark and
  a limited language set.
- **Plan:**
  1. Add at least one additional multilingual benchmark to the
     downstream side of the analysis (candidates: Global-MMLU,
     XCOPA, XStoryCloze, Belebele). Pick the one(s) with the largest
     overlap with our 15 languages.
  2. If possible, expand the language set on the intrinsic side to
     cover under-represented scripts already supported by our corpora.
  3. Re-run the LME analysis on the union of benchmarks and report
     whether the tokenizer-quality effect generalizes.

### RVLSR-S1. Figure 1 readability — **[Fig]**
- **Source:** Comments / Suggestions, paragraph 1.
- **Plan:**
  1. Drop the second decimal place in all numeric annotations.
  2. Increase the font size for both axis labels and in-cell numbers.
  3. Re-export the figure at the new size and verify legibility at
     two-column width.

---

## Cross-cutting actions

| Action | Driven by |
| --- | --- |
| Soften causal claims in abstract / conclusion | R3Gib-W4 |
| Reframe English null result | R3Gib-W2 |
| New "Comparison with Qtok" subsection | R3Gib-W3 |
| New "Practical Recommendations" paragraph | R3Gib-S2 |
| Inferred tokenizer-training distribution as covariate | RVLSR-W1 |
| Open-data LM replication of scaling result | RVLSR-W1 |
| `log(training_tokens)` covariate in LME | RVLSR-W2 |
| Additional multilingual benchmark(s) | RVLSR-W3 |
| Figure 1 typography pass | RVLSR-S1 |
| Related-Work additions: TokSuite, Qtok comparison, parity-based BPE, character-level, pixel-based | R3Gib-S1, R3Gib-W3, RVLSR-W2 |

## Out-of-scope (acknowledged in Limitations only)

- Full retraining of LMs with controlled tokenizers (compute-bound).
- Direct experiments on character-level / pixel-based tokenizers
  (R3Gib-S1 explicitly accepts a pointer instead of experiments).
