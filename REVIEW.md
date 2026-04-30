Paper Decision
Decisionby Program Chairs25 Apr 2026, 03:34 (modified: 25 Apr 2026, 08:49)Program Chairs, Area Chairs, Reviewers, AuthorsRevisions
Decision: Accept
Official Review of Submission50 by Reviewer 3Gib
Official Reviewby Reviewer 3Gib13 Apr 2026, 16:49 (modified: 25 Apr 2026, 08:59)Program Chairs, Area Chairs, Reviewer 3Gib, AuthorsRevisions
Paper Summary:
The paper addresses different aspects of tokenization and assesses LLMs accordingly across 15 languages. Evaluating tokenizers, whether intrinsically or through downstream tasks, is an important problem that, despite tokenizers being omnipresent in modern NLP, is still not fully solved. Recent work has shown how tokenization can lead to disparity across languages and their speakers, and this paper contributes to this line of research.

Summary Of Strengths:
There are many metrics in the literature for tokenizer assessment, and I find it valuable that this paper brings them together in one place. TokLens can be seen as a suite for future tokenization assessment, and such a unified toolkit is useful for both researchers and model developers who want to audit cross-lingual fairness before training.

The factorial analysis (models × metrics × languages) is enlightening and gives a clear picture of how tokenizer quality has evolved from GPT-2 to more recent models like Qwen2.5 and Gemma-2. I also appreciate the careful statistical methodology. The three-tier analysis is well-designed: the authors do not overclaim from the English-only null result, and instead move to linear mixed-effects models for the multilingual case, which appropriately accounts for the clustered structure of the data

Summary Of Weaknesses:
I think the topic of the paper is pretty timely and I appreciate the focus on such an important task. However, there are a few major weaknesses that I hope could be addressed:

The parity metric is defined in Section 3 as a ratio over parallel text, but Section 4.2 explicitly states that the corpora are non-parallel native Wikipedia articles chosen to avoid translationese. It is unclear how parity is actually computed in this setup; are token counts normalized by character or byte count of each monolingual corpus? Is an aligned subset used? Since parity is one of the six headline metrics and appears in Figure 1 and the LME analysis, the paper should clarify this.

Open LLM Leaderboard v2 measures reasoning, math, instruction following, and knowledge; things that should depend mostly on model capability, not tokenizer efficiency on Wikipedia text. Finding "no correlation with English benchmarks" is then unsurprising and not very informative: the benchmarks don't stress tokenization. A more appropriate English test would be something like perplexity on held-out text, long-context retrieval, or tasks where token budget matters (cost-per-query, effective context). The null result is presented as a finding, but it may just reflect a mismatch between metric and benchmark.

Qtok is cited as the most directly related prior work. also a multilingual tokenizer evaluation framework covering 13 tokenizers. The authors say "TokLens differs by focusing on the correlation between intrinsic metrics and downstream benchmarks," but there is no empirical comparison: do the two toolkits agree on tokenizer rankings? Does TokLens add metrics Qtok lacks? For a toolkit paper, benchmarking against the closest existing toolkit should be standard.

Phrases like "tokenizer quality has a measurable impact on multilingual LLM performance" (abstract) and "tokenizer quality modulates scaling benefits" (Section 5.6) are causal claims, but the evidence is entirely correlational and confounded with pretraining data composition. The Limitations section admits this, but the abstract and conclusion don't soften accordingly.

Comments Suggestions And Typos:
There is an emerging line of work on mitigating tokenization unfairness (e.g., parity-based BPE and similar approaches), and there are also alternative tokenization paradigms such as character-level and pixel-based tokenization. It would have been interesting to see how TokLens metrics behave on these alternatives, or at least a short discussion on how the toolkit could be extended in that direction. I do not consider the absence of these experiments as a real weakness, since the authors acknowledge this scope limitation in the Limitations section, but adding a short pointer to mitigation methods in Related Work would strengthen the paper.

Also, the paper could benefit from a more explicit discussion on what actionable recommendations TokLens gives to tokenizer designers beyond "prioritize STRR". For example, what vocabulary size or allocation strategy should developers aim for in order to reach STRR ≥ 0.3 for most scripts? The discussion hints at this but does not go deep.

Confidence: 5 = Positive that my evaluation is correct. I read the paper very carefully and am familiar with related work.
Soundness: 2.5 = Between Poor and Acceptable: Several aspects are weak or under-supported.
Rating: 5: Marginally below acceptance threshold
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review
Official Review of Submission50 by Reviewer VLSR
Official Reviewby Reviewer VLSR03 Apr 2026, 16:53 (modified: 25 Apr 2026, 08:59)Program Chairs, Area Chairs, Reviewer VLSR, AuthorsRevisions
Paper Summary:
The authors introduce a toolkit to evaluate intrinsic tokenizer quality metrics (fertility, compression, cross-lingual parity, etc.). They compute intrinsic quality scores across 24 tokenizers and aim to find correlation with downstream task performance of existing LLMs trained with these tokenizers. They find no significant correlation for English, but do find that intrinsic tokenizer metrics significantly correlate with per-language performance in a multilingual setting.

Summary Of Strengths:
The paper is well written and easy to follow, and presents many interesting results with rigorous testing for significance.
The significant correlation of per-language performance with intrinsic tokenizer metrics has, as far as I know, not been shown before for decoder language modes.
Summary Of Weaknesses:
I believe the controlled experiment on Qwen scaling is too strongly confounded: the Qwen tokenizer training mix likely prioritizes the same languages as the data the Qwen language models are trained on. So it is quite plausible that higher intrinsic tokenizer quality metrics only implies higher performance in some language since the Qwen LMs have seen more data in that language. The authors do note this issue (Line 416), but I don't think the proposed solution (estimating pretraining data per language through Wikipedia size; Line 419) is sufficient since it would need evidence that Wikipedia size sufficiently correlates with typical amount of pretraining data in some language. An alternative may be to infer the data distribution the tokenizer is trained on (see [1]), then use the weaker assumption that the tokenizer training data follows the same distribution as the LM training data, or to use LMs where the training data is open.
The confound of model size may not be sufficiently addressed by controlling for parameters. Maybe, controlling for training tokens is also necessary. Referring to TokSuite ([2])) for a more controlled comparison would also be good.
More benchmarks and languages would of course strengthen the paper; in particular, the strong reliance solely on MMLU-ProX is problematic.
Comments Suggestions And Typos:
The numbers in Figure 1 are hard to read. I would suggest dropping the second decimal place and increasing the font size. The text in this figure could also generally be a bit larger.
Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 4 = Strong: This study provides sufficient support for all of its claims. Some extra experiments could be nice, but not essential.
Rating: 9: Top 15% of accepted papers, strong accept
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review