"""Compare two or more tokenizers side by side."""

from __future__ import annotations

from toklens import metrics
from toklens.report import Report


class ComparisonReport:
    """Side-by-side comparison of multiple tokenizer reports."""

    def __init__(self, reports: list[Report], vocab_overlap_data: dict[str, int] | None = None):
        self.reports = reports
        self.vocab_overlap_data = vocab_overlap_data

    @property
    def tokenizer_names(self) -> list[str]:
        return [r.tokenizer_name for r in self.reports]

    def summary(self) -> str:
        """Return a formatted side-by-side comparison table."""
        if not self.reports:
            return "No reports to compare"

        all_langs = sorted(set().union(*(r.langs for r in self.reports)))
        all_metrics = sorted(set().union(*(r.metric_names for r in self.reports)))

        lines = ["TokLens Comparison", "=" * 60]

        for metric in all_metrics:
            lines.append(f"\n{metric}:")
            header = f"  {'lang':<6}" + "".join(f"{n[:16]:>18}" for n in self.tokenizer_names)
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))
            for lang in all_langs:
                row = f"  {lang:<6}"
                for report in self.reports:
                    val = report.results.get(lang, {}).get(metric, float("nan"))
                    row += f"{val:>18.4f}"
                lines.append(row)

        if self.vocab_overlap_data:
            lines.append("\nVocabulary Overlap:")
            for key, val in self.vocab_overlap_data.items():
                lines.append(f"  {key}: {val}")

        return "\n".join(lines)

    def vocab_overlap(self) -> dict[str, int] | None:
        """Return vocabulary overlap data if available."""
        return self.vocab_overlap_data

    def plot(self, save_path: str | None = None):
        """Generate comparison bar charts."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install toklens[viz]")
            return None

        all_metrics = sorted(set().union(*(r.metric_names for r in self.reports)))
        all_langs = sorted(set().union(*(r.langs for r in self.reports)))
        n_tokenizers = len(self.reports)
        n_metrics = len(all_metrics)

        figsize = (5 * n_metrics, max(4, len(all_langs) * 0.4))
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]

        x = np.arange(len(all_langs))
        width = 0.8 / n_tokenizers

        for i, metric in enumerate(all_metrics):
            ax = axes[i]
            for j, report in enumerate(self.reports):
                vals = [report.results.get(lang, {}).get(metric, 0) for lang in all_langs]
                ax.barh(x + j * width, vals, width, label=report.tokenizer_name[:20])
            ax.set_yticks(x + width * (n_tokenizers - 1) / 2)
            ax.set_yticklabels(all_langs)
            ax.set_title(metric)
            ax.legend(fontsize=7)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


def compare(
    tokenizer_a: str,
    tokenizer_b: str,
    langs: list[str] | None = None,
    ref_lang: str = "en",
    split: str = "devtest",
) -> ComparisonReport:
    """Compare two tokenizers on the same corpora.

    Args:
        tokenizer_a: HuggingFace model name for first tokenizer.
        tokenizer_b: HuggingFace model name for second tokenizer.
        langs: Language codes to evaluate.
        ref_lang: Reference language for parity.
        split: FLORES-200 split.

    Returns:
        A ComparisonReport with side-by-side results.
    """
    from toklens.analyzer import Analyzer

    a = Analyzer.from_pretrained(tokenizer_a)
    b = Analyzer.from_pretrained(tokenizer_b)

    report_a = a.evaluate(langs=langs, ref_lang=ref_lang, split=split)
    report_b = b.evaluate(langs=langs, ref_lang=ref_lang, split=split)

    overlap = metrics.compute_vocab_overlap(a.tokenizer, b.tokenizer)

    return ComparisonReport([report_a, report_b], vocab_overlap_data=overlap)
