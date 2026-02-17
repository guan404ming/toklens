"""Report generation: tables, plots, LaTeX, CSV export."""

from __future__ import annotations

import csv
import io
from typing import Any


class Report:
    """Evaluation report for a single tokenizer across languages."""

    def __init__(self, tokenizer_name: str, results: dict[str, dict[str, float]]):
        """Initialize report.

        Args:
            tokenizer_name: Name of the tokenizer.
            results: Dict of {lang: {metric: value}}.
        """
        self.tokenizer_name = tokenizer_name
        self.results = results

    @property
    def langs(self) -> list[str]:
        return sorted(self.results.keys())

    @property
    def metric_names(self) -> list[str]:
        if not self.results:
            return []
        first = next(iter(self.results.values()))
        return sorted(first.keys())

    def fertility(self) -> dict[str, float]:
        """Get fertility scores per language."""
        return {lang: m["fertility"] for lang, m in self.results.items() if "fertility" in m}

    def strr(self) -> dict[str, float]:
        """Get STRR scores per language."""
        return {lang: m["strr"] for lang, m in self.results.items() if "strr" in m}

    def cpt(self) -> dict[str, float]:
        """Get CPT scores per language."""
        return {lang: m["cpt"] for lang, m in self.results.items() if "cpt" in m}

    def compression_ratio(self) -> dict[str, float]:
        """Get compression ratio per language."""
        return {
            lang: m["compression_ratio"]
            for lang, m in self.results.items()
            if "compression_ratio" in m
        }

    def parity(self, ref: str = "en") -> dict[str, float]:
        """Get parity ratio per language vs reference."""
        return {lang: m["parity"] for lang, m in self.results.items() if "parity" in m}

    def nsl(self) -> dict[str, float]:
        """Get NSL scores per language."""
        return {lang: m["nsl"] for lang, m in self.results.items() if "nsl" in m}

    def to_dict(self) -> dict[str, Any]:
        """Return raw results as nested dict."""
        return {
            "tokenizer": self.tokenizer_name,
            "results": self.results,
        }

    def summary(self) -> str:
        """Return a formatted console table."""
        if not self.results:
            return f"No results for {self.tokenizer_name}"

        metrics = self.metric_names
        header = f"{'lang':<6}" + "".join(f"{m:>18}" for m in metrics)
        sep = "=" * len(header)
        lines = [f"TokLens Report: {self.tokenizer_name}", sep, header, "-" * len(header)]

        for lang in self.langs:
            row = f"{lang:<6}"
            for m in metrics:
                val = self.results[lang].get(m, float("nan"))
                row += f"{val:>18.4f}"
            lines.append(row)

        return "\n".join(lines)

    def to_latex(self) -> str:
        """Return a LaTeX table string."""
        if not self.results:
            return ""

        metrics = self.metric_names
        cols = "l" + "r" * len(metrics)
        header = "Lang & " + " & ".join(metrics) + r" \\"

        rows = []
        for lang in self.langs:
            vals = " & ".join(f"{self.results[lang].get(m, 0):.4f}" for m in metrics)
            rows.append(f"{lang} & {vals} \\\\")

        return "\n".join([
            r"\begin{table}[h]",
            rf"\caption{{TokLens: {self.tokenizer_name}}}",
            rf"\begin{{tabular}}{{{cols}}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    def to_csv(self, path: str | None = None) -> str:
        """Export to CSV. If path given, write to file. Returns CSV string."""
        output = io.StringIO()
        metrics = self.metric_names
        writer = csv.writer(output)
        writer.writerow(["lang", *metrics])
        for lang in self.langs:
            row = [lang] + [f"{self.results[lang].get(m, 0):.4f}" for m in metrics]
            writer.writerow(row)
        csv_str = output.getvalue()
        if path:
            with open(path, "w") as f:
                f.write(csv_str)
        return csv_str

    def plot(self, save_path: str | None = None) -> Any:
        """Generate a heatmap of metrics across languages.

        Args:
            save_path: If given, save figure to this path.

        Returns:
            matplotlib Figure, or None if matplotlib not installed.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install toklens[viz]")
            return None

        metrics = self.metric_names
        langs = self.langs
        data = np.array([
            [self.results[lang].get(m, 0) for m in metrics] for lang in langs
        ])

        fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.5), max(4, len(langs) * 0.5)))
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.set_yticks(range(len(langs)))
        ax.set_yticklabels(langs)
        ax.set_title(f"TokLens: {self.tokenizer_name}")
        fig.colorbar(im)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
