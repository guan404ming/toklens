"""Command-line interface for TokLens."""

from __future__ import annotations

import argparse
import json
import sys


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run analysis on a single tokenizer."""
    from toklens.analyzer import Analyzer

    analyzer = Analyzer.from_pretrained(args.tokenizer)
    langs = args.langs.split(",") if args.langs else None
    report = analyzer.evaluate(langs=langs, ref_lang=args.ref)

    fmt = args.format
    if fmt == "json":
        print(json.dumps(report.to_dict(), indent=2))
    elif fmt == "csv":
        print(report.to_csv())
    elif fmt == "latex":
        print(report.to_latex())
    else:
        print(report.summary())

    if args.plot:
        report.plot(save_path=args.plot)
        print(f"\nPlot saved to {args.plot}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Run comparison between two tokenizers."""
    from toklens.compare import compare

    langs = args.langs.split(",") if args.langs else None
    result = compare(args.tokenizer_a, args.tokenizer_b, langs=langs, ref_lang=args.ref)

    fmt = args.format
    if fmt == "json":
        data = {
            "tokenizers": result.tokenizer_names,
            "reports": [r.to_dict() for r in result.reports],
            "vocab_overlap": result.vocab_overlap_data,
        }
        print(json.dumps(data, indent=2))
    else:
        print(result.summary())

    if args.plot:
        result.plot(save_path=args.plot)
        print(f"\nPlot saved to {args.plot}")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the toklens CLI."""
    parser = argparse.ArgumentParser(
        prog="toklens",
        description="TokLens: Multilingual Tokenizer Analysis",
    )
    sub = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze a single tokenizer")
    p_analyze.add_argument("tokenizer", help="HuggingFace model name or path")
    p_analyze.add_argument("--langs", help="Comma-separated language codes (e.g., en,zh,ja)")
    p_analyze.add_argument("--ref", default="en", help="Reference language for parity")
    p_analyze.add_argument("--format", choices=["table", "json", "csv", "latex"], default="table")
    p_analyze.add_argument("--plot", help="Save plot to this path")

    # compare
    p_compare = sub.add_parser("compare", help="Compare two tokenizers")
    p_compare.add_argument("tokenizer_a", help="First tokenizer")
    p_compare.add_argument("tokenizer_b", help="Second tokenizer")
    p_compare.add_argument("--langs", help="Comma-separated language codes")
    p_compare.add_argument("--ref", default="en", help="Reference language for parity")
    p_compare.add_argument("--format", choices=["table", "json"], default="table")
    p_compare.add_argument("--plot", help="Save plot to this path")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
