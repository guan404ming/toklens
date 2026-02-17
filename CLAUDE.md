# TokLens - Claude Code Instructions

## Project Overview

TokLens is a multilingual tokenizer analysis toolkit. Pure Python, no GPU.

## Tech Stack

- Python 3.10+
- Dependencies: tokenizers (HuggingFace), numpy, matplotlib (optional)
- Package manager: uv
- Build: pyproject.toml (src layout)
- Testing: pytest
- Linting: ruff
- Type checking: ty (astral-sh/ty)

## Code Style

- Minimal changes, keep clean and clear
- Reuse, don't rebuild
- No duplication
- Comments: simple and clear
- No em dashes, use commas or periods
- No emojis in code or docs
- Type hints on all public functions
- Docstrings: Google style, brief

## Project Structure

```
src/toklens/
├── analyzer.py      # Load tokenizer, run evaluation
├── metrics.py       # All metric implementations
├── corpora.py       # Built-in multilingual test data
├── compare.py       # Two-tokenizer comparison
├── report.py        # Output: tables, plots, LaTeX
└── cli.py           # Command-line interface
```

## Key Design Decisions

- HuggingFace tokenizers handles all tokenization (already Rust-backed)
- TokLens only computes statistics on token IDs, no need for Rust
- Corpora loaded from FLORES-200 via HF datasets, cached locally
- All metrics return plain dicts or numpy arrays, not custom classes

## Commands

- `uv run pytest` - run tests
- `uv run ruff check .` - lint
- `uv run ruff format .` - format
- `uvx ty check` - type check
