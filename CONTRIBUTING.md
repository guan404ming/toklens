# Contributing to TokLens

## Dev Setup

```bash
git clone https://github.com/guan404ming/toklens.git
cd toklens
uv venv
uv sync --all-extras
```

## Running Checks

```bash
uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
uvx ty check               # type check
```

## Code Style

- Follow existing patterns in the codebase
- Type hints on all public functions
- Google-style docstrings, keep them brief
- No emojis in code or docs
- Run all checks before submitting a PR

## Adding a New Metric

1. Add the function to `src/toklens/metrics.py`
2. Add it to `compute_all()` if it applies to single-tokenizer evaluation
3. Add a test in `tests/test_metrics.py`
4. Document the metric in the README metrics table

## Adding a New Corpus/Dataset

1. Add a loader function to `src/toklens/corpora.py`
2. Update `get_texts()` to support the new dataset
3. Document the dataset in the README

## Pull Requests

- One feature per PR
- Include tests for new functionality
- All checks must pass
