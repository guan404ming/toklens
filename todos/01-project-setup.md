# 01 - Project Setup

## Goal

Initialize the Python package with pyproject.toml, src layout, and dev tooling.

## Tasks

- [ ] Create `pyproject.toml` with uv, ruff, pytest config
- [ ] Create `src/toklens/__init__.py`
- [ ] Add dev dependencies: ruff, pytest, pytest-cov
- [ ] Add runtime dependencies: tokenizers, numpy
- [ ] Add optional dependencies: matplotlib
- [ ] Verify `uv run pytest` and `uv run ruff check .` work
- [ ] Add `.gitignore` for Python

## Acceptance Criteria

- `uv sync` installs all deps
- `uv run python -c "import toklens"` works
- Linting and testing commands run without errors
