# 08 - PyPI Publish

## Goal

Publish toklens to PyPI so users can `pip install toklens`.

## Tasks

- [ ] Finalize pyproject.toml metadata (author, description, classifiers, URLs)
- [ ] Add py.typed marker for type checking support
- [ ] Write a concise PyPI description (README serves as long description)
- [ ] Test build with `uv build`
- [ ] Test install from wheel in a clean venv
- [ ] Publish to TestPyPI first
- [ ] Publish to PyPI

## Acceptance Criteria

- `pip install toklens` works in a fresh environment
- `from toklens import Analyzer` works after install
- PyPI page shows correct metadata and description
