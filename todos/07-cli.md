# 07 - CLI

## Goal

Add a command-line interface for quick tokenizer analysis without writing Python.

## Tasks

- [ ] `toklens analyze <tokenizer> --langs en,zh,ja` runs evaluation and prints summary
- [ ] `toklens compare <tok_a> <tok_b> --langs en,zh` prints comparison table
- [ ] `--format` flag: table (default), csv, latex, json
- [ ] `--plot` flag: save chart to file
- [ ] Use argparse or typer (minimal dependency)
- [ ] Add CLI entry point in pyproject.toml

## Acceptance Criteria

- `uv run toklens analyze gpt2 --langs en` prints a valid table
- `uv run toklens compare gpt2 meta-llama/Llama-3.1-8B --format json` outputs JSON
