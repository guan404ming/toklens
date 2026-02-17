#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Collect benchmark scores ==="
uv run python -m experiments.01_collect_benchmarks

echo ""
echo "=== Step 2: Compute TokLens metrics ==="
uv run python -m experiments.02_compute_metrics

echo ""
echo "=== Step 3: Correlation analysis ==="
uv run python -m experiments.03_correlation

echo ""
echo "=== Step 4: Generate figures ==="
uv run python -m experiments.04_figures

echo ""
echo "=== Done. All results in experiments/ ==="
