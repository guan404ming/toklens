# 06 - Report and Visualization

## Goal

Implement output formats: console tables, plots, and LaTeX export.

## Tasks

- [ ] `Report.summary()` prints a formatted console table
- [ ] `Report.plot()` generates matplotlib bar/heatmap charts
- [ ] `Report.to_latex()` outputs a LaTeX table ready for papers
- [ ] `Report.to_dict()` returns raw data as nested dict
- [ ] `Report.to_csv(path)` exports to CSV
- [ ] Heatmap: languages x metrics for a single tokenizer
- [ ] Bar chart: metric comparison across tokenizers
- [ ] Unit tests for each output format

## Acceptance Criteria

- All output methods produce valid, well-formatted output
- Plots render without errors when matplotlib is installed
- LaTeX output compiles cleanly
