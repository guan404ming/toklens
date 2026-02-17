"""Tests for report module."""

from toklens.report import Report


def _make_report() -> Report:
    return Report("test-tokenizer", {
        "en": {"fertility": 1.2, "cpt": 4.5, "strr": 0.8, "compression_ratio": 4.0, "nsl": 0.15},
        "zh": {"fertility": 2.5, "cpt": 1.8, "strr": 0.3, "compression_ratio": 2.1, "nsl": 0.45},
    })


def test_report_langs():
    report = _make_report()
    assert report.langs == ["en", "zh"]


def test_report_metric_names():
    report = _make_report()
    assert "fertility" in report.metric_names
    assert "strr" in report.metric_names


def test_fertility_accessor():
    report = _make_report()
    f = report.fertility()
    assert f["en"] == 1.2
    assert f["zh"] == 2.5


def test_strr_accessor():
    report = _make_report()
    s = report.strr()
    assert s["en"] == 0.8


def test_summary_string():
    report = _make_report()
    s = report.summary()
    assert "test-tokenizer" in s
    assert "en" in s
    assert "zh" in s


def test_to_dict():
    report = _make_report()
    d = report.to_dict()
    assert d["tokenizer"] == "test-tokenizer"
    assert "en" in d["results"]


def test_to_csv():
    report = _make_report()
    csv_str = report.to_csv()
    assert "lang" in csv_str
    assert "fertility" in csv_str
    assert "en" in csv_str


def test_to_latex():
    report = _make_report()
    latex = report.to_latex()
    assert "\\begin{table}" in latex
    assert "test-tokenizer" in latex
    assert "\\end{table}" in latex


def test_empty_report():
    report = Report("empty", {})
    assert report.summary() == "No results for empty"
    assert report.to_latex() == ""
