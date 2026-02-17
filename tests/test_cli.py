"""Tests for CLI module."""

from toklens.cli import main


def test_cli_no_args(capsys):
    """CLI with no args prints help and exits."""
    try:
        main([])
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert "toklens" in captured.out or "usage" in captured.out.lower()
