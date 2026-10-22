from pathlib import Path

from sloth.cli import build_parser, main
from sloth.filters import FilterResult


def test_cli_includes_expected_commands() -> None:
    parser = build_parser()
    action = next(a for a in parser._actions if a.dest == "command")
    commands = set(action.choices.keys())
    assert commands == {"run", "undo", "report", "empty-review", "eval"}


def test_run_dry_run_lists_expected_reasons(monkeypatch, capsys) -> None:
    fake_files = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]

    def fake_discover(_input_folder, manifest_path=None):  # noqa: ANN001, ARG001
        return fake_files

    def fake_filter(_files, _settings):  # noqa: ANN001, ARG001
        return [
            FilterResult(
                path=Path("blurry.jpg"),
                is_junk=True,
                reason="blurry",
                score=12.3,
                blur_score=12.3,
                brightness_score=120.0,
                duplicate_distance=None,
                is_screenshot=False,
            ),
            FilterResult(
                path=Path("dark.jpg"),
                is_junk=True,
                reason="dark",
                score=20.0,
                blur_score=150.0,
                brightness_score=20.0,
                duplicate_distance=None,
                is_screenshot=False,
            ),
            FilterResult(
                path=Path("dup.jpg"),
                is_junk=True,
                reason="duplicate",
                score=3.0,
                blur_score=110.0,
                brightness_score=110.0,
                duplicate_distance=3,
                is_screenshot=False,
            ),
        ]

    monkeypatch.setattr("sloth.cli.discover_images", fake_discover)
    monkeypatch.setattr("sloth.cli.filter_images", fake_filter)

    exit_code = main(["run", "--dry-run", "--input-folder", "sample_photos"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "blurry:" in out
    assert "dark:" in out
    assert "duplicate:" in out
    assert "blurry.jpg" in out
    assert "dark.jpg" in out
    assert "dup.jpg" in out
