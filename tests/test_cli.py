from pathlib import Path
import json

from sloth.cli import build_parser, main
from sloth.config import Settings
from sloth.filters import FilterResult
from sloth.review import DecisionRecord


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
    monkeypatch.setattr("sloth.cli.load_review_prompt", lambda _v: "prompt")
    monkeypatch.setattr(
        "sloth.cli.review_survivors",
        lambda **kwargs: [  # noqa: ARG005
            DecisionRecord(
                run_id="run",
                file="c.jpg",
                stage="gemma",
                verdict="keep",
                reason="useful",
                caption="caption",
                model="gemma3:4b",
                prompt_version="review_v1",
                seconds=1.0,
            )
        ],
    )

    exit_code = main(["run", "--dry-run", "--input-folder", "sample_photos"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "3 photos" in out
    assert "3 junk by rules" in out


def test_run_sends_only_rule_survivors_to_gemma_and_logs_decisions(
    tmp_path: Path, monkeypatch
) -> None:
    p1 = tmp_path / "rule_junk.jpg"
    p2 = tmp_path / "survivor_keep.jpg"
    p3 = tmp_path / "survivor_junk.jpg"
    for path in (p1, p2, p3):
        path.write_bytes(b"img")

    def fake_discover(_input_folder, manifest_path=None):  # noqa: ANN001, ARG001
        return [p1, p2, p3]

    def fake_filter(_files, _settings):  # noqa: ANN001, ARG001
        return [
            FilterResult(
                path=p1,
                is_junk=True,
                reason="blurry",
                score=1.2,
                blur_score=1.2,
                brightness_score=130.0,
                duplicate_distance=None,
                is_screenshot=False,
            ),
            FilterResult(
                path=p2,
                is_junk=False,
                reason=None,
                score=0.0,
                blur_score=200.0,
                brightness_score=120.0,
                duplicate_distance=None,
                is_screenshot=False,
            ),
            FilterResult(
                path=p3,
                is_junk=False,
                reason=None,
                score=0.0,
                blur_score=200.0,
                brightness_score=120.0,
                duplicate_distance=None,
                is_screenshot=False,
            ),
        ]

    seen: dict[str, object] = {}

    def fake_review_survivors(  # noqa: PLR0913
        photo_paths,
        model_name,
        image_size,
        prompt_text,
        run_id,
        prompt_version,
        gemma_client=None,
    ):
        seen["photo_paths"] = list(photo_paths)
        seen["model_name"] = model_name
        seen["prompt_version"] = prompt_version
        return [
            DecisionRecord(
                run_id=run_id,
                file=str(p2),
                stage="gemma",
                verdict="keep",
                reason="useful",
                caption="person at table",
                model=model_name,
                prompt_version=prompt_version,
                seconds=0.15,
            ),
            DecisionRecord(
                run_id=run_id,
                file=str(p3),
                stage="gemma",
                verdict="junk",
                reason="throwaway_screenshot",
                caption="otp code",
                model=model_name,
                prompt_version=prompt_version,
                seconds=0.12,
            ),
        ]

    monkeypatch.setattr("sloth.cli.discover_images", fake_discover)
    monkeypatch.setattr("sloth.cli.filter_images", fake_filter)
    monkeypatch.setattr("sloth.cli.review_survivors", fake_review_survivors)
    monkeypatch.setattr("sloth.cli.load_review_prompt", lambda _v: "prompt")
    monkeypatch.setattr("sloth.cli.build_journal_from_run", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("sloth.cli._build_run_id", lambda: "run-test-1")
    monkeypatch.setattr(
        "sloth.cli.load_settings",
        lambda: Settings(input_folder=tmp_path, model_name="gemma3:4b"),
    )

    exit_code = main(["run", "--input-folder", str(tmp_path)])
    assert exit_code == 0
    assert seen["photo_paths"] == [p2, p3]

    decisions_path = tmp_path / "decisions.jsonl"
    rows = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert all("model" in row and "prompt_version" in row and "seconds" in row for row in rows)
    assert {row["stage"] for row in rows} == {"rules", "gemma"}

    manifest = tmp_path / "manifest.jsonl"
    manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [row["stage"] for row in manifest_rows] == ["rules", "gemma"]
