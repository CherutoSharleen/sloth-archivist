from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from sloth.cli import main


def test_run_then_undo_restores_files_byte_for_byte(tmp_path: Path) -> None:
    work = tmp_path / "sample_photos"
    work.mkdir(parents=True, exist_ok=True)
    _seed_test_photos(work)

    baseline = _snapshot_files(work)
    assert baseline

    run_code = main(["run", "--input-folder", str(work)])
    assert run_code == 0

    manifest = work / "manifest.jsonl"
    assert manifest.exists()
    entries = _read_manifest(manifest)
    assert entries

    run_id = str(entries[-1]["run_id"])
    run_entries = [row for row in entries if str(row.get("run_id")) == run_id]
    assert run_entries

    undo_code = main(["undo", "--run", run_id, "--input-folder", str(work)])
    assert undo_code == 0

    for rel_path, content in baseline.items():
        current = work / rel_path
        assert current.exists(), f"Missing file after undo: {rel_path}"
        assert current.read_bytes() == content, f"Content changed for {rel_path}"


def _snapshot_files(folder: Path) -> dict[Path, bytes]:
    snapshot: dict[Path, bytes] = {}
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        snapshot[path.relative_to(folder)] = path.read_bytes()
    return snapshot


def _read_manifest(manifest_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _seed_test_photos(folder: Path) -> None:
    base = Image.new("RGB", (1200, 800), color=(220, 220, 220))
    draw = ImageDraw.Draw(base)
    draw.rectangle((50, 50, 1150, 750), outline=(0, 0, 0), width=8)
    draw.line((0, 400, 1200, 400), fill=(30, 30, 30), width=7)
    draw.ellipse((200, 200, 520, 520), fill=(80, 80, 80))

    original = folder / "original.jpg"
    base.save(original)
    base.resize((950, 620), Image.Resampling.LANCZOS).save(folder / "dup.jpg")
    Image.new("RGB", (600, 400), color=(160, 160, 160)).save(folder / "keep.jpg")
