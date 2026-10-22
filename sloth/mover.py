from __future__ import annotations

import json
import shutil
from pathlib import Path

from send2trash import send2trash


def move_to_review(
    file: Path,
    reason: str,
    run_id: str,
    stage: str,
    score: float,
    input_folder: Path | None = None,
    manifest_path: Path | None = None,
) -> Path:
    """Move a file into the review bucket and append a manifest entry."""
    source = file.expanduser().resolve()
    root = (input_folder or source.parent).expanduser().resolve()
    manifest = (manifest_path or root / "manifest.jsonl").expanduser().resolve()

    destination_dir = root / "_sloth_review" / reason
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(destination_dir / source.name)

    shutil.move(str(source), str(destination))
    _append_manifest(
        manifest,
        {
            "run_id": run_id,
            "original": str(source),
            "moved_to": str(destination),
            "stage": stage,
            "reason": reason,
            "score": score,
        },
    )
    return destination


def undo(
    run_id: str, input_folder: Path | None = None, manifest_path: Path | None = None
) -> int:
    """Undo all file moves recorded for a run_id."""
    root = (input_folder or Path.cwd()).expanduser().resolve()
    manifest = (manifest_path or root / "manifest.jsonl").expanduser().resolve()
    if not manifest.exists():
        return 0

    restored = 0
    entries = _read_manifest_entries(manifest)
    for entry in reversed(entries):
        if str(entry.get("run_id", "")) != run_id:
            continue

        original = Path(str(entry.get("original", ""))).expanduser().resolve()
        moved_to = Path(str(entry.get("moved_to", ""))).expanduser().resolve()
        if not moved_to.exists():
            continue
        if original.exists():
            raise FileExistsError(f"Cannot restore file; destination exists: {original}")

        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(moved_to), str(original))
        restored += 1

    return restored


def empty_review(input_folder: Path | None = None) -> int:
    """Send review files to recycle bin after explicit confirmation."""
    root = (input_folder or Path.cwd()).expanduser().resolve()
    review_root = root / "_sloth_review"
    if not review_root.exists():
        print("No review folder found.")
        return 0

    reason_dirs = [p for p in review_root.iterdir() if p.is_dir()]
    if not reason_dirs:
        print("Review folder is empty.")
        return 0

    total = 0
    print("Review counts:")
    for reason_dir in sorted(reason_dirs, key=lambda p: p.name.lower()):
        count = sum(1 for p in reason_dir.rglob("*") if p.is_file())
        total += count
        print(f"  {reason_dir.name}: {count}")

    answer = input('Type "yes" to move these files to Recycle Bin: ').strip().lower()
    if answer != "yes":
        print("Cancelled.")
        return 0

    for reason_dir in reason_dirs:
        send2trash(str(reason_dir))
    print(f"Sent {total} file(s) to Recycle Bin.")
    return total


def _append_manifest(manifest_path: Path, row: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_manifest_entries(manifest_path: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                entries.append(row)
    return entries


def _unique_destination(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    index = 1
    while True:
        candidate = target_path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1
