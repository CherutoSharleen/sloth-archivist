from __future__ import annotations

import json
import warnings
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
HEIC_SUFFIXES = {".heic", ".heif"}


def discover_images(input_folder: Path, manifest_path: Path | None = None) -> list[Path]:
    """Find supported images to process for a run."""
    folder = input_folder.expanduser().resolve()
    manifest = manifest_path or folder / "manifest.jsonl"
    seen = _load_manifest_paths(manifest, folder)

    candidates: list[Path] = []
    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue
        if "_sloth_review" in file_path.parts:
            continue

        suffix = file_path.suffix.lower()
        if suffix in HEIC_SUFFIXES:
            warnings.warn(
                f"Skipping HEIC/HEIF file (unsupported): {file_path}",
                UserWarning,
                stacklevel=2,
            )
            continue
        if suffix not in IMAGE_SUFFIXES:
            continue

        resolved = file_path.resolve()
        if resolved in seen:
            continue
        candidates.append(resolved)

    return sorted(candidates)


def _load_manifest_paths(manifest_path: Path, base_folder: Path) -> set[Path]:
    if not manifest_path.exists():
        return set()

    seen: set[Path] = set()
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(row, dict):
                continue
            for key in ("original", "moved_to"):
                value = row.get(key)
                if isinstance(value, str) and value:
                    seen.add(_normalize_path(value, base_folder))
    return seen


def _normalize_path(value: str, base_folder: Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    return (base_folder / raw).resolve()
