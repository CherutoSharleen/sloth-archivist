from __future__ import annotations

import argparse
from pathlib import Path

from sloth.config import load_settings
from sloth.ingest import discover_images
from sloth.metadata import read_photo_info


def main() -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="Print date and place for each discoverable photo."
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=str(settings.input_folder),
        help="Folder to scan for photos.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest.jsonl path used to skip already processed files.",
    )
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    manifest_path = Path(args.manifest) if args.manifest else None
    photos = discover_images(input_folder, manifest_path=manifest_path)

    if not photos:
        print("No supported photos found.")
        return 0

    for photo in photos:
        info = read_photo_info(photo)
        place = info.place or "Unknown place"
        date_text = info.taken_at.isoformat()
        print(f"{date_text} | {place} | {photo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
