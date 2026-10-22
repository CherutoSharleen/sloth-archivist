from __future__ import annotations

import json
from pathlib import Path

import pytest

from sloth.ingest import discover_images


def test_discover_images_skips_review_heic_and_manifest(tmp_path: Path) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()

    kept_png = photos / "keep.png"
    kept_png.write_bytes(b"png")
    in_manifest = photos / "already.jpg"
    in_manifest.write_bytes(b"jpg")
    in_review = photos / "_sloth_review" / "rules" / "moved.jpeg"
    in_review.parent.mkdir(parents=True)
    in_review.write_bytes(b"jpeg")
    heic = photos / "raw.heic"
    heic.write_bytes(b"heic")

    manifest = photos / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"original": str(in_manifest.resolve())}) + "\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="HEIC/HEIF"):
        discovered = discover_images(photos, manifest)
    assert discovered == [kept_png.resolve()]
