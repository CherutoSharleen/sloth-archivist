from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from sloth.metadata import read_photo_info


def _write_image(path: Path) -> None:
    image = Image.new("RGB", (10, 10), color="red")
    image.save(path)


def test_read_photo_info_uses_file_mtime_when_exif_missing(tmp_path: Path) -> None:
    photo = tmp_path / "no_exif.jpg"
    _write_image(photo)

    ts = 1_700_000_000
    import os

    os.utime(photo, (ts, ts))

    info = read_photo_info(photo)
    assert int(info.taken_at.timestamp()) == ts
    assert info.latitude is None
    assert info.longitude is None
    assert info.place is None


def test_read_photo_info_uses_sidecar_when_exif_missing(
    tmp_path: Path, monkeypatch
) -> None:
    photo = tmp_path / "from_sidecar.png"
    _write_image(photo)
    sidecar = tmp_path / "from_sidecar.png.json"
    sidecar.write_text(
        """
{
  "photoTakenTime": {"timestamp": "1696118400"},
  "geoData": {"latitude": 40.7128, "longitude": -74.0060}
}
""".strip(),
        encoding="utf-8",
    )

    def fake_search(_coords, mode=1):  # noqa: ANN001, ARG001
        return [{"name": "New York", "cc": "US"}]

    monkeypatch.setattr("sloth.metadata.rg.search", fake_search)

    info = read_photo_info(photo)
    assert info.taken_at == datetime.fromtimestamp(1_696_118_400, tz=timezone.utc)
    assert info.latitude == 40.7128
    assert info.longitude == -74.006
    assert info.place == "New York, US"
