from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import reverse_geocoder as rg
from PIL import Image, ExifTags


@dataclass(frozen=True)
class PhotoInfo:
    path: Path
    taken_at: datetime
    latitude: float | None
    longitude: float | None
    place: str | None


def read_photo_info(photo_path: Path) -> PhotoInfo:
    path = photo_path.expanduser().resolve()
    exif = _read_exif(path)
    sidecar = _read_takeout_sidecar(path)

    taken_at = exif["taken_at"] or sidecar["taken_at"] or _file_mtime(path)
    latitude = exif["latitude"] if exif["latitude"] is not None else sidecar["latitude"]
    longitude = exif["longitude"] if exif["longitude"] is not None else sidecar["longitude"]
    place = _lookup_place(latitude, longitude)

    return PhotoInfo(
        path=path,
        taken_at=taken_at,
        latitude=latitude,
        longitude=longitude,
        place=place,
    )


def _read_exif(path: Path) -> dict[str, Any]:
    taken_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None

    with Image.open(path) as image:
        exif = image.getexif()
        if exif:
            dt = exif.get(36867)  # DateTimeOriginal
            if isinstance(dt, str):
                taken_at = _parse_exif_datetime(dt)

            gps = exif.get(34853)  # GPSInfo
            if isinstance(gps, dict):
                mapped = {
                    ExifTags.GPSTAGS.get(tag_id, tag_id): value
                    for tag_id, value in gps.items()
                }
                latitude = _gps_to_decimal(
                    mapped.get("GPSLatitude"), mapped.get("GPSLatitudeRef")
                )
                longitude = _gps_to_decimal(
                    mapped.get("GPSLongitude"), mapped.get("GPSLongitudeRef")
                )

    return {"taken_at": taken_at, "latitude": latitude, "longitude": longitude}


def _read_takeout_sidecar(path: Path) -> dict[str, Any]:
    sidecar_path = path.with_name(f"{path.name}.json")
    if not sidecar_path.exists():
        return {"taken_at": None, "latitude": None, "longitude": None}

    with sidecar_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return {"taken_at": None, "latitude": None, "longitude": None}

    taken_at = _parse_sidecar_timestamp(data.get("photoTakenTime"))

    geo = data.get("geoData")
    if not isinstance(geo, dict):
        geo = data.get("geoDataExif")
    latitude = _as_float(geo.get("latitude")) if isinstance(geo, dict) else None
    longitude = _as_float(geo.get("longitude")) if isinstance(geo, dict) else None

    return {"taken_at": taken_at, "latitude": latitude, "longitude": longitude}


def _lookup_place(latitude: float | None, longitude: float | None) -> str | None:
    if latitude is None or longitude is None:
        return None
    result = rg.search((latitude, longitude), mode=1)
    if not result:
        return None
    city = result[0].get("name")
    country = result[0].get("cc")
    if not city or not country:
        return None
    return f"{city}, {country}"


def _parse_exif_datetime(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_sidecar_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, dict):
        return None
    timestamp = value.get("timestamp")
    if timestamp is None:
        return None
    try:
        seconds = int(str(timestamp))
    except ValueError:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def _gps_to_decimal(dms: Any, ref: Any) -> float | None:
    if not isinstance(dms, (list, tuple)) or len(dms) != 3:
        return None
    degrees = _rational_to_float(dms[0])
    minutes = _rational_to_float(dms[1])
    seconds = _rational_to_float(dms[2])
    if degrees is None or minutes is None or seconds is None:
        return None

    value = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ("S", "W"):
        value *= -1
    return value


def _rational_to_float(value: Any) -> float | None:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            numerator, denominator = value
            return float(numerator) / float(denominator)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _file_mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
