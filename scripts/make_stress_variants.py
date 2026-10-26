from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SEED = 42
RNG = random.Random(SEED)


@dataclass(frozen=True)
class LabelRow:
    file: str
    label: str
    expected_reason: str
    note: str


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    originals_dir = root / "stress_assets" / "real" / "originals"
    set_dir = root / "stress_assets" / "real" / "set"
    labels_path = root / "eval" / "labels_real.csv"

    originals = sorted(originals_dir.glob("keep_*.jpg"))
    if len(originals) < 20:
        raise RuntimeError("Need downloaded originals first. Run scripts/fetch_stress_assets.py.")

    if set_dir.exists():
        shutil.rmtree(set_dir)
    set_dir.mkdir(parents=True, exist_ok=True)

    timeline = _build_timeline(len(originals) + 55)
    gps_pool = [
        (38.7114, -9.1291, "Lisbon", "PT"),
        (40.7580, -73.9855, "New York", "US"),
        (35.6762, 139.6503, "Tokyo", "JP"),
    ]

    labels: list[LabelRow] = []
    index = 0

    # Copy originals into flat set with EXIF datetime and sidecar metadata.
    original_datetimes: dict[str, datetime] = {}
    original_gps: dict[str, tuple[float, float] | None] = {}
    for original in originals:
        out_name = original.name
        out_path = set_dir / out_name
        dt = timeline[index]
        index += 1
        with Image.open(original) as image:
            _save_with_exif_datetime(image.convert("RGB"), out_path, dt)
        gps = gps_pool[index % len(gps_pool)] if index % 2 == 0 else None
        original_datetimes[out_name] = dt
        original_gps[out_name] = (gps[0], gps[1]) if gps else None
        _write_sidecar(out_path, dt, original_gps[out_name])
        labels.append(LabelRow(out_name, "keep", "keep", "wikimedia original"))

    source_names = [p.name for p in originals]
    used_sources: set[str] = set()

    def pick_source(prefer_unique: bool = True) -> Path:
        nonlocal source_names, used_sources
        candidates = [name for name in source_names if name not in used_sources] if prefer_unique else source_names
        if not candidates:
            candidates = source_names
        chosen = RNG.choice(candidates)
        used_sources.add(chosen)
        return set_dir / chosen

    # 12 near-duplicates including three burst sets of 3.
    dup_count = 0
    burst_sources = [pick_source(True), pick_source(True), pick_source(True)]
    for burst_idx, source in enumerate(burst_sources, start=1):
        base_dt = original_datetimes[source.name]
        for frame in range(1, 4):
            dup_count += 1
            out_name = f"variant_duplicate_burst{burst_idx}_{frame:02d}.jpg"
            out_path = set_dir / out_name
            _make_duplicate(source, out_path, jitter=frame)
            dt = base_dt + timedelta(seconds=frame)
            _rewrite_exif_datetime(out_path, dt)
            _write_sidecar(out_path, dt, original_gps[source.name])
            labels.append(LabelRow(out_name, "junk", "duplicate", f"burst set {burst_idx}"))

    while dup_count < 12:
        source = pick_source(True)
        out_name = f"variant_duplicate_{dup_count+1:02d}.jpg"
        out_path = set_dir / out_name
        _make_duplicate(source, out_path, jitter=dup_count + 1)
        dt = original_datetimes[source.name] + timedelta(seconds=RNG.randint(-4, 4))
        _rewrite_exif_datetime(out_path, dt)
        _write_sidecar(out_path, dt, original_gps[source.name])
        labels.append(LabelRow(out_name, "junk", "duplicate", "near duplicate"))
        dup_count += 1

    for idx_blur in range(1, 11):
        source = pick_source(True)
        out_name = f"variant_blurry_{idx_blur:02d}.jpg"
        out_path = set_dir / out_name
        _make_blurry(source, out_path)
        dt = original_datetimes[source.name] + timedelta(seconds=RNG.randint(-6, 6))
        _rewrite_exif_datetime(out_path, dt)
        _write_sidecar(out_path, dt, original_gps[source.name])
        labels.append(LabelRow(out_name, "junk", "blurry", "gaussian blur"))

    for idx_dark in range(1, 9):
        source = pick_source(True)
        out_name = f"variant_dark_{idx_dark:02d}.jpg"
        out_path = set_dir / out_name
        _make_dark(source, out_path)
        dt = original_datetimes[source.name] + timedelta(seconds=RNG.randint(-6, 6))
        _rewrite_exif_datetime(out_path, dt)
        _write_sidecar(out_path, dt, original_gps[source.name])
        labels.append(LabelRow(out_name, "junk", "dark", "underexposed + noise"))

    for idx_bright in range(1, 7):
        source = pick_source(True)
        out_name = f"variant_overexposed_{idx_bright:02d}.jpg"
        out_path = set_dir / out_name
        _make_overexposed(source, out_path)
        dt = original_datetimes[source.name] + timedelta(seconds=RNG.randint(-6, 6))
        _rewrite_exif_datetime(out_path, dt)
        _write_sidecar(out_path, dt, original_gps[source.name])
        labels.append(LabelRow(out_name, "junk", "overexposed", "overexposed scene"))

    for idx_pocket in range(1, 5):
        out_name = f"variant_accidental_{idx_pocket:02d}.jpg"
        out_path = set_dir / out_name
        _make_pocket(out_path)
        dt = timeline[index]
        index += 1
        _rewrite_exif_datetime(out_path, dt)
        _write_sidecar(out_path, dt, None)
        labels.append(LabelRow(out_name, "junk", "accidental", "pocket shot"))

    throwaway_kinds = [
        "otp_code",
        "error_resolved",
        "loading_spinner",
        "download_toast",
        "meme_fake_text",
        "auth_token",
        "timeout_error",
        "verification_code",
        "network_retry",
        "session_expired",
    ]
    for idx_shot, kind in enumerate(throwaway_kinds, start=1):
        out_name = f"Screenshot_{kind}_{idx_shot:02d}.png"
        out_path = set_dir / out_name
        _draw_screenshot(out_path, kind=kind, keepworthy=False)
        dt = timeline[index]
        index += 1
        _write_sidecar(out_path, dt, None)
        labels.append(LabelRow(out_name, "junk", "screenshot_throwaway", "throwaway screenshot"))

    keepworthy_kinds = [
        "boarding_pass",
        "train_ticket_qr",
        "delivery_address",
        "booking_reference",
        "saved_map_pin",
    ]
    for idx_shot, kind in enumerate(keepworthy_kinds, start=1):
        out_name = f"Screenshot_{kind}_{idx_shot:02d}.png"
        out_path = set_dir / out_name
        _draw_screenshot(out_path, kind=kind, keepworthy=True)
        dt = timeline[index]
        index += 1
        gps = gps_pool[idx_shot % len(gps_pool)]
        _write_sidecar(out_path, dt, (gps[0], gps[1]))
        labels.append(LabelRow(out_name, "keep", "keep", "keep-worthy screenshot"))

    _write_labels(labels_path, labels)

    print(f"Wrote {len(labels)} labeled files to {set_dir}")
    print(f"Labels: {labels_path}")
    return 0


def _build_timeline(total: int) -> list[datetime]:
    base = datetime(2026, 8, 3, 8, 0, 0, tzinfo=timezone.utc)
    points: list[datetime] = []
    for day in range(7):
        clusters = RNG.randint(2, 4)
        anchors = sorted(RNG.sample([8, 10, 13, 16, 19, 21], k=clusters))
        for hour in anchors:
            shots = RNG.randint(3, 6)
            for _ in range(shots):
                minute = RNG.randint(0, 59)
                second = RNG.randint(0, 59)
                points.append(base + timedelta(days=day, hours=hour - 8, minutes=minute, seconds=second))
    points.sort()
    if len(points) < total:
        while len(points) < total:
            points.append(points[-1] + timedelta(minutes=5))
    return points[:total]


def _save_with_exif_datetime(image: Image.Image, out_path: Path, dt: datetime) -> None:
    exif = Image.Exif()
    stamp = dt.strftime("%Y:%m:%d %H:%M:%S")
    exif[36867] = stamp
    exif[306] = stamp
    image.save(out_path, format="JPEG", quality=92, exif=exif)


def _rewrite_exif_datetime(path: Path, dt: datetime) -> None:
    with Image.open(path) as image:
        exif = image.getexif()
        stamp = dt.strftime("%Y:%m:%d %H:%M:%S")
        exif[36867] = stamp
        exif[306] = stamp
        image.convert("RGB").save(path, format="JPEG", quality=92, exif=exif)


def _make_duplicate(source: Path, out_path: Path, jitter: int) -> None:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        w, h = rgb.size
        crop_ratio = RNG.uniform(0.02, 0.03)
        dx = int(w * crop_ratio)
        dy = int(h * crop_ratio)
        box = (dx, dy, w - dx, h - dy)
        cropped = rgb.crop(box).resize((w, h), Image.Resampling.LANCZOS)
        bright = ImageEnhance.Brightness(cropped).enhance(RNG.uniform(1.03, 1.07))
        if jitter % 3 == 0:
            bright = bright.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80))
        elif jitter % 2 == 0:
            bright = bright.filter(ImageFilter.GaussianBlur(radius=0.8))
        _save_with_exif_datetime(bright, out_path, datetime.now(timezone.utc))


def _make_blurry(source: Path, out_path: Path) -> None:
    with Image.open(source) as image:
        blur = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=RNG.uniform(6, 10)))
        _save_with_exif_datetime(blur, out_path, datetime.now(timezone.utc))


def _make_dark(source: Path, out_path: Path) -> None:
    with Image.open(source) as image:
        img = image.convert("RGB")
        dark = ImageEnhance.Brightness(img).enhance(RNG.uniform(0.08, 0.15))
        arr = np.asarray(dark, dtype=np.int16)
        noise = RNG.randint(8, 20)
        noisy = np.clip(arr + np.random.default_rng(SEED + arr.shape[0]).integers(-noise, noise + 1, arr.shape), 0, 255).astype(np.uint8)
        out = Image.fromarray(noisy, mode="RGB")
        _save_with_exif_datetime(out, out_path, datetime.now(timezone.utc))


def _make_overexposed(source: Path, out_path: Path) -> None:
    with Image.open(source) as image:
        img = image.convert("RGB")
        blown = ImageEnhance.Brightness(img).enhance(RNG.uniform(2.4, 3.0))
        _save_with_exif_datetime(blown, out_path, datetime.now(timezone.utc))


def _make_pocket(out_path: Path) -> None:
    width = 1280
    height = 960
    noise = np.random.default_rng(SEED + 900 + RNG.randint(1, 500)).integers(0, 35, (height, width, 3), dtype=np.uint8)
    base = Image.fromarray(noise, mode="RGB")
    dark = ImageEnhance.Brightness(base).enhance(0.05)
    blur = dark.filter(ImageFilter.GaussianBlur(radius=12))
    vignette = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(vignette)
    draw.ellipse((-200, -150, width + 200, height + 150), fill=130)
    final = ImageChops.multiply(blur, Image.merge("RGB", (vignette, vignette, vignette)))
    _save_with_exif_datetime(final, out_path, datetime.now(timezone.utc))


def _draw_screenshot(path: Path, kind: str, keepworthy: bool) -> None:
    w, h = 1170, 2532
    canvas = Image.new("RGB", (w, h), color=(247, 249, 252))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, w, 130), fill=(15, 23, 42))
    draw.text((40, 50), "9:41", fill=(255, 255, 255), font=font)
    draw.text((w - 250, 50), "5G 100%", fill=(255, 255, 255), font=font)

    card_top = 220
    draw.rounded_rectangle((60, card_top, w - 60, h - 220), radius=36, fill=(255, 255, 255), outline=(210, 215, 225), width=3)
    draw.text((100, card_top + 50), kind.replace("_", " ").title(), fill=(20, 28, 40), font=font)

    lines = _screenshot_lines(kind, keepworthy)
    y = card_top + 130
    for line in lines:
        draw.text((100, y), line, fill=(46, 52, 64), font=font)
        y += 54

    if "qr" in kind:
        qr_left = w - 390
        qr_top = card_top + 360
        draw.rectangle((qr_left, qr_top, qr_left + 250, qr_top + 250), fill=(0, 0, 0))
        draw.rectangle((qr_left + 30, qr_top + 30, qr_left + 220, qr_top + 220), fill=(255, 255, 255))
        for ix in range(10):
            for iy in range(10):
                if (ix * iy + ix + iy) % 3 == 0:
                    draw.rectangle((qr_left + 40 + ix * 18, qr_top + 40 + iy * 18, qr_left + 52 + ix * 18, qr_top + 52 + iy * 18), fill=(0, 0, 0))

    if "loading" in kind:
        cx = w // 2
        cy = h // 2
        for idx in range(12):
            alpha = int(255 * (idx + 1) / 12)
            draw.ellipse((cx - 20 - idx * 3, cy - 20 - idx * 3, cx + 20 + idx * 3, cy + 20 + idx * 3), outline=(80, 100, 140, alpha))

    canvas.save(path, format="PNG")


def _screenshot_lines(kind: str, keepworthy: bool) -> list[str]:
    if keepworthy:
        mapping = {
            "boarding_pass": ["Passenger: Alex Morgan", "Flight: SA 142", "Seat: 14A", "Boarding: 18:25"],
            "train_ticket_qr": ["Route: Lisbon -> Porto", "Coach: C", "Seat: 23", "Ref: TR-48219"],
            "delivery_address": ["Order Confirmed", "221 Harbor Street", "Apt 4B", "ETA: Tomorrow 14:00"],
            "booking_reference": ["Hotel Booking", "Reference: BK-739205", "Check-in: 12 Aug", "Nights: 3"],
            "saved_map_pin": ["Saved place: Riverside Cafe", "Pin: 38.7114,-9.1291", "Note: meet at noon"],
        }
        return mapping.get(kind, ["Saved record", "Reference available"]) 

    mapping = {
        "otp_code": ["Your one-time code is 834219", "Expires in 2 minutes", "Do not share this code."],
        "error_resolved": ["Upload failed", "Tap to retry", "Issue resolved"],
        "loading_spinner": ["Loading your feed...", "Please wait"],
        "download_toast": ["Download complete", "tap to open"],
        "meme_fake_text": ["When the model returns JSON first try", "everyone in the sprint", "(totally fake meme text)"],
        "auth_token": ["Security code: 445971", "Use once and discard"],
        "timeout_error": ["Request timed out", "Try again later", "Resolved"],
        "verification_code": ["Verification: 771095", "valid for 60 seconds"],
        "network_retry": ["No connection", "Retrying in 10 seconds"],
        "session_expired": ["Session expired", "Please sign in again"],
    }
    return mapping.get(kind, ["Temporary screenshot", "throwaway info"])


def _write_sidecar(path: Path, dt: datetime, gps: tuple[float, float] | None) -> None:
    payload: dict[str, object] = {
        "photoTakenTime": {"timestamp": str(int(dt.timestamp()))},
        "geoData": {},
    }
    if gps is not None:
        payload["geoData"] = {"latitude": gps[0], "longitude": gps[1]}
    sidecar = path.with_name(f"{path.name}.json")
    sidecar.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _write_labels(path: Path, rows: list[LabelRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "label", "expected_reason", "note"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file": row.file,
                    "label": row.label,
                    "expected_reason": row.expected_reason,
                    "note": row.note,
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
