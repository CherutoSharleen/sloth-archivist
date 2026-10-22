from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from sloth.config import Settings
from sloth.filters import filter_images


def _write_pattern(path: Path, size: tuple[int, int] = (1024, 768)) -> None:
    image = Image.new("RGB", size, color=(210, 210, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, size[0] - 40, size[1] - 40), outline=(10, 10, 10), width=8)
    draw.ellipse(
        (
            int(size[0] * 0.15),
            int(size[1] * 0.2),
            int(size[0] * 0.45),
            int(size[1] * 0.55),
        ),
        fill=(40, 40, 40),
    )
    draw.rectangle(
        (
            int(size[0] * 0.55),
            int(size[1] * 0.35),
            int(size[0] * 0.9),
            int(size[1] * 0.75),
        ),
        fill=(120, 120, 120),
    )
    draw.line((0, size[1] // 2, size[0], size[1] // 2), fill=(0, 0, 0), width=6)
    image.save(path)


def test_sharp_image_scores_higher_than_blurred_copy(tmp_path: Path) -> None:
    sharp = tmp_path / "sharp.jpg"
    _write_pattern(sharp, (900, 600))

    with Image.open(sharp) as img:
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=4))
        draw = ImageDraw.Draw(blurred_img)
        draw.rectangle((10, 10, 40, 40), fill=(255, 255, 255))
        blurred_img.save(tmp_path / "blurred.jpg")

    blurred = tmp_path / "blurred.jpg"
    settings = Settings(duplicate_hash_distance=-1, blur_threshold=200.0)
    results = {r.path.name: r for r in filter_images([sharp, blurred], settings)}

    assert results["sharp.jpg"].blur_score > results["blurred.jpg"].blur_score
    assert results["blurred.jpg"].reason == "blurry"
    assert results["blurred.jpg"].is_junk is True


def test_darkened_copy_is_marked_dark(tmp_path: Path) -> None:
    dark = tmp_path / "dark.jpg"
    _write_pattern(dark, (900, 600))

    with Image.open(dark) as img:
        darkened = ImageOps.autocontrast(img).point(lambda p: p * 0.08)
        darkened.save(dark)

    settings = Settings(dark_threshold=40.0, bright_threshold=215.0, blur_threshold=1.0)
    result = filter_images([dark], settings)[0]

    assert result.reason == "dark"
    assert result.is_junk is True
    assert result.brightness_score < settings.dark_threshold


def test_resized_duplicate_marks_only_less_sharp_as_duplicate(tmp_path: Path) -> None:
    original = tmp_path / "original.jpg"
    _write_pattern(original, (1400, 900))

    resized_copy = tmp_path / "copy_resized.jpg"
    with Image.open(original) as img:
        img.resize((900, 580), Image.Resampling.LANCZOS).save(resized_copy)

    screenshot_png = tmp_path / "Screenshot 2026-10-22 203000.png"
    Image.new("RGB", (1366, 768), color=(255, 255, 255)).save(screenshot_png)

    settings = Settings(duplicate_hash_distance=8, blur_threshold=1.0)
    results = {r.path.name: r for r in filter_images([original, resized_copy, screenshot_png], settings)}

    duplicate_entries = [results["original.jpg"], results["copy_resized.jpg"]]
    duplicate_reasons = [entry.reason for entry in duplicate_entries]
    assert duplicate_reasons.count("duplicate") == 1

    dup = next(entry for entry in duplicate_entries if entry.reason == "duplicate")
    keeper = next(entry for entry in duplicate_entries if entry.reason is None)
    assert dup.is_junk is True
    assert keeper.is_junk is False
    assert dup.duplicate_distance is not None
    assert results["Screenshot 2026-10-22 203000.png"].is_screenshot is True
    assert results["Screenshot 2026-10-22 203000.png"].is_junk is False
