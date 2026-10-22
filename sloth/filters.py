from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from .config import Settings


@dataclass(frozen=True)
class FilterResult:
    path: Path
    is_junk: bool
    reason: str | None
    score: float
    blur_score: float
    brightness_score: float
    duplicate_distance: int | None
    is_screenshot: bool


@dataclass(frozen=True)
class _ImageMetrics:
    path: Path
    hash_value: imagehash.ImageHash
    blur_score: float
    brightness_score: float
    width: int
    height: int
    is_screenshot: bool


def filter_images(photo_paths: list[Path], settings: Settings) -> list[FilterResult]:
    """Apply duplicate/quality heuristics and return per-photo filter decisions."""
    metrics = [_collect_metrics(path) for path in photo_paths]
    duplicate_distances = _duplicate_members(metrics, settings.duplicate_hash_distance)

    results: list[FilterResult] = []
    for metric in sorted(metrics, key=lambda m: str(m.path)):
        duplicate_distance = duplicate_distances.get(metric.path)
        reason: str | None = None
        score = 0.0

        if metric.is_screenshot:
            reason = None
            score = 0.0
        elif duplicate_distance is not None:
            reason = "duplicate"
            score = float(duplicate_distance)
        elif metric.blur_score < settings.blur_threshold:
            reason = "blurry"
            score = metric.blur_score
        elif metric.brightness_score < settings.dark_threshold:
            reason = "dark"
            score = metric.brightness_score
        elif metric.brightness_score > settings.bright_threshold:
            reason = "overexposed"
            score = metric.brightness_score

        results.append(
            FilterResult(
                path=metric.path,
                is_junk=reason in {"duplicate", "blurry", "dark", "overexposed"},
                reason=reason,
                score=score,
                blur_score=metric.blur_score,
                brightness_score=metric.brightness_score,
                duplicate_distance=duplicate_distance,
                is_screenshot=metric.is_screenshot,
            )
        )
    return results


def _collect_metrics(path: Path) -> _ImageMetrics:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        gray_512 = _grayscale_resized(rgb, 512)
        phash = imagehash.phash(rgb)
        blur = laplacian_variance(gray_512)
        brightness = float(np.mean(gray_512))
        screenshot = _looks_like_screenshot(path, width, height)

    return _ImageMetrics(
        path=path,
        hash_value=phash,
        blur_score=blur,
        brightness_score=brightness,
        width=width,
        height=height,
        is_screenshot=screenshot,
    )


def _grayscale_resized(image: Image.Image, size: int) -> np.ndarray:
    resized = image.resize((size, size), Image.Resampling.LANCZOS).convert("L")
    return np.asarray(resized, dtype=np.float32)


def laplacian_variance(gray_image: np.ndarray) -> float:
    """Return Laplacian variance on a grayscale image array."""
    center = gray_image * -4.0
    left = np.roll(gray_image, 1, axis=1)
    right = np.roll(gray_image, -1, axis=1)
    up = np.roll(gray_image, 1, axis=0)
    down = np.roll(gray_image, -1, axis=0)
    laplacian = center + left + right + up + down

    # Remove wrapped borders introduced by np.roll.
    inner = laplacian[1:-1, 1:-1]
    return float(np.var(inner))


def _duplicate_members(
    metrics: list[_ImageMetrics], duplicate_hash_distance: int
) -> dict[Path, int]:
    if len(metrics) < 2 or duplicate_hash_distance < 0:
        return {}

    parent = list(range(len(metrics)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    pairwise_distance: dict[tuple[int, int], int] = {}
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            distance = metrics[i].hash_value - metrics[j].hash_value
            pairwise_distance[(i, j)] = distance
            if distance <= duplicate_hash_distance:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(len(metrics)):
        groups.setdefault(find(idx), []).append(idx)

    duplicate_map: dict[Path, int] = {}
    for member_indexes in groups.values():
        if len(member_indexes) <= 1:
            continue

        keep_index = sorted(
            member_indexes,
            key=lambda i: (-metrics[i].blur_score, str(metrics[i].path)),
        )[0]
        for idx in member_indexes:
            if idx == keep_index:
                continue
            i, j = sorted((idx, keep_index))
            duplicate_map[metrics[idx].path] = pairwise_distance.get((i, j), 0)

    return duplicate_map


def _looks_like_screenshot(path: Path, width: int, height: int) -> bool:
    if path.suffix.lower() != ".png":
        return False
    if "screenshot" in path.name.lower():
        return True
    return _is_screen_like_dimensions(width, height)


def _is_screen_like_dimensions(width: int, height: int) -> bool:
    long_edge = max(width, height)
    short_edge = min(width, height)
    if long_edge < 1000 or short_edge < 700:
        return False
    ratio = long_edge / short_edge
    return 1.3 <= ratio <= 2.2
