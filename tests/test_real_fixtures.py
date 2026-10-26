from __future__ import annotations

import shutil
from pathlib import Path

from sloth.config import Settings
from sloth.filters import filter_images
from sloth.mover import move_to_review, undo


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "real"


def test_real_fixture_rules_flag_expected_variants() -> None:
    originals = sorted(FIXTURE_DIR.glob("keep_original_*.jpg"))

    quality_variants = [
        FIXTURE_DIR / "variant_blurry_01.jpg",
        FIXTURE_DIR / "variant_dark_01.jpg",
        FIXTURE_DIR / "variant_overexposed_01.jpg",
        FIXTURE_DIR / "variant_accidental_01.jpg",
    ]

    settings_quality = Settings(
        blur_threshold=120.0,
        dark_threshold=42.0,
        bright_threshold=212.0,
        duplicate_hash_distance=-1,
    )
    original_results = {row.path.name: row for row in filter_images(originals, settings_quality)}
    quality_results = {row.path.name: row for row in filter_images(quality_variants, settings_quality)}

    assert all(original_results[p.name].is_junk is False for p in originals)
    assert quality_results["variant_blurry_01.jpg"].reason == "blurry"
    assert quality_results["variant_dark_01.jpg"].reason == "dark"
    assert quality_results["variant_overexposed_01.jpg"].reason == "overexposed"
    assert quality_results["variant_accidental_01.jpg"].reason in {"dark", "blurry"}

    duplicate_pair = [FIXTURE_DIR / "keep_original_01.jpg", FIXTURE_DIR / "variant_duplicate_01.jpg"]
    duplicate_results = filter_images(
        duplicate_pair,
        Settings(duplicate_hash_distance=8, blur_threshold=120.0, dark_threshold=42.0, bright_threshold=212.0),
    )
    duplicate_reasons = [row.reason for row in duplicate_results]
    assert duplicate_reasons.count("duplicate") == 1


def test_burst_set_keeps_sharpest_frame() -> None:
    burst = sorted(FIXTURE_DIR.glob("variant_duplicate_burst1_*.jpg"))
    settings = Settings(duplicate_hash_distance=8, blur_threshold=100.0)
    results = filter_images(burst, settings)

    kept = [r for r in results if r.reason is None]
    junk = [r for r in results if r.reason == "duplicate"]
    assert len(kept) == 1
    assert len(junk) == 2


def test_move_and_undo_restores_fixture_copy_byte_for_byte(tmp_path: Path) -> None:
    work = tmp_path / "fixture_copy"
    shutil.copytree(FIXTURE_DIR, work)

    baseline = {p.relative_to(work): p.read_bytes() for p in work.rglob("*") if p.is_file()}
    run_id = "real-fixture-run"

    settings = Settings(
        blur_threshold=120.0,
        dark_threshold=42.0,
        bright_threshold=212.0,
        duplicate_hash_distance=8,
    )
    images = [p for p in work.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    results = filter_images(images, settings)

    moved = 0
    for row in results:
        if row.is_junk and row.reason:
            move_to_review(
                file=row.path,
                reason=row.reason,
                run_id=run_id,
                stage="rules",
                score=row.score,
                input_folder=work,
            )
            moved += 1

    assert moved > 0
    restored = undo(run_id, input_folder=work)
    assert restored == moved

    after = {p.relative_to(work): p.read_bytes() for p in work.rglob("*") if p.is_file()}
    after.pop(Path("manifest.jsonl"), None)
    assert baseline == after
