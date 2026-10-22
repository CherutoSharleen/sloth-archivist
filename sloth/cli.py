from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import load_settings
from .filters import FilterResult, filter_images
from .ingest import discover_images
from .mover import empty_review, move_to_review, undo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sloth")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_sub = subparsers.add_parser("run")
    run_sub.add_argument("--input-folder", type=Path, default=None)
    run_sub.add_argument("--manifest", type=Path, default=None)
    run_sub.add_argument("--dry-run", action="store_true")
    run_sub.set_defaults(handler=_run_command)

    undo_sub = subparsers.add_parser("undo")
    undo_sub.add_argument("--run", dest="run_id", required=True)
    undo_sub.add_argument("--input-folder", type=Path, default=None)
    undo_sub.add_argument("--manifest", type=Path, default=None)
    undo_sub.set_defaults(handler=_undo_command)

    empty_sub = subparsers.add_parser("empty-review")
    empty_sub.add_argument("--input-folder", type=Path, default=None)
    empty_sub.set_defaults(handler=_empty_review_command)

    for command in ("report", "eval"):
        sub = subparsers.add_parser(command)
        sub.set_defaults(handler=_not_implemented)

    return parser


def _not_implemented(_args: argparse.Namespace) -> int:
    print("not implemented")
    return 0


def _run_command(args: argparse.Namespace) -> int:
    settings = load_settings()
    input_folder = (args.input_folder or settings.input_folder).expanduser()
    manifest_path = args.manifest

    photos = discover_images(input_folder, manifest_path=manifest_path)
    if not photos:
        print("No supported photos found.")
        return 0

    results = filter_images(photos, settings)
    if args.dry_run:
        _print_dry_run_summary(results)
        print(f"Dry run complete: {len(results)} photo(s) analyzed.")
        return 0

    run_id = _build_run_id()
    moved_by_reason: dict[str, int] = defaultdict(int)
    moved_total = 0
    manifest_path = args.manifest
    for result in results:
        if not result.is_junk or result.reason is None:
            continue
        move_to_review(
            file=result.path,
            reason=result.reason,
            run_id=run_id,
            stage="rules",
            score=result.score,
            input_folder=input_folder,
            manifest_path=manifest_path,
        )
        moved_by_reason[result.reason] += 1
        moved_total += 1

    if moved_total:
        print(f"Run ID: {run_id}")
        for reason in sorted(moved_by_reason.keys()):
            print(f"{reason}: {moved_by_reason[reason]}")
    print(
        f"Run complete: {len(results)} photo(s) analyzed, {moved_total} moved to review."
    )
    return 0


def _print_dry_run_summary(results: list[FilterResult]) -> None:
    by_reason: dict[str, list[FilterResult]] = defaultdict(list)
    for result in results:
        if result.reason in {"blurry", "dark", "duplicate"}:
            by_reason[result.reason].append(result)

    for reason in ("blurry", "dark", "duplicate"):
        matches = by_reason.get(reason, [])
        if not matches:
            continue
        print(f"{reason}:")
        for item in sorted(matches, key=lambda x: x.path.name.lower()):
            print(f"  - {item.path.name} (score={item.score:.2f})")


def _undo_command(args: argparse.Namespace) -> int:
    settings = load_settings()
    input_folder = (args.input_folder or settings.input_folder).expanduser()
    restored = undo(args.run_id, input_folder=input_folder, manifest_path=args.manifest)
    print(f"Undo complete: restored {restored} file(s) for run {args.run_id}.")
    return 0


def _empty_review_command(args: argparse.Namespace) -> int:
    settings = load_settings()
    input_folder = (args.input_folder or settings.input_folder).expanduser()
    empty_review(input_folder=input_folder)
    return 0


def _build_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return handler(args)
