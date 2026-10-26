from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import load_settings
from .filters import filter_images
from .ingest import discover_images
from .journal import build_journal_from_run
from .mover import empty_review, move_to_review, undo
from .report import build_report
from .review import (
    DecisionRecord,
    PROMPT_VERSION,
    append_decisions,
    load_review_prompt,
    review_survivors,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sloth")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_sub = subparsers.add_parser("run")
    run_sub.add_argument("--input", "--input-folder", dest="input_folder", type=Path, default=None)
    run_sub.add_argument("--manifest", type=Path, default=None)
    run_sub.add_argument("--dry-run", action="store_true")
    run_sub.set_defaults(handler=_run_command)

    undo_sub = subparsers.add_parser("undo")
    undo_sub.add_argument("--run", dest="run_id", required=True)
    undo_sub.add_argument("--input", "--input-folder", dest="input_folder", type=Path, default=None)
    undo_sub.add_argument("--manifest", type=Path, default=None)
    undo_sub.set_defaults(handler=_undo_command)

    empty_sub = subparsers.add_parser("empty-review")
    empty_sub.add_argument("--input", "--input-folder", dest="input_folder", type=Path, default=None)
    empty_sub.set_defaults(handler=_empty_review_command)

    report_sub = subparsers.add_parser("report")
    report_sub.add_argument("--input", "--input-folder", dest="input_folder", type=Path, default=None)
    report_sub.add_argument("--out", type=Path, default=None)
    report_sub.set_defaults(handler=_report_command)

    eval_sub = subparsers.add_parser("eval")
    eval_sub.add_argument("--input", "--input-folder", dest="input_folder", type=Path, default=None)
    eval_sub.add_argument("--labels", type=Path, default=Path("eval/labels.csv"))
    eval_sub.add_argument("--model", default=None)
    eval_sub.set_defaults(handler=_eval_command)

    return parser


def _run_command(args: argparse.Namespace) -> int:
    settings = load_settings()
    input_folder = (args.input_folder or settings.input_folder).expanduser().resolve()
    manifest_path = args.manifest

    photos = discover_images(input_folder, manifest_path=manifest_path)
    if not photos:
        print("No supported photos found.")
        return 0

    run_id = _build_run_id()
    started = datetime.now(timezone.utc)
    filtered = filter_images(photos, settings)
    rule_junk = [r for r in filtered if r.is_junk and r.reason is not None]
    survivors = [r.path for r in filtered if not r.is_junk]

    if args.dry_run:
        gemma_decisions = _dry_run_gemma_decisions(
            survivors, run_id=run_id, model_name=settings.model_name
        )
    else:
        prompt_text = load_review_prompt(PROMPT_VERSION)
        gemma_decisions = review_survivors(
            photo_paths=survivors,
            model_name=settings.model_name,
            image_size=settings.image_size,
            prompt_text=prompt_text,
            run_id=run_id,
            prompt_version=PROMPT_VERSION,
        )

    rule_decisions = [
        DecisionRecord(
            run_id=run_id,
            file=str(r.path),
            stage="rules",
            verdict="junk",
            reason=r.reason or "rules",
            caption="",
            model=settings.model_name,
            prompt_version=PROMPT_VERSION,
            seconds=0.0,
        )
        for r in rule_junk
    ]
    all_decisions = rule_decisions + gemma_decisions
    append_decisions(input_folder / "decisions.jsonl", all_decisions)

    moved_by_reason: dict[str, int] = defaultdict(int)
    moved_total = 0
    junk_rules = len(rule_junk)
    junk_gemma = 0
    needs_review = 0
    kept = 0
    gemma_seconds = 0.0
    if not args.dry_run:
        for result in rule_junk:
            move_to_review(
                file=result.path,
                reason=result.reason or "rules",
                run_id=run_id,
                stage="rules",
                score=result.score,
                input_folder=input_folder,
                manifest_path=manifest_path,
            )
            moved_by_reason[result.reason or "rules"] += 1
            moved_total += 1

    for row in gemma_decisions:
        gemma_seconds += row.seconds
        if row.verdict == "junk":
            junk_gemma += 1
            if not args.dry_run:
                move_to_review(
                    file=Path(row.file),
                    reason=row.reason,
                    run_id=run_id,
                    stage="gemma",
                    score=row.seconds,
                    input_folder=input_folder,
                    manifest_path=manifest_path,
                )
                moved_by_reason[row.reason] += 1
                moved_total += 1
        elif row.verdict == "needs_review":
            needs_review += 1
        else:
            kept += 1

    journal_entries = 0
    if not args.dry_run:
        journal_entries = len(build_journal_from_run(input_folder, run_id, settings.model_name))

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    seconds_per_photo = elapsed / len(photos) if photos else 0.0
    summary = {
        "run_id": run_id,
        "dry_run": bool(args.dry_run),
        "photos_scanned": len(photos),
        "junk_rules": junk_rules,
        "junk_gemma": junk_gemma,
        "needs_review": needs_review,
        "kept": kept,
        "seconds_per_photo": seconds_per_photo,
        "total_seconds": elapsed,
        "journal_entries": journal_entries,
        "started_at": started.isoformat(),
    }
    (input_folder / "latest_run.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    if not args.dry_run and moved_total:
        print(f"Run ID: {run_id}")
        for reason in sorted(moved_by_reason):
            print(f"{reason}: {moved_by_reason[reason]}")

    print(
        f"{len(photos)} photos · {junk_rules} junk by rules · {junk_gemma} junk by Gemma · "
        f"{needs_review} needs review · {kept} kept · {seconds_per_photo:.0f} s/photo"
    )
    return 0


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


def _report_command(args: argparse.Namespace) -> int:
    settings = load_settings()
    input_folder = (args.input_folder or settings.input_folder).expanduser().resolve()
    out_file = build_report(input_folder=input_folder, out_dir=args.out)
    print(f"Report written: {out_file}")
    return 0


def _eval_command(args: argparse.Namespace) -> int:
    from eval.eval import run_eval

    settings = load_settings()
    input_folder = (args.input_folder or settings.input_folder).expanduser().resolve()
    text = run_eval(input_folder=input_folder, labels_csv=args.labels, model_name=args.model)
    print(text)
    return 0


def _build_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _dry_run_gemma_decisions(
    survivors: list[Path], run_id: str, model_name: str
) -> list[DecisionRecord]:
    rows: list[DecisionRecord] = []
    for path in survivors:
        name = path.name.lower()
        if "screenshot" in name or "otp" in name or "meme" in name:
            verdict = "junk"
            reason = "throwaway_screenshot"
            caption = "likely throwaway screenshot"
        else:
            verdict = "keep"
            reason = "likely useful"
            caption = "photo pending full model review"
        rows.append(
            DecisionRecord(
                run_id=run_id,
                file=str(path),
                stage="gemma",
                verdict=verdict,
                reason=reason,
                caption=caption,
                model=model_name,
                prompt_version=PROMPT_VERSION,
                seconds=0.0,
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return handler(args)
