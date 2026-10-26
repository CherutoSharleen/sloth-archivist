from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from sloth.config import load_settings
from sloth.filters import filter_images
from sloth.review import PROMPT_VERSION, load_review_prompt, review_survivors


def run_eval(input_folder: Path, labels_csv: Path, model_name: str | None = None) -> str:
    settings = load_settings()
    model = model_name or settings.model_name

    rows = _read_labels(labels_csv, input_folder)
    paths = [row["path"] for row in rows]
    filtered = {r.path: r for r in filter_images(paths, settings)}

    rule_junk = {p for p, r in filtered.items() if r.is_junk}
    survivors = [p for p in paths if p not in rule_junk]
    prompt = load_review_prompt(PROMPT_VERSION)
    gemma_rows = review_survivors(
        photo_paths=survivors,
        model_name=model,
        image_size=settings.image_size,
        prompt_text=prompt,
        run_id="eval",
        prompt_version=PROMPT_VERSION,
    )
    gemma_map = {Path(row.file): row for row in gemma_rows}

    tp = fp = fn = 0
    decided_rules = 0
    decided_gemma = 0
    total_seconds = 0.0
    retries = 0
    for row in rows:
        path = row["path"]
        gold = row["label"]
        pred = "keep"
        if path in rule_junk:
            pred = "junk"
            decided_rules += 1
        else:
            decision = gemma_map.get(path)
            if decision:
                decided_gemma += 1
                total_seconds += decision.seconds
                pred = "junk" if decision.verdict == "junk" else "keep"
                if decision.reason == "invalid_model_output":
                    retries += 1

        if pred == "junk" and gold == "junk":
            tp += 1
        elif pred == "junk" and gold == "keep":
            fp += 1
        elif pred == "keep" and gold == "junk":
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    spp = total_seconds / len(paths) if paths else 0.0
    retry_rate = retries / decided_gemma if decided_gemma else 0.0

    report = (
        f"junk precision: {precision:.3f}\n"
        f"junk recall: {recall:.3f}\n"
        f"decided by rules: {decided_rules}\n"
        f"decided by Gemma: {decided_gemma}\n"
        f"seconds per photo: {spp:.2f}\n"
        f"JSON retry rate: {retry_rate:.3f}"
    )
    _append_eval_md(
        labels_csv.parent / "EVAL.md",
        model=model,
        precision=precision,
        recall=recall,
        rules=decided_rules,
        gemma=decided_gemma,
        spp=spp,
        retry_rate=retry_rate,
    )
    return report


def _read_labels(labels_csv: Path, input_folder: Path) -> list[dict[str, Path | str]]:
    out: list[dict[str, Path | str]] = []
    with labels_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            file_name = str(row.get("file", "")).strip()
            label = str(row.get("label", "")).strip().lower()
            if not file_name or label not in {"keep", "junk"}:
                continue
            out.append({"path": (input_folder / file_name).resolve(), "label": label})
    return out


def _append_eval_md(
    path: Path,
    model: str,
    precision: float,
    recall: float,
    rules: int,
    gemma: int,
    spp: float,
    retry_rate: float,
) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        path.write_text(
            "| date | model | review_prompt | journal_prompt | precision | recall | rules | gemma | s/photo | retry_rate |\n"
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {stamp} | {model} | review_v1 | journal_v1 | {precision:.3f} | {recall:.3f} | {rules} | {gemma} | {spp:.2f} | {retry_rate:.3f} |\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_photos")
    parser.add_argument("--labels", default="eval/labels.csv")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    text = run_eval(Path(args.input), Path(args.labels), model_name=args.model)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
