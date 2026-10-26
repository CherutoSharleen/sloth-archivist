from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sloth.config import load_settings
from sloth.filters import filter_images
from sloth.review import PROMPT_VERSION, load_review_prompt, review_survivors


def run_eval(input_folder: Path, labels_csv: Path, model_name: str | None = None) -> str:
    settings = load_settings()
    model = model_name or settings.model_name

    rows = _read_labels(labels_csv, input_folder)
    if not rows:
        raise ValueError(f"No valid label rows found in {labels_csv}")

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

    overall = {"tp": 0, "fp": 0, "fn": 0}
    reason_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    confusion: Counter[tuple[str, str]] = Counter()
    wrong_rows: list[dict[str, str]] = []

    decided_rules = 0
    decided_gemma = 0
    total_seconds = 0.0
    retries = 0

    for row in rows:
        path: Path = row["path"]
        file_name = path.name
        expected_label = str(row["label"])
        expected_reason = str(row.get("expected_reason", "") or "").strip().lower()

        pred_label = "keep"
        pred_reason = "keep"
        stage = "gemma"

        if path in rule_junk:
            decided_rules += 1
            stage = "rules"
            pred_label = "junk"
            pred_reason = str(filtered[path].reason or "rules")
        else:
            decision = gemma_map.get(path)
            if decision is not None:
                decided_gemma += 1
                total_seconds += decision.seconds
                pred_label = "junk" if decision.verdict == "junk" else "keep"
                pred_reason = decision.reason if decision.verdict == "junk" else "keep"
                if decision.reason == "invalid_model_output":
                    retries += 1

        if pred_label == "junk" and expected_label == "junk":
            overall["tp"] += 1
        elif pred_label == "junk" and expected_label == "keep":
            overall["fp"] += 1
        elif pred_label == "keep" and expected_label == "junk":
            overall["fn"] += 1

        expected_reason_for_table = expected_reason or ("keep" if expected_label == "keep" else "junk")
        predicted_reason_for_table = pred_reason or ("keep" if pred_label == "keep" else "junk")
        confusion[(expected_reason_for_table, predicted_reason_for_table)] += 1

        if expected_label == "junk":
            bucket = expected_reason_for_table
            if pred_label == "junk" and predicted_reason_for_table == bucket:
                reason_stats[bucket]["tp"] += 1
            elif pred_label == "junk" and predicted_reason_for_table != bucket:
                reason_stats[bucket]["fn"] += 1
                reason_stats[predicted_reason_for_table]["fp"] += 1
            elif pred_label == "keep":
                reason_stats[bucket]["fn"] += 1

        wrong = pred_label != expected_label
        if not wrong and expected_label == "junk":
            wrong = predicted_reason_for_table != expected_reason_for_table

        if wrong:
            wrong_rows.append(
                {
                    "file": file_name,
                    "expected_label": expected_label,
                    "expected_reason": expected_reason_for_table,
                    "predicted_label": pred_label,
                    "predicted_reason": predicted_reason_for_table,
                    "stage": stage,
                }
            )

    precision = _ratio(overall["tp"], overall["tp"] + overall["fp"])
    recall = _ratio(overall["tp"], overall["tp"] + overall["fn"])
    spp = total_seconds / len(paths)
    retry_rate = _ratio(retries, decided_gemma)

    lines: list[str] = []
    lines.append(f"junk precision: {precision:.3f}")
    lines.append(f"junk recall: {recall:.3f}")
    lines.append(f"decided by rules: {decided_rules}")
    lines.append(f"decided by Gemma: {decided_gemma}")
    lines.append(f"seconds per photo: {spp:.2f}")
    lines.append(f"JSON retry rate: {retry_rate:.3f}")
    lines.append("")
    lines.append("per-reason precision/recall (junk reasons)")
    lines.append("reason,tp,fp,fn,precision,recall")

    reason_keys = sorted(k for k in reason_stats.keys() if k not in {"", "keep"})
    for key in reason_keys:
        tp = reason_stats[key]["tp"]
        fp = reason_stats[key]["fp"]
        fn = reason_stats[key]["fn"]
        r_precision = _ratio(tp, tp + fp)
        r_recall = _ratio(tp, tp + fn)
        lines.append(f"{key},{tp},{fp},{fn},{r_precision:.3f},{r_recall:.3f}")

    lines.append("")
    lines.append("confusion(expected_reason -> predicted_reason,count)")
    lines.append("expected_reason,predicted_reason,count")
    for (exp, pred), count in sorted(confusion.items(), key=lambda item: (item[0][0], item[0][1])):
        lines.append(f"{exp},{pred},{count}")

    lines.append("")
    lines.append("wrong_predictions")
    lines.append("file,expected_label,expected_reason,predicted_label,predicted_reason,stage")
    for row in wrong_rows:
        lines.append(
            f"{row['file']},{row['expected_label']},{row['expected_reason']},{row['predicted_label']},{row['predicted_reason']},{row['stage']}"
        )

    report = "\n".join(lines)

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
            expected_reason = str(row.get("expected_reason", "")).strip().lower()
            note = str(row.get("note", "")).strip()
            if not file_name or label not in {"keep", "junk"}:
                continue
            out.append(
                {
                    "path": (input_folder / file_name).resolve(),
                    "label": label,
                    "expected_reason": expected_reason,
                    "note": note,
                }
            )
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


def _ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return num / den


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
