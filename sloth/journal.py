from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .factcheck import check_entry
from .metadata import read_photo_info

PROMPT_VERSION = "journal_v1"


@dataclass(frozen=True)
class JournalResult:
    date_key: str
    status: str  # pass|retried|template
    path: Path
    photo_count: int


def build_journal_from_run(
    input_folder: Path, run_id: str, model_name: str
) -> list[JournalResult]:
    decisions = _read_decisions_for_run(input_folder / "decisions.jsonl", run_id)
    kept = [d for d in decisions if d.get("verdict") == "keep"]
    if not kept:
        return []

    prompt_text = _load_prompt(PROMPT_VERSION)
    style_samples = _load_style_samples()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        path = Path(str(row["file"]))
        info = read_photo_info(path)
        grouped[info.taken_at.date().isoformat()].append(
            {
                "path": path,
                "time": info.taken_at.strftime("%H:%M"),
                "place": info.place or "Unknown place",
                "caption": str(row.get("caption", "")).strip(),
            }
        )

    journal_dir = input_folder / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    log_path = journal_dir / "journal_log.jsonl"
    results: list[JournalResult] = []

    for date_key in sorted(grouped.keys()):
        facts = grouped[date_key]
        facts_text = _facts_text(facts)

        status = "pass"
        entry = _ask_journal_model(
            model_name=model_name,
            prompt=prompt_text,
            style_samples=style_samples,
            facts_text=facts_text,
            feedback=None,
        )
        check = check_entry(entry, facts_text) if entry.strip() else check_entry("missing entry", "")
        if not check.ok:
            status = "retried"
            feedback = "The facts do not mention: " + ", ".join(check.unsupported_words)
            entry = _ask_journal_model(
                model_name=model_name,
                prompt=prompt_text,
                style_samples=style_samples,
                facts_text=facts_text,
                feedback=feedback,
            )
            check = check_entry(entry, facts_text) if entry.strip() else check_entry("missing entry", "")
            if not check.ok:
                status = "template"
                entry = _template_entry(date_key, facts)

        out_path = journal_dir / f"{date_key}.md"
        out_path.write_text(_render_markdown(date_key, entry, facts), encoding="utf-8")
        _append_jsonl(
            log_path,
            {
                "run_id": run_id,
                "date": date_key,
                "status": status,
                "file": str(out_path),
                "photo_count": len(facts),
            },
        )
        results.append(
            JournalResult(
                date_key=date_key,
                status=status,
                path=out_path,
                photo_count=len(facts),
            )
        )
    return results


def _read_decisions_for_run(path: Path, run_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and str(row.get("run_id")) == run_id:
            rows.append(row)
    return rows


def _load_prompt(version: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "prompts" / f"{version}.txt").read_text(encoding="utf-8").strip()


def _load_style_samples() -> str:
    root = Path(__file__).resolve().parents[1]
    style_path = root / "prompts" / "style_samples.txt"
    if style_path.exists():
        return style_path.read_text(encoding="utf-8").strip()
    return ""


def _facts_text(facts: list[dict[str, Any]]) -> str:
    lines = []
    by_place: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in facts:
        by_place[str(item["place"])].append(item)
    for place in sorted(by_place.keys()):
        lines.append(f"place: {place}")
        for item in sorted(by_place[place], key=lambda row: row["time"]):
            lines.append(f"  - time: {item['time']}; caption: {item['caption'] or 'no caption'}")
    return "\n".join(lines)


def _ask_journal_model(
    model_name: str, prompt: str, style_samples: str, facts_text: str, feedback: str | None
) -> str:
    message = (
        f"{prompt}\n\nStyle samples:\n{style_samples}\n\nFacts:\n{facts_text}\n\n"
        "Write 80-150 words in first person."
    )
    if feedback:
        message += f"\n\nRevision feedback: {feedback}"
    try:
        import ollama

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": message}],
            "options": {"temperature": 0},
        }
        host = __import__("os").getenv("OLLAMA_HOST")
        if host:
            kwargs["host"] = host
        response = ollama.chat(**kwargs)
        content = response.get("message", {}).get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    except Exception:
        pass
    return ""


def _template_entry(date_key: str, facts: list[dict[str, Any]]) -> str:
    pretty_day = datetime.strptime(date_key, "%Y-%m-%d").strftime("%A")
    place = facts[0]["place"] if facts else "Unknown place"
    captions = [f["caption"] for f in facts if f["caption"]]
    sample = ", ".join(captions[:3]) if captions else "no detailed captions"
    return f"{pretty_day}, {place}: {len(facts)} photos — {sample}."


def _render_markdown(date_key: str, entry: str, facts: list[dict[str, Any]]) -> str:
    dt = datetime.strptime(date_key, "%Y-%m-%d")
    header_date = f"{dt.strftime('%A')}, {dt.day} {dt.strftime('%B %Y')}"
    place = facts[0]["place"] if facts else "Unknown place"
    lines = [f"# {header_date} · {place}", "", entry.strip(), "", "## Photos", ""]
    for item in facts:
        rel = item["path"].as_posix()
        lines.append(f"- [{item['path'].name}]({rel}) — {item['time']} · {item['place']}")
    return "\n".join(lines) + "\n"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True) + "\n")
