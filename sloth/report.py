from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


def build_report(input_folder: Path, out_dir: Path | None = None) -> Path:
    out_root = (out_dir or input_folder).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    assets_dir = out_root / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    latest_path = input_folder / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError("latest_run.json not found. Run `sloth run` first.")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))

    run_id = str(latest["run_id"])
    decisions = _rows_for_run(input_folder / "decisions.jsonl", run_id)
    manifest = _rows_for_run(input_folder / "manifest.jsonl", run_id)
    journal_rows = _rows_for_run(input_folder / "journal" / "journal_log.jsonl", run_id)

    by_reason: dict[str, list[dict[str, Any]]] = {}
    for row in decisions:
        if row.get("verdict") != "junk":
            continue
        reason = str(row.get("reason", "unknown"))
        by_reason.setdefault(reason, []).append(row)

    thumb_map = _copy_thumbnails(decisions, assets_dir)
    html_text = _render_html(latest, by_reason, decisions, journal_rows, thumb_map)
    out_file = out_root / "report.html"
    out_file.write_text(html_text, encoding="utf-8")

    # Also copy journal files for local navigation in docs output.
    journal_src = input_folder / "journal"
    journal_dst = out_root / "journal"
    if journal_src.exists():
        if journal_dst.exists():
            shutil.rmtree(journal_dst)
        shutil.copytree(journal_src, journal_dst)

    return out_file


def _rows_for_run(path: Path, run_id: str) -> list[dict[str, Any]]:
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


def _copy_thumbnails(decisions: list[dict[str, Any]], assets_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in decisions:
        file_path = Path(str(row.get("file", "")))
        if not file_path.exists():
            continue
        key = str(file_path)
        if key in mapping:
            continue
        thumb_name = f"{len(mapping)+1:04d}.jpg"
        thumb_path = assets_dir / thumb_name
        try:
            with Image.open(file_path) as img:
                rgb = img.convert("RGB")
                rgb.thumbnail((240, 240))
                rgb.save(thumb_path, format="JPEG", quality=85)
            mapping[key] = f"report_assets/{thumb_name}"
        except Exception:
            continue
    return mapping


def _render_html(
    latest: dict[str, Any],
    by_reason: dict[str, list[dict[str, Any]]],
    decisions: list[dict[str, Any]],
    journal_rows: list[dict[str, Any]],
    thumb_map: dict[str, str],
) -> str:
    summary = (
        f"{latest.get('photos_scanned', 0)} photos scanned · "
        f"{latest.get('junk_rules', 0)} moved by rules · "
        f"{latest.get('junk_gemma', 0)} moved by Gemma · "
        f"{latest.get('kept', 0)} kept · "
        f"{latest.get('journal_entries', 0)} entries · "
        f"{latest.get('seconds_per_photo', 0):.2f} s/photo · "
        f"{latest.get('total_seconds', 0):.2f}s total"
    )
    sections = []
    for reason, rows in sorted(by_reason.items()):
        cards = []
        for row in rows:
            file_str = str(row.get("file", ""))
            img = thumb_map.get(file_str, "")
            note = row.get("caption") or row.get("reason") or ""
            cards.append(
                f"""
                <article class="card">
                  {'<img src="'+html.escape(img)+'" alt="thumbnail"/>' if img else ''}
                  <div class="meta"><strong>{html.escape(Path(file_str).name)}</strong></div>
                  <div class="meta">reason: {html.escape(reason)}</div>
                  <div class="meta">note: {html.escape(str(note))}</div>
                </article>
                """
            )
        sections.append(
            f"<h3>{html.escape(reason)}</h3><div class='grid'>{''.join(cards)}</div>"
        )

    journal_html = []
    for row in journal_rows:
        file_path = Path(str(row.get("file", "")))
        label = str(row.get("status", "unknown"))
        title = file_path.name
        journal_html.append(
            f"<li><a href='journal/{html.escape(title)}'>{html.escape(title)}</a> "
            f"<span class='badge'>{html.escape(label)}</span></li>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Sloth Report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#f2f7f2;color:#243128;margin:0;padding:16px;}}
.container{{max-width:980px;margin:0 auto;}}
.bar{{background:#d9e8d8;border-radius:12px;padding:14px;margin-bottom:16px;line-height:1.5;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;}}
.card{{background:#fff;border-radius:12px;padding:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);}}
.card img{{width:100%;height:120px;object-fit:cover;border-radius:8px;}}
.meta{{font-size:.9rem;margin-top:6px;word-break:break-word;}}
.note{{background:#fff8dc;border-radius:8px;padding:10px;margin:10px 0;}}
.badge{{background:#d0e7cc;padding:2px 8px;border-radius:999px;font-size:.8rem;}}
@media (max-width:640px){{body{{padding:10px;}}}}
</style></head>
<body><div class="container">
<h1>Sloth report</h1>
<div class="bar">{html.escape(summary)}</div>
<section><h2>Review grid</h2>
<p class="note">Nothing was deleted. Use <code>sloth undo --run &lt;id&gt;</code> to restore moved files.</p>
{''.join(sections)}
</section>
<section><h2>Journal entries</h2><ul>{''.join(journal_html)}</ul></section>
</div></body></html>"""
