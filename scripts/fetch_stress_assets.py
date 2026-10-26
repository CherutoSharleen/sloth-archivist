from __future__ import annotations

import csv
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SlothArchivist/1.0 (https://github.com/CherutoSharleen/sloth-archivist)"
REQUEST_DELAY_SECONDS = 1.0
ALLOWED_LICENSES = {
    "cc0",
    "public domain",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
}
PEOPLE_WORDS = {
    "portrait",
    "people",
    "crowd",
    "man",
    "woman",
    "child",
    "family",
    "selfie",
    "wedding",
    "festival",
}


@dataclass(frozen=True)
class CategorySpec:
    key: str
    quota: int
    searches: tuple[str, ...]


CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec("landscape", 10, ("mountain landscape", "alpine valley", "volcanic mountain")),
    CategorySpec("water", 8, ("lake shore", "beach coastline", "coastal cliffs")),
    CategorySpec("city", 8, ("city street architecture", "historic building exterior", "urban streetscape")),
    CategorySpec("market", 7, ("market stall exterior", "shopfront street market", "bazaar goods")),
    CategorySpec("food", 7, ("food on plate", "restaurant dish", "meal on table")),
    CategorySpec("document", 5, ("receipt scan", "ticket document", "printed invoice")),
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    originals_dir = root / "stress_assets" / "real" / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)
    sources_csv = root / "stress_assets" / "real" / "SOURCES.csv"

    existing_rows = _read_sources(sources_csv)
    used_titles = {row["title"] for row in existing_rows}
    rows = dict((row["file"], row) for row in existing_rows)

    random.seed(42)
    counts: dict[str, int] = {spec.key: 0 for spec in CATEGORIES}

    for spec in CATEGORIES:
        existing_files = sorted(originals_dir.glob(f"keep_{spec.key}_*.jpg"))
        counts[spec.key] = len(existing_files)

        next_index = len(existing_files) + 1
        for query in spec.searches:
            if counts[spec.key] >= spec.quota:
                break
            gsrcontinue = None
            page_safety = 0
            while counts[spec.key] < spec.quota and page_safety < 40:
                page_safety += 1
                payload = _api_request(query, gsrcontinue)
                pages = payload.get("query", {}).get("pages", {})
                for page in pages.values():
                    if counts[spec.key] >= spec.quota:
                        break
                    candidate = _extract_candidate(page)
                    if candidate is None:
                        continue
                    if candidate["title"] in used_titles:
                        continue

                    file_name = f"keep_{spec.key}_{next_index:02d}.jpg"
                    out_path = originals_dir / file_name
                    if out_path.exists():
                        used_titles.add(candidate["title"])
                        counts[spec.key] += 1
                        next_index += 1
                        continue

                    ok = _download_thumbnail(candidate["thumb_url"], out_path)
                    if not ok:
                        continue

                    used_titles.add(candidate["title"])
                    counts[spec.key] += 1
                    next_index += 1
                    rows[file_name] = {
                        "file": file_name,
                        "category": spec.key,
                        "title": candidate["title"],
                        "author": candidate["author"],
                        "license": candidate["license"],
                        "source_url": candidate["source_url"],
                    }

                gsrcontinue = payload.get("continue", {}).get("gsroffset")
                if gsrcontinue is None:
                    break

    _write_sources(sources_csv, sorted(rows.values(), key=lambda r: r["file"]))
    _write_attribution(root / "stress_assets" / "real" / "ATTRIBUTION.md", sources_csv)

    print("Per-category totals:")
    unmet = False
    for spec in CATEGORIES:
        count = counts[spec.key]
        print(f"- {spec.key}: {count}/{spec.quota}")
        if count < spec.quota:
            unmet = True
    if unmet:
        print("WARNING: one or more quotas were not met.")
    else:
        print("All category quotas met.")
    return 0


def _api_request(query: str, gsrcontinue: int | None) -> dict[str, Any]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrlimit": 50,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": 1280,
    }
    if gsrcontinue is not None:
        params["gsroffset"] = gsrcontinue

    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    data = _request_bytes(url)
    return json.loads(data.decode("utf-8"))


def _request_bytes(url: str, attempts: int = 5) -> bytes:
    backoff = 1.0
    for idx in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            time.sleep(REQUEST_DELAY_SECONDS)
            with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and idx < attempts - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except urllib.error.URLError:
            if idx < attempts - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    raise RuntimeError("Request failed after retries")


def _extract_candidate(page: dict[str, Any]) -> dict[str, str] | None:
    title = str(page.get("title", "")).strip()
    info_list = page.get("imageinfo") or []
    if not title or not info_list:
        return None

    info = info_list[0]
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    if width < 800 or height <= 0:
        return None
    if width / max(height, 1) > 3.0:
        return None

    ext = info.get("extmetadata") or {}
    license_name = _ext_value(ext, "LicenseShortName").strip()
    if not _license_allowed(license_name):
        return None

    description = _ext_value(ext, "ImageDescription")
    text_blob = f"{title} {description}".lower()
    if any(word in text_blob for word in PEOPLE_WORDS):
        return None

    author = _clean_text(_ext_value(ext, "Artist") or _ext_value(ext, "Credit"))
    source_url = str(info.get("descriptionurl") or info.get("url") or "").strip()
    thumb_url = str(info.get("thumburl") or info.get("url") or "").strip()
    if not thumb_url:
        return None

    return {
        "title": title,
        "author": author or "unknown",
        "license": license_name,
        "source_url": source_url,
        "thumb_url": thumb_url,
    }


def _download_thumbnail(url: str, out_path: Path) -> bool:
    try:
        blob = _request_bytes(url)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp = out_path.with_suffix(".tmp")
        temp.write_bytes(blob)
        with Image.open(temp) as image:
            image.convert("RGB").save(out_path, format="JPEG", quality=92)
        temp.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _read_sources(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if all(k in row for k in ("file", "category", "title", "author", "license", "source_url")):
                rows.append({k: str(row.get(k, "")) for k in ("file", "category", "title", "author", "license", "source_url")})
    return rows


def _write_sources(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "category", "title", "author", "license", "source_url"])
        writer.writeheader()
        writer.writerows(rows)


def _write_attribution(path: Path, sources_csv: Path) -> None:
    rows = _read_sources(sources_csv)
    lines = ["# Attribution", "", "Generated from `SOURCES.csv`.", ""]
    for row in rows:
        title = row["title"] or row["file"]
        author = row["author"] or "unknown"
        license_name = row["license"] or "unknown"
        source = row["source_url"] or ""
        lines.append(f"- **{row['file']}** — *{title}* by {author}; license: {license_name}; source: {source}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _license_allowed(value: str) -> bool:
    low = value.strip().lower()
    return any(token in low for token in ALLOWED_LICENSES)


def _ext_value(extmetadata: dict[str, Any], key: str) -> str:
    raw = extmetadata.get(key)
    if isinstance(raw, dict):
        val = raw.get("value")
        return str(val or "")
    return ""


def _clean_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


if __name__ == "__main__":
    raise SystemExit(main())
