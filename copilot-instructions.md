# Sloth Archivist – instructions for Copilot

## What this project is
A Python command-line tool that cleans a photo folder overnight and writes a daily journal
from the photos worth keeping. It runs fully offline on a Windows laptop.
Submission for the DEV Hacktoberfest Open-Source AI Challenge, Week 3 "Slow and Steady".
Deadline: Saturday, October 24, 2026. Prefer simple, working code over clever code.

## Environment
- Windows 11, 16GB RAM, no CUDA GPU. Python 3.11+.
- The only AI model is Gemma, run locally through Ollama, called with the `ollama` Python package.
- The model name comes from config (default `gemma3:4b`); never hard-code it.
- Never call a cloud AI API (OpenAI, Anthropic, Google Gemini API, etc.) from the product.

## Hard rules
1. NEVER delete files. Junk is MOVED to `<input>/_sloth_review/<reason>/` and every move is
   appended to `manifest.jsonl` so `undo` can restore it. Only the `empty-review` command may
   remove files, only after the user types "yes", and only via `send2trash` (Recycle Bin).
2. Default to dry runs during development. `--dry-run` must write decisions and the report
   without moving anything.
3. Never point the tool at real photo folders in examples or tests. Use `sample_photos/`
   or `C:\sloth-test\photos`.
4. When unsure, the verdict is `needs_review`, not `junk`.
5. Gemma calls use `temperature: 0` and a JSON schema via Ollama's `format` parameter.
   Validate every response with Pydantic. Retry malformed output twice, then `needs_review`.
6. Prompts live in `prompts/` as versioned text files (`review_v1.txt`, `journal_v1.txt`).
   Every decision logged records the prompt version and model name.
7. Every module gets pytest tests. Tests must not need Ollama: use a fake Gemma client.
8. Paths with `pathlib`. Works on Windows paths with spaces.

## Layout
```
sloth/              package
  config.py         thresholds, model name, folders (dataclass, loaded from config.toml)
  ingest.py         find image files, skip ones already in the manifest
  metadata.py       EXIF date + GPS (Pillow), offline place names (reverse_geocoder)
  filters.py        duplicates (imagehash), blur (Laplacian variance), too dark/bright
  gemma.py          Ollama client wrapper + FakeGemma for tests
  review.py         Gemma keep/junk review with JSON schema + retries
  mover.py          move to review folder, manifest.jsonl, undo
  journal.py        group kept photos by day/place, write Markdown entries
  factcheck.py      check journal text against its source facts
  report.py         static HTML report (summary, review grid, journal)
  cli.py            commands: run, undo, report, empty-review, eval
prompts/            versioned prompt files
tests/              pytest
eval/               labels.csv, eval.py, EVAL.md, REVIEW.md
sample_photos/      ~20 non-sensitive photos for tests, CI and the demo
docs/               GitHub Pages: the published sample report
.github/workflows/  ci.yml
```

## Data formats
Model review output:
`{"caption": str, "verdict": "keep"|"junk"|"needs_review", "reason": str, "confidence": float}`

`manifest.jsonl` line:
`{"run_id", "original", "moved_to", "stage": "rules"|"gemma", "reason", "score"}`

`decisions.jsonl` line:
`{"run_id", "file", "stage", "verdict", "reason", "caption", "model", "prompt_version", "seconds"}`

Journal: `journal/YYYY-MM-DD.md`, heading `# <Weekday>, <D Month YYYY> · <Place>`.

## Style
- Small functions, type hints, docstrings on public functions.
- Explain non-obvious choices in a short comment (why, not what).
- Print a clear one-line summary at the end of every command.
