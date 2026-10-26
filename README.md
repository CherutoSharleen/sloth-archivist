# Sloth Archivist

[See an example](docs/report.html)

Sloth Archivist is an offline-first Python CLI that triages photo folders and writes day-by-day journal entries from the photos worth keeping. It uses deterministic image quality rules first, then local Gemma inference through Ollama for the remaining photos, and keeps a full audit trail of every decision.

## Why this project exists

Large camera rolls usually contain a small number of meaningful photos mixed with duplicates, blurry shots, over/under-exposed images, and throwaway screenshots. Manual cleanup is slow and hard to maintain consistently.

Sloth Archivist automates this workflow with three priorities:

1. **Safety first**: nothing is deleted during triage; junk is moved to a review folder and can be restored with one command.
2. **Local inference**: Gemma runs through Ollama on your machine.
3. **Traceability**: every decision is logged with stage, reason, model, prompt version, and timing.

## Core workflow

`sloth run` executes this pipeline:

1. **Ingest (`ingest.py`)**  
   Finds `.jpg/.jpeg/.png`, skips `_sloth_review`, ignores already-moved files from `manifest.jsonl`, and warns on HEIC/HEIF.

2. **Rule filters (`filters.py`)**  
   - Near-duplicates by perceptual hash (`imagehash.phash`)  
   - Blur by Laplacian variance  
   - Brightness by grayscale mean (dark / overexposed)  
   - Screenshot flagging for Gemma review

3. **Gemma review (`review.py`, `gemma.py`)**  
   Only photos that survive rules go to Gemma. Responses are schema-validated JSON with retry-on-invalid behavior.

4. **Move + journal + reporting**  
   - Junk photos are moved to `_sloth_review/<reason>/` (not deleted)  
   - Moves are logged in `manifest.jsonl`  
   - `decisions.jsonl` records one decision row per photo  
   - Kept photos can be grouped into journal entries with fact-check fallback  
   - `sloth report` builds a static HTML report

## Project structure

```text
sloth/
  cli.py
  config.py
  ingest.py
  metadata.py
  filters.py
  gemma.py
  review.py
  mover.py
  journal.py
  factcheck.py
  report.py
prompts/
  review_v1.txt
  journal_v1.txt
  style_samples.txt
eval/
  eval.py
  labels.csv
tests/
docs/
```

## Requirements

- Python 3.11+
- Ollama installed and available (for real Gemma runs)
- Local model pulled (default: `gemma3:4b`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

`sloth/config.py` defines defaults. You can override with optional `config.toml`:

```toml
input_folder = "C:\\Users\\USER\\Pictures\\Screenshots"
model_name = "gemma3:4b"
blur_threshold = 100.0
dark_threshold = 40.0
bright_threshold = 215.0
duplicate_hash_distance = 5
image_size = 768
```

Environment variables:

- `OLLAMA_HOST`: custom Ollama endpoint
- `SLOTH_FAKE_GEMMA=1`: deterministic non-network review path (useful for CI/smoke tests)

## CLI commands

```bash
python -m sloth run --input <folder> [--dry-run]
python -m sloth undo --run <run_id> --input <folder>
python -m sloth empty-review --input <folder>
python -m sloth report --input <folder> [--out docs]
python -m sloth eval --input <folder> --labels eval/labels.csv [--model <name>]
```

### Dry-run output

`run --dry-run` prints a compact run summary:

```text
N photos Â· X junk by rules Â· Y junk by Gemma Â· Z needs review Â· K kept Â· S s/photo
```

## Data files written by the pipeline

- `manifest.jsonl`: file move log (`rules` / `gemma` stages)
- `decisions.jsonl`: per-photo decision records
- `latest_run.json`: summary snapshot for report generation
- `journal/YYYY-MM-DD.md`: generated daily entries
- `journal/journal_log.jsonl`: `pass` / `retried` / `template` state per day

## Safety and undo

- Triage does not delete source photos.
- `undo` restores files moved in a specific run.
- `empty-review` requires explicit `yes` confirmation and uses recycle-bin semantics (`send2trash`).

## Testing

Run test suite:

```bash
pytest
```

The test suite validates rule filters, ingest/metadata parsing, move/undo integrity, Gemma schema/retry behavior, and CLI integration paths.

## CI

The GitHub Actions workflow:

- installs dependencies on Python 3.11,
- runs `pytest`,
- runs `sloth run --input sample_photos --dry-run`,
- uploads the dry-run summary artifact.

## Current status

This repository contains a working baseline for:

- deterministic photo triage rules,
- local Gemma decision integration,
- journaling with fact-check fallback,
- static report generation,
- evaluation metrics logging.

For a guided architecture overview, open:

- `docs/python-ml-walkthrough.html`

## Stress test asset rebuild

Rebuild the reproducible real-world stress set locally (seed 42):

`ash
python scripts/fetch_stress_assets.py
python scripts/make_stress_variants.py
python -m sloth eval --labels eval/labels_real.csv --input stress_assets/real/set
` 

Committed metadata and credits:
- stress_assets/real/SOURCES.csv`r
- stress_assets/real/ATTRIBUTION.md`r

Large generated folders are intentionally ignored by git:
- stress_assets/real/originals/`r
- stress_assets/real/set/`r

