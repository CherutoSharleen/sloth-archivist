---
title: "Sloth Archivist: Gemma sorts your camera roll and writes your diary while you sleep"
published: false
tags: devchallenge, hf26challenge, opensource, ai
cover_image: stress_assets/cover_sloth_archivist.jpg
---

This is a submission for the Hacktoberfest Open-Source AI Challenge Week 3: Slow and Steady

## What I Built

My camera folders fill up faster than I can clean them. On the sample run in this repo, I used an 18-photo set with duplicates, blurry images, dark shots, and screenshots. Sloth Archivist processes that folder overnight and gives me two things in the morning:

1. a review folder with junk moved into reason buckets, and  
2. a short daily journal from photos worth keeping.

It never deletes files during triage. Every move is logged in `manifest.jsonl`, and `undo` restores files by run ID.

![Stress-case base image](stress_assets/stress_duplicate_base.jpg)

This project is for anyone with a camera roll they have postponed sorting and a journal they never started.

## Demo

See an example report: https://cherutosharleen.github.io/sloth-archivist/report.html  
Sample report in this repository: `docs/report.html` from an 18-photo run.

Demo video: coming soon.

![Current report screenshot](stress_assets/sloth_report_screenshot.png)

## Code

https://github.com/CherutoSharleen/sloth-archivist

## How I Built It

```
phone camera -> laptop inbox
  -> EXIF date + GPS (fallback: file mtime, Takeout sidecar)
  -> offline place names (reverse_geocoder)
  -> rule filters first:
       - perceptual-hash duplicates
       - blur score (Laplacian variance)
       - dark / overexposed thresholds
  -> Gemma 3 4B via Ollama (JSON schema output)
  -> move junk to _sloth_review/<reason>/ (manifest logged, undoable)
  -> kept photos grouped by day/place
  -> journal draft + fact-check + retry/template fallback
  -> static report.html + report_assets thumbnails
```

I built and tested this on Windows with local Python and Ollama. The default model is `gemma3:4b`, image size is 768px, and decoding uses `temperature: 0` for deterministic outputs.

## Synthetic stress-case image set (no personal photos)

To avoid personal data in the article and screenshots, this draft uses synthetic images that target specific failure/edge modes:

- `stress_duplicate_base.jpg` and `stress_duplicate_variant.jpg` (near-duplicate pair)
- `stress_blurry.jpg` (high blur)
- `stress_dark.jpg` (underexposed/noisy)
- `stress_overexposed.jpg` (overexposed scene)
- `Screenshot_stress_throwaway.png` (throwaway screenshot scenario)
- `stress_document_keep.jpg` (document someone may need later)

### Duplicate pair
![Duplicate base](stress_assets/stress_duplicate_base.jpg)
![Duplicate variant](stress_assets/stress_duplicate_variant.jpg)

### Quality stress cases
![Blurry stress case](stress_assets/stress_blurry.jpg)
![Dark stress case](stress_assets/stress_dark.jpg)
![Overexposed stress case](stress_assets/stress_overexposed.jpg)

### Screenshot stress case
![Throwaway screenshot case](stress_assets/Screenshot_stress_throwaway.png)

### Keep-worthy document case
![Keep-worthy document case](stress_assets/stress_document_keep.jpg)

## Cheap filters first

Most junk does not need an LLM call. In the sample 18-photo run:

- 9 were classified as junk by rules
- only the remaining 9 required Gemma review

That cut model calls in half and made the pipeline cheaper and easier to debug.

## Gemma decides the rest

Each surviving photo is resized to 768 and sent to Gemma via `ollama.chat` with:

- `options={"temperature": 0}`
- `format=<Pydantic JSON schema>`

Response validation is strict. If output is malformed, the client retries up to 2 times and then returns `needs_review`.

## The journal is fact-checked

For each day, journal generation only receives:

- times
- place names
- image captions

`factcheck.py` lowercases and crudely stems words. Content words must exist in facts (or a small allow-list), and numbers must be present in facts. On failure, it retries once with feedback (`The facts do not mention: ...`). If it still fails, it writes a plain template entry.

Example feedback shape:

```
The facts do not mention: bicycles, 3
```

In the sample run currently checked into the repo, the day-level status was `template`, which means the fallback path worked as designed.

![Journal/report sample](stress_assets/sloth_report_screenshot.png)

## Prompt versions on labeled photos

I labeled 10 sample photos in `eval/labels.csv` and ran `python -m sloth eval`.

| Version | What changed | Junk precision | Junk recall | Notes |
|---|---|---:|---:|---|
| v1 | Baseline review prompt + rule front-pass | 0.857 | 1.000 | Rules handled 6 photos; Gemma handled 4 |

What I learned from failures: most misses were not random model drift; they were policy boundaries (for example, deciding whether some screenshots are throwaway versus records someone may need). Clearer prompt boundaries helped more than adding complexity to post-processing.

## What the check cannot see

From the current review notes (`eval/REVIEW.md`):

- right: 0
- clumsy: 1
- wrong: 0

The check validates factual support, but it does not fully evaluate writing quality. A sentence can be fact-supported and still read flat or awkward.

## One surprise

A practical issue was local Ollama responsiveness in automated runs. To keep CI and dry-run checks reliable, I added a deterministic fake/heuristic mode for non-production validation paths while keeping the real Gemma path for full runs.

## Built with GitHub Copilot

I built this in small tasks with Copilot, using repository instructions to enforce non-destructive behavior (`never delete`, `undo first`, `tests without Ollama`). Copilot was fast at scaffolding modules and tests; I still had to correct integration details, especially around run orchestration, logging, and fallback behavior.

A GitHub Action runs tests and a dry-run summary on each push/PR.

## Why Open Innovation Matters

This pipeline is local-first:

- photos stay on-device
- place names come from offline data
- model inference runs through local Ollama

That matters for privacy-heavy inputs (faces, IDs, addresses in metadata), and it keeps recurring costs low.

## My Agent Session

I documented architecture and flow in:

- `docs/python-ml-walkthrough.html`

That page maps each Python module, the ML inference path, retries, and data artifacts end to end.

![Architecture/output sample](stress_assets/sloth_report_screenshot.png)

## The Lazy Payoff

From the latest sample run:

- 18 photos scanned
- 9 moved by rules
- 4 moved by Gemma
- 5 kept
- 1 journal entry written

In practice, that means I review a much smaller folder and still keep a written daily trace.

## Prize Categories

**Best Use of Gemma**  
Gemma handles captioning, keep/junk/needs_review decisions, and journal drafting with strict JSON validation and retry logic.

**Best Use of GitHub Copilot**  
Copilot accelerated module-by-module implementation and tests; CI runs those tests and a dry-run summary on each push.

**Best Use of Entire**  
The implementation process was structured to preserve traceability through run logs (`manifest.jsonl`, `decisions.jsonl`, `latest_run.json`) and reproducible command paths.
