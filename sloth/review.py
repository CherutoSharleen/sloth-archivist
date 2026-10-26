from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .gemma import GemmaClient, GemmaResult

PROMPT_VERSION = "review_v1"


@dataclass(frozen=True)
class DecisionRecord:
    run_id: str
    file: str
    stage: str
    verdict: str
    reason: str
    caption: str
    model: str
    prompt_version: str
    seconds: float


def load_review_prompt(prompt_version: str = PROMPT_VERSION) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / f"{prompt_version}.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


def review_survivors(
    photo_paths: list[Path],
    model_name: str,
    image_size: int,
    prompt_text: str,
    run_id: str,
    prompt_version: str = PROMPT_VERSION,
    gemma_client: GemmaClient | None = None,
) -> list[DecisionRecord]:
    if gemma_client is None and os.getenv("SLOTH_FAKE_GEMMA") == "1":
        gemma_client = _HeuristicGemma()
    client = gemma_client or GemmaClient(model_name=model_name, image_size=image_size)
    decisions: list[DecisionRecord] = []

    for photo_path in photo_paths:
        reviewed = client.review_image(photo_path, prompt_text)
        decisions.append(
            DecisionRecord(
                run_id=run_id,
                file=str(photo_path),
                stage="gemma",
                verdict=reviewed.verdict,
                reason=reviewed.reason,
                caption=reviewed.caption,
                model=model_name,
                prompt_version=prompt_version,
                seconds=reviewed.seconds,
            )
        )
    return decisions


def append_decisions(decisions_path: Path, rows: list[DecisionRecord]) -> None:
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.__dict__, ensure_ascii=True) + "\n")


class _HeuristicGemma:
    def review_image(self, image_path: Path, prompt: str) -> GemmaResult:  # noqa: ARG002
        name = image_path.name.lower()
        if "screenshot" in name or "otp" in name or "meme" in name:
            return GemmaResult(
                caption="likely throwaway screenshot",
                verdict="junk",
                reason="throwaway_screenshot",
                confidence=0.7,
                seconds=0.0,
                call_seconds=[0.0],
            )
        return GemmaResult(
            caption="photo with potentially useful context",
            verdict="keep",
            reason="useful",
            confidence=0.7,
            seconds=0.0,
            call_seconds=[0.0],
        )
