from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field, ValidationError


class ReviewDecision(BaseModel):
    caption: str
    verdict: Literal["keep", "junk", "needs_review"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class GemmaResult:
    caption: str
    verdict: Literal["keep", "junk", "needs_review"]
    reason: str
    confidence: float
    seconds: float
    call_seconds: list[float]


class GemmaClient:
    def __init__(
        self,
        model_name: str,
        image_size: int,
        max_retries: int = 2,
        host: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.image_size = image_size
        self.max_retries = max_retries
        self.host = host or os.getenv("OLLAMA_HOST")

    def review_image(self, image_path: Path, prompt: str) -> GemmaResult:
        image_bytes = _load_resized_image_bytes(image_path, self.image_size)
        call_seconds: list[float] = []
        schema = ReviewDecision.model_json_schema()

        for _attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = _call_ollama_chat(
                    model=self.model_name,
                    message=prompt,
                    image_bytes=image_bytes,
                    schema=schema,
                    host=self.host,
                )
                decision = _parse_response(response)
                elapsed = time.perf_counter() - started
                call_seconds.append(elapsed)
                return GemmaResult(
                    caption=decision.caption,
                    verdict=decision.verdict,
                    reason=decision.reason,
                    confidence=decision.confidence,
                    seconds=sum(call_seconds),
                    call_seconds=call_seconds,
                )
            except (ValidationError, ValueError, RuntimeError):
                elapsed = time.perf_counter() - started
                call_seconds.append(elapsed)
                continue

        return GemmaResult(
            caption="",
            verdict="needs_review",
            reason="invalid_model_output",
            confidence=0.0,
            seconds=sum(call_seconds),
            call_seconds=call_seconds,
        )


class FakeGemma:
    def __init__(self, responses: list[ReviewDecision | dict[str, Any]] | None = None) -> None:
        self._responses = responses or [
            ReviewDecision(
                caption="",
                verdict="needs_review",
                reason="fake-default",
                confidence=0.0,
            )
        ]
        self._index = 0

    def review_image(self, image_path: Path, prompt: str) -> GemmaResult:  # noqa: ARG002
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1

        if isinstance(response, ReviewDecision):
            decision = response
        else:
            decision = ReviewDecision.model_validate(response)

        return GemmaResult(
            caption=decision.caption,
            verdict=decision.verdict,
            reason=decision.reason,
            confidence=decision.confidence,
            seconds=0.0,
            call_seconds=[0.0],
        )


def _load_resized_image_bytes(image_path: Path, image_size: int) -> bytes:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        resized = rgb.resize((image_size, image_size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _call_ollama_chat(
    model: str,
    message: str,
    image_bytes: bytes,
    schema: dict[str, Any],
    host: str | None,
) -> dict[str, Any]:
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("ollama package is required") from exc

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": message, "images": [image_bytes]}],
        "options": {"temperature": 0},
        "format": schema,
    }
    if host:
        kwargs["host"] = host
    def _run() -> Any:
        return ollama.chat(**kwargs)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            response = future.result(timeout=20)
        except TimeoutError as exc:
            raise RuntimeError("ollama chat timed out") from exc
    if not isinstance(response, dict):
        raise ValueError("Unexpected ollama response type")
    return response


def _parse_response(response: dict[str, Any]) -> ReviewDecision:
    message = response.get("message")
    if not isinstance(message, dict):
        raise ValueError("Missing message in response")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Missing message content")
    return ReviewDecision.model_validate_json(content)
