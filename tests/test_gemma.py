from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from sloth.gemma import FakeGemma, GemmaClient


def _write_image(path: Path) -> None:
    Image.new("RGB", (200, 120), color=(128, 100, 90)).save(path)


def test_gemma_client_uses_env_host_and_validates_response(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "photo.jpg"
    _write_image(image_path)

    captured: dict[str, object] = {}

    def fake_call(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {
            "message": {
                "content": json.dumps(
                    {
                        "caption": "a room",
                        "verdict": "keep",
                        "reason": "clear and useful",
                        "confidence": 0.92,
                    }
                )
            }
        }

    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setattr("sloth.gemma._call_ollama_chat", fake_call)

    client = GemmaClient(model_name="gemma3:4b", image_size=768)
    result = client.review_image(image_path=image_path, prompt="review image")

    assert result.verdict == "keep"
    assert result.reason == "clear and useful"
    assert result.confidence == 0.92
    assert len(result.call_seconds) == 1
    assert result.seconds >= 0.0

    assert captured["host"] == "http://127.0.0.1:11434"
    assert captured["model"] == "gemma3:4b"
    assert captured["message"] == "review image"
    assert captured["schema"] is not None
    assert isinstance(captured["image_bytes"], bytes)


def test_gemma_client_retries_and_falls_back_to_needs_review(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "photo.jpg"
    _write_image(image_path)

    attempts = {"count": 0}

    def always_invalid(**kwargs):  # noqa: ANN003, ARG001
        attempts["count"] += 1
        return {"message": {"content": '{"not":"expected-shape"}'}}

    monkeypatch.setattr("sloth.gemma._call_ollama_chat", always_invalid)

    client = GemmaClient(model_name="gemma3:4b", image_size=768, max_retries=2)
    result = client.review_image(image_path=image_path, prompt="review image")

    assert attempts["count"] == 3
    assert result.verdict == "needs_review"
    assert result.reason == "invalid_model_output"
    assert result.confidence == 0.0
    assert len(result.call_seconds) == 3
    assert result.seconds >= 0.0


def test_fake_gemma_returns_canned_responses(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    _write_image(image_path)

    fake = FakeGemma(
        responses=[
            {
                "caption": "desk",
                "verdict": "junk",
                "reason": "duplicate",
                "confidence": 0.8,
            },
            {
                "caption": "street",
                "verdict": "keep",
                "reason": "memorable",
                "confidence": 0.9,
            },
        ]
    )

    first = fake.review_image(image_path, "p1")
    second = fake.review_image(image_path, "p2")

    assert first.verdict == "junk"
    assert first.reason == "duplicate"
    assert second.verdict == "keep"
    assert second.reason == "memorable"
