from __future__ import annotations

import re
from dataclasses import dataclass

ALLOW_WORDS = {
    "a",
    "an",
    "and",
    "in",
    "the",
    "i",
    "me",
    "my",
    "we",
    "our",
    "was",
    "were",
    "had",
    "am",
    "is",
    "are",
    "felt",
    "feel",
    "feeling",
    "enjoyed",
    "liked",
    "love",
    "happy",
    "calm",
    "quiet",
    "morning",
    "afternoon",
    "evening",
    "night",
    "today",
    "yesterday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


@dataclass(frozen=True)
class FactCheckResult:
    ok: bool
    unsupported_words: list[str]


def check_entry(text: str, facts: str, extra_allow: set[str] | None = None) -> FactCheckResult:
    """Check whether content words in text are supported by supplied facts."""
    allow = {_stem(word) for word in ALLOW_WORDS}
    if extra_allow:
        allow.update(_stem(w) for w in extra_allow)

    fact_words = {_stem(w) for w in _words(facts)}
    fact_numbers = set(_numbers(facts))
    text_numbers = set(_numbers(text))

    unsupported: list[str] = []
    for raw in _words(text):
        stemmed = _stem(raw)
        if not stemmed:
            continue
        if stemmed in allow:
            continue
        if stemmed in fact_words:
            continue
        unsupported.append(raw.lower())

    if text_numbers - fact_numbers:
        unsupported.extend(sorted(text_numbers - fact_numbers))

    unsupported_sorted = sorted(set(unsupported))
    return FactCheckResult(ok=not unsupported_sorted, unsupported_words=unsupported_sorted)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())


def _numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+\b", text)


def _stem(word: str) -> str:
    value = word.lower()
    for suffix in ("ingly", "edly", "ing", "edly", "ed", "ly", "es", "s"):
        if value.endswith(suffix) and len(value) > len(suffix) + 2:
            value = value[: -len(suffix)]
            break
    return value
