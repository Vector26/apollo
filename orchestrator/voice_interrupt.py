"""Spoken phrases that stop playback and return to listening (without full open-mic barge-in)."""

from __future__ import annotations

import re

_PHRASES = (
    r"wait(?:\s+wait)?",
    r"stop(?:\s+(?:talking|it|please))?",
    r"hold on|hold up|hang on",
    r"pause",
    r"listen(?:\s+to\s+me|\s+up)?",
    r"one sec(?:ond)?",
    r"enough|quiet|shush",
    r"let me (?:speak|talk|say (?:something|this))",
    r"ok(?:ay)?\s+stop",
    r"just stop",
    r"can you stop",
    r"stop for a (?:sec|second|minute)",
)

# Must be the whole utterance (partials are short; avoids "wait for the build").
_INTERRUPT = re.compile(r"^\s*(?:" + "|".join(_PHRASES) + r")\s*[.!?]?\s*$", re.I)


def is_interrupt_command(text: str) -> bool:
    if not text or not text.strip():
        return False
    t = re.sub(r"\s+", " ", text.lower().strip())
    return bool(_INTERRUPT.match(t))
