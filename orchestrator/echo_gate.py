"""Lexical echo gate: drop a transcript that is just the assistant's own voice coming back.

Compares word pairs, not words: a reply like "what is the plan" shares plenty of single words
with what was just said, but almost none of the sequence. Real echo reproduces the sequence.
"""

from __future__ import annotations

import re
import time
from collections import deque
from typing import Callable

_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


def _pairs(words: list[str]) -> list[tuple[str, str]]:
    return list(zip(words, words[1:]))


class EchoGate:
    def __init__(self, window_s: float = 8.0, drop_at: float = 0.7, min_words: int = 4,
                 clock: Callable[[], float] = time.monotonic):
        self.window_s = window_s
        self.drop_at = drop_at
        self.min_words = min_words
        self._clock = clock
        self._heard: deque[tuple[float, set[tuple[str, str]]]] = deque(maxlen=64)

    def heard(self, text: str) -> None:
        """The assistant's audio for ``text`` just started playing."""
        pairs = set(_pairs(_words(text)))
        if pairs:
            self._heard.append((self._clock(), pairs))

    def overlap(self, text: str) -> float:
        """Share of the transcript's word pairs found in what was spoken within the window."""
        pairs = _pairs(_words(text))
        if not pairs:
            return 0.0
        cutoff = self._clock() - self.window_s
        spoken: set[tuple[str, str]] = set()
        for t, p in self._heard:
            if t >= cutoff:
                spoken |= p
        return sum(1 for p in pairs if p in spoken) / len(pairs)

    def is_echo(self, text: str) -> bool:
        if len(_words(text)) < self.min_words:
            return False  # "yes", "stop", "what?" are never discarded
        return self.overlap(text) >= self.drop_at
