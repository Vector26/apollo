"""Things the owner said that were cancelled before any answer was heard (they kept talking, so the
reply was barged in on). They are not lost: the next turn carries them, so every part gets answered."""
from __future__ import annotations

import time
from typing import Callable

MAX_ITEMS = 4
MAX_AGE_S = 90.0


class UnansweredCarry:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._items: list[tuple[str, float]] = []
        self._clock = clock

    def add(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._items = (self._items + [(text, self._clock())])[-MAX_ITEMS:]

    def pending(self) -> list[str]:
        now = self._clock()
        self._items = [(t, at) for t, at in self._items if now - at <= MAX_AGE_S]
        return [t for t, _ in self._items]

    def merged(self, text: str) -> str:
        """The owner's words for this turn, with anything still unanswered ahead of them, in order."""
        return " ".join([*self.pending(), text])

    def settle(self, taken: list[str]) -> None:
        """The turn that carried these was answered: drop exactly those (newer ones stay)."""
        n = len(taken)
        if n and [t for t, _ in self._items[:n]] == taken:
            self._items = self._items[n:]
        else:
            self._items = [(t, at) for t, at in self._items if t not in taken]
