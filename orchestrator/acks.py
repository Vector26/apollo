"""Intent-keyed acknowledgement lines, spoken only while the brain is still silent.

The line comes from the same classifier the personality automaton uses, so a question that
will take a while ("check the GPU", "think about the design") gets a fitting, honest line
and small talk gets none. Lines are pre-rendered once at startup: playing one costs no GPU.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from persona_automaton import classify

DEFAULT_LINES: dict[str, tuple[str, ...]] = {
    "ops": ("Checking.", "Looking into it.", "On it."),
    "deep": ("Let me think that through.", "Considering that."),
}


class Acks:
    def __init__(self, cfg: Mapping | None = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.after_s = float(cfg.get("after_ms", 1000)) / 1000.0
        self._lines: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in DEFAULT_LINES.items()}
        for symbol, lines in (cfg.get("lines") or {}).items():
            self._lines[str(symbol)] = tuple(str(x) for x in lines if str(x).strip())
        self._next: dict[str, int] = {}

    def all_lines(self) -> tuple[str, ...]:
        """Every line that may be spoken (for pre-rendering)."""
        seen: dict[str, None] = {}
        for lines in self._lines.values():
            for line in lines:
                seen.setdefault(line)
        return tuple(seen)

    def pick(self, user_text: str | None) -> str | None:
        """The line for this utterance, or None when it needs none. Rotates so a line never repeats back to back."""
        if not self.enabled or not user_text:
            return None
        symbol = classify(user_text)
        lines: Sequence[str] = self._lines.get(symbol, ())
        if not lines:
            return None
        i = self._next.get(symbol, 0) % len(lines)
        self._next[symbol] = i + 1
        return lines[i]
