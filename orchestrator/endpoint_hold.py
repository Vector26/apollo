"""Turn-taking helpers: don't answer half a sentence, don't answer a throat-clear.

Whisper hands over whatever was said before the pause. "I just" or "Um..." is not
a turn: the first is held for the continuation, the second is dropped.
"""

from __future__ import annotations

import re
import time

# Words that cannot end a thought. Kept conservative: a wrong hold costs the user a wait.
DANGLING = frozenset({
    "and", "but", "or", "so", "because", "if", "to", "the", "a", "an", "of", "with",
    "my", "your", "just", "like", "um", "uh", "uhm", "er", "erm", "hmm",
})
FILLERS = frozenset({"um", "uh", "uhm", "er", "erm", "hmm", "mm", "mhm", "ah", "oh", "huh"})
_WORD = re.compile(r"[a-z']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("\u2019", "'"))


def is_filler_only(text: str) -> bool:
    """True for "Um...", "Uh, hmm." and the like: nothing to answer."""
    words = _words(text)
    return bool(words) and all(w in FILLERS for w in words)


def ends_dangling(text: str) -> bool:
    """True when the utterance stops on a word that needs a continuation."""
    stripped = text.strip()
    if not stripped or stripped.endswith(("?", "!")) or is_filler_only(stripped):
        return False
    words = _words(stripped)
    return bool(words) and words[-1] in DANGLING


class FragmentHold:
    """Holds one unfinished utterance so the next one continues it.

    ``peek_merge`` is read-only (a speculative turn may be redone for the same audio);
    ``clear`` is called once the merged turn is confirmed.
    """

    def __init__(self, window_s: float = 2.5, clock=time.monotonic):
        self.window_s = window_s
        self._clock = clock
        self._text = ""
        self._stamp = 0.0
        self._seq = 0

    @property
    def pending(self) -> bool:
        return bool(self._text)

    @property
    def text(self) -> str:
        return self._text

    @property
    def seq(self) -> int:
        return self._seq

    def hold(self, text: str) -> int:
        self._text = text.strip()
        self._stamp = self._clock()
        self._seq += 1
        return self._seq

    def peek_merge(self, text: str) -> str:
        if not self._text:
            return text
        return f"{self._text.rstrip(' .…,-—')} {text.strip()}".strip()

    def expired(self) -> bool:
        return self.pending and self._clock() - self._stamp >= self.window_s

    def take(self) -> str:
        text, self._text = self._text, ""
        return text

    def clear(self) -> None:
        self._text = ""
