"""In-process TTS with clause-level streaming (facade over voice pipeline).

Replaces the Voicebox /speak HTTP round trip on the live path. The LLM token
stream is split into clauses; each clause is synthesized as soon as it exists.
"""

from apollo_home import apollo_path
import logging
import re
from typing import Generator, Optional, Tuple

import numpy as np

from voice.pipeline import VoicePipeline

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = apollo_path("models/chatterbox-turbo")
DEFAULT_REF_AUDIO = apollo_path("models/refs/default.wav")

_HARD_STOP = re.compile(r"[.!?][\"')\]]*(\s|$)")
_SOFT_STOP = re.compile(r"[,;:—–-][\"')\]]*(\s|$)")
_SOFT_MIN_WORDS = 4
_FIRST_SOFT_MIN_WORDS = 2  # first clause of a reply: short = sooner first audio
_FORCE_WORDS = 10
_FORCE_MAX_WORDS = 14  # when the tenth word cannot end a clause, wait for a better stop (never past this)
_BREAK_BEFORE = frozenset({
    "and", "but", "or", "so", "because", "which", "that", "while", "when", "where",
    "with", "then", "since", "although", "though", "if",
})
# a clause that ends on one of these sounds cut off: extend it instead
_NO_END = frozenset({
    "and", "but", "or", "so", "because", "if", "to", "the", "a", "an", "of", "with", "my", "your",
    "in", "on", "at", "for", "that", "which", "is", "are", "was", "be", "as", "by", "from", "into",
    "about", "than", "then", "i", "you", "it", "just", "like",
})


def _bare(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


class ClauseSplitter:
    """Feed LLM tokens; get clauses out as early as speakable."""

    def __init__(self):
        self._buf = ""
        self._emitted = False

    def feed(self, token: str) -> Optional[str]:
        self._buf += token
        m = _HARD_STOP.search(self._buf)
        if m:
            return self._take_upto(m.end())
        min_words = _SOFT_MIN_WORDS if self._emitted else _FIRST_SOFT_MIN_WORDS
        for m in _SOFT_STOP.finditer(self._buf):
            if len(self._buf[: m.end()].split()) >= min_words:
                return self._take_upto(m.end())
        return self._forced()

    def _forced(self) -> Optional[str]:
        """No stop in sight: cut at a natural boundary, not blindly at a word count.

        Only words already followed by whitespace count: the last one may still be arriving.
        """
        words = [m for m in re.finditer(r"\S+\s+", self._buf)]
        if len(words) < _FORCE_WORDS:
            return None
        window = words[:_FORCE_MAX_WORDS]
        for i in range(len(window) - 1, 3, -1):  # latest conjunction/preposition that leaves 4+ words
            if _bare(window[i].group()) in _BREAK_BEFORE:
                return self._take_upto(window[i - 1].end())
        for n in range(_FORCE_WORDS, len(window) + 1):
            if _bare(window[n - 1].group()) not in _NO_END:
                return self._take_upto(window[n - 1].end())
        if len(window) >= _FORCE_MAX_WORDS:
            return self._take_upto(window[-1].end())
        return None

    def flush(self) -> Optional[str]:
        return self._take_upto(len(self._buf))

    def _take_upto(self, end: int) -> Optional[str]:
        clause = self._buf[:end].strip()
        self._buf = self._buf[end:]
        if not re.search(r"\w", clause):
            return None  # bare punctuation left after a forced split: nothing to say
        self._emitted = True
        return clause


class ChatterboxEngine:
    """Backward-compatible wrapper around VoicePipeline."""

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        ref_audio: str = DEFAULT_REF_AUDIO,
        device: str = "auto",
        profile: str = "jarvis",
        tts_cfg: Optional[dict] = None,
    ):
        cfg = dict(tts_cfg or {})
        cfg.setdefault("model_dir", model_dir)
        cfg.setdefault("device", device)
        cfg.setdefault("ref_audio", ref_audio)
        self._pipeline = VoicePipeline(profile, tts_cfg=cfg)
        self.device = self._pipeline.device
        self.ref_audio = ref_audio
        self.sample_rate = self._pipeline.sample_rate
        self.display_name = self._pipeline.display_name
        self.profile_id = profile
        self._phrases: dict[str, Tuple[np.ndarray, int]] = {}

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        return self._pipeline.synthesize(text)

    def prerender(self, lines) -> None:
        """Synthesize fixed lines once (also loads every lazy model)."""
        for text in lines:
            self._phrases[text] = self._pipeline.synthesize(text)

    def phrase(self, text: str) -> Tuple[np.ndarray, int]:
        """Pre-rendered audio for a fixed line; synthesized and kept on first use."""
        if text not in self._phrases:
            self._phrases[text] = self._pipeline.synthesize(text)
        return self._phrases[text]


def split_token_stream(tokens) -> Generator[str, None, None]:
    splitter = ClauseSplitter()
    for token in tokens:
        clause = splitter.feed(token)
        if clause:
            yield clause
    tail = splitter.flush()
    if tail:
        yield tail
