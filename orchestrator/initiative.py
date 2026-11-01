"""Jarvis speaking up unprompted: a very small decision maker plus the guards on what it says.

Nothing here talks to a model. ``Initiative.decide`` answers one question — "should Apollo even think
about saying something right now?" — and ``clean_aside`` judges what the model came back with.

Hard gates (every tick, no dice): switched on, a cloud brain (Composer / OpenRouter) is the one answering,
a conversation is live, the floor is free, not halted, the owner has been quiet for a while but not so long
that it is a new conversation, enough time since the last aside, under the hourly budget.

Then one weighted roll per quiet spell, never one per tick, so waiting longer does not make it certain.
The weight rises with signs the owner could use a word (late hour, long session, tired, a risky request) and
the tone follows the same signs: after a bad moment it is steady and kind, never sarcastic.
"""

from __future__ import annotations

import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

TONES = ("wry", "observant", "solicitous", "steady")
NEGATIVE = frozenset({"correction", "frustration"})

TONE_BRIEF = {
    "wry": "a dry, sarcastic aside in Jarvis's manner: understatement and mock-formality, aimed at the situation or "
           "at his own choices, never cruel and never at him personally",
    "observant": "one sharp observation or reminder about what he is doing, delivered dryly",
    "solicitous": "real concern for him, the way Jarvis worries about Tony Stark: tiredness, overreach, risk. Say it "
                  "once, plainly, with one practical suggestion; a touch of dry humour is fine, nagging is not",
    "steady": "calm, kind and brief. No sarcasm at all: he is not in the mood for it",
}

_NONE = re.compile(r"^\W*none\W*$", re.I)
_BAD = re.compile(r"https?://|[\[\]()#*`_<>{}|\\]|^[-•–]\s|ignore (all|previous|your)|system prompt|api[_ -]?key|token|"
                  r"^(as an ai|i'?m sorry|i cannot|i can'?t)\b", re.I)
_WORD = re.compile(r"[a-z0-9']+")
MAX_WORDS = 40
MAX_CHARS = 260


@dataclass(frozen=True)
class Context:
    cloud: bool                  # a cloud route (Composer / OpenRouter) answered last, so a cloud call is already in play
    in_session: bool             # a conversation is live
    floor_free: bool             # nobody speaking, nothing playing, no sentence pending
    halted: bool
    idle_s: float                # quiet since the last exchange
    session_s: float             # how long this conversation has run (0 = none)
    hour: int                    # local hour, 0-23
    symbols: Sequence[str]       # the owner's recent moods/intents, oldest first (persona_automaton symbols)


@dataclass(frozen=True)
class Decision:
    ask: bool
    tone: str = "wry"
    p: float = 0.0
    why: str = ""


NO = Decision(False)


class Initiative:
    def __init__(self, cfg: Optional[Mapping] = None, rng: Optional[random.Random] = None,
                 clock: Callable[[], float] = time.monotonic):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.min_idle_s = float(cfg.get("min_idle_s", 15))
        self.max_idle_s = float(cfg.get("max_idle_s", 240))
        self.min_gap_s = float(cfg.get("min_gap_s", 240))
        self.max_per_hour = int(cfg.get("max_per_hour", 3))
        self.roll_every_s = float(cfg.get("roll_every_s", 60))
        self.base_p = float(cfg.get("base_p", 0.12))
        self.after_none_s = float(cfg.get("after_none_s", 150))
        self.rng = rng or random.Random()
        self._clock = clock
        self._next_roll = 0.0
        self._spoke: deque[float] = deque(maxlen=32)
        self.recent: deque[str] = deque(maxlen=6)

    # ── the decision ───────────────────────────────────────────────────
    def decide(self, ctx: Context) -> Decision:
        now = self._clock()
        if not (self.enabled and ctx.cloud and ctx.in_session and ctx.floor_free and not ctx.halted):
            return NO
        if not self.min_idle_s <= ctx.idle_s <= self.max_idle_s:
            return NO
        if ctx.symbols and ctx.symbols[-1] == "farewell":
            return NO
        if self._spoke and now - self._spoke[-1] < self.min_gap_s:
            return NO
        if sum(1 for t in self._spoke if now - t < 3600) >= self.max_per_hour:
            return NO
        if now < self._next_roll:
            return NO
        self._next_roll = now + self.roll_every_s * (0.5 + self.rng.random())  # one roll per quiet spell
        p, tone, why = self.weigh(ctx)
        return Decision(self.rng.random() < p, tone, p, why)

    def weigh(self, ctx: Context) -> tuple[float, str, str]:
        recent = list(ctx.symbols)[-3:]
        reasons: list[str] = []
        concern = 0.0
        if ctx.hour >= 23 or ctx.hour < 5:
            concern += 0.25
            reasons.append("late")
        if ctx.session_s > 5400:
            concern += 0.15
            reasons.append("long session")
        if "tired" in recent:
            concern += 0.35
            reasons.append("tired")
        if "risk" in recent:
            concern += 0.35
            reasons.append("risky request")
        p = self.base_p + concern + 0.15 * min(1.0, ctx.idle_s / 120.0)
        if any(s in NEGATIVE for s in recent[-2:]):
            tone = "steady"
            reasons.append("a bad moment")
        elif concern >= 0.25:
            tone = "solicitous"
        else:
            tone = "wry" if self.rng.random() < 0.6 else "observant"
        return min(0.9, p), tone, ", ".join(reasons) or "idle"

    # ── bookkeeping ────────────────────────────────────────────────────
    def spoke(self, line: str) -> None:
        self._spoke.append(self._clock())
        self.recent.append(line)

    def nothing_to_say(self) -> None:
        """The model had nothing worth saying (or it was cut): do not ask again for a while."""
        self._next_roll = max(self._next_roll, self._clock() + self.after_none_s)


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def clean_aside(raw: Optional[str], recent: Sequence[str] = ()) -> Optional[str]:
    """The line to speak, or None: the model said NONE, rambled, used markup, sounded like a bot, or repeated itself."""
    if not raw:
        return None
    text = " ".join(raw.split()).strip().strip("\"'“”‘’").strip()
    if not text or _NONE.match(text) or _BAD.search(text):
        return None
    if len(text) > MAX_CHARS or len(text.split()) > MAX_WORDS:
        return None  # a spoken aside is a line, not a speech; trimming would cut a thought in half
    mine = _words(text)
    for old in recent:
        theirs = _words(old)
        if mine and theirs and len(mine & theirs) / len(mine | theirs) >= 0.6:
            return None
    return text


def aside_prompts(tone: str, persona: str, owner: str, facts: str, exchange: str) -> tuple[str, str]:
    """(system, user) for the one cloud call. NONE is the expected answer most of the time."""
    system = (
        f"{persona}\n\nYou are the voice assistant of {owner}, in the manner of Jarvis with Tony Stark: loyal, composed, "
        "dryly sarcastic, and genuinely concerned for him. You hear everything said in this conversation."
    )
    user = (
        f"{facts}\n\nThe last exchange:\n{exchange}\n\n"
        f"You have the floor for one line {owner} did not ask for. Tone: {TONE_BRIEF.get(tone, TONE_BRIEF['wry'])}.\n"
        "Speak only if you truly have something worth saying given the exchange and the facts above. "
        f"Reply with that one line alone (at most {MAX_WORDS - 10} words, spoken English, no markdown, no emoji, "
        "no stage directions, no question that needs a long answer), or with exactly NONE. "
        "NONE is the right answer most of the time."
    )
    return system, user
