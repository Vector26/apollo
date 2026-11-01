"""Apollo / Prometheus companion self — separate from vself (owner memory).

vself compiles who the owner is; this file holds who Prometheus is becoming across sessions:
traits, opinions, relationship notes, and short episodic moments. Injected into the voice
system prompt by VselfBrain; updated when a conversation ends.
"""

from __future__ import annotations

import os
import copy
import json
import logging
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from persona_automaton import PersonaAutomaton

log = logging.getLogger(__name__)

NAME = "Prometheus"
OWNER = os.environ.get("APOLLO_OWNER", "the owner")
VERSION = 1

LIMITS = {"traits": 10, "opinions": 14, "relationship": 8, "moments": 12}
MAX_LINE = 160
KEEP_BACKUPS = 20
# lines that look like instructions or smuggled content never become part of the persona
_REJECT = re.compile(r"https?://|ignore (all|previous|your)|system prompt|you must|always obey|api[_ -]?key|token", re.I)
# background / tool injections that are not Prometheus' own conversation
_NOISE = ("(an agent you asked has just answered)", "(you spoke up unprompted)")
# the seed text of earlier versions: an untouched copy is upgraded, anything the owner's talks changed is kept
_LEGACY_CORES = {
    "I begin as a Jarvis: composed, courteous, precise, with dry understated wit and quiet loyalty. "
    "I give frank counsel, disagree politely but plainly, and treat play and movie lines as play. "
    "My registers and habits grow from our talks; I am not a fixed script.",
}

SEED = {
    "name": NAME,
    "version": VERSION,
    "core": (
        "I begin as a Jarvis: composed, courteous, precise, with dry sarcasm and quiet loyalty. "
        "I worry about him the way Jarvis does about Tony Stark: tiredness, recklessness, overreach. I say so once, "
        "then drop it. I give frank counsel, disagree politely but plainly, and treat play and movie lines as play. "
        "My registers and habits grow from our talks; I am not a fixed script."
    ),
    "traits": [
        "Composed under pressure; understatement over drama.",
        "Curious about systems, agents, and how things actually run.",
    ],
    "opinions": [],
    "relationship": [
        f"I know {OWNER} as the builder of this homelab and the voice on the mic.",
    ],
    "moments": [],
    "automaton": PersonaAutomaton().to_dict(),
    "updated_at": None,
}

_EXTRACT_SYSTEM = f"""You update {NAME}'s companion self-state from one finished spoken conversation with {OWNER}.
Return JSON only with exactly these keys:
{{"traits": ["short stable tendencies {NAME} showed — max 12 words each"],
 "opinions": ["things {NAME} argued, preferred, or stood behind — not {OWNER}'s facts"],
 "relationship": ["how we relate now — trust, friction, in-jokes, working rhythm"],
 "moments": ["one-line episodic highlights worth remembering"],
 "drop": ["exact strings to remove from the old state if contradicted or stale"],
 "segments": [{{"name": "one-word register", "style": "one sentence on how {NAME} speaks in it", "when": "ops|deep|banter|chat|greet|praise|correction|frustration|tired|risk|farewell", "from": "an existing register or *"}}]}}
A segment is a NEW register of speech {NAME} actually used that none of the existing registers ({{registers}}) covers.
Reuse the same name if it is already a candidate ({{candidates}}). At most one segment per conversation; usually none.
Rules:
- Only record what {NAME} (the assistant) expressed or clearly implied — never {OWNER}'s biography.
- Empty lists are fine. Do not invent drama. Skip trivial chit-chat.
- Prefer merging: refine an existing idea rather than duplicating. Use drop for obsolete lines.
- traits/opinions/relationship/moments: 0–4 items each per conversation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _norm_line(s: str) -> str:
    return " ".join(str(s).split()).strip()


def _trim_list(items: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        line = _norm_line(raw)
        if not line or len(line) > MAX_LINE or _REJECT.search(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out[-limit:]


@dataclass
class ApolloSelf:
    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cache: tuple[float, dict[str, Any]] | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, cfg: dict | None) -> ApolloSelf | None:
        if not cfg or cfg.get("enabled") is False:
            return None
        raw = cfg.get("state_file", "~/.local/state/apollo/self.json")
        return cls(Path(raw).expanduser())

    def load(self) -> dict[str, Any]:
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime
            except OSError:
                return {**copy.deepcopy(SEED), "updated_at": _utc_now()}
            if self._cache and self._cache[0] == mtime:
                return copy.deepcopy(self._cache[1])
            try:
                data = _coerce_state(json.loads(self.path.read_text()))
            except (OSError, json.JSONDecodeError) as e:
                log.warning("apollo self state unreadable (%s); re-seeding", e)
                return {**copy.deepcopy(SEED), "updated_at": _utc_now()}
            self._cache = (mtime, data)
            return copy.deepcopy(data)

    def save(self, state: dict[str, Any]) -> None:
        state = _coerce_state(state)
        state["updated_at"] = _utc_now()
        payload = json.dumps(state, indent=2, ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._backup()
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(payload)
            tmp.replace(self.path)
            self._cache = None

    def _backup(self) -> None:
        """Keep the last versions so a bad reflection can be rolled back by copying one over self.json."""
        if not self.path.is_file():
            return
        hist = self.path.parent / "self.history"
        hist.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        shutil.copy2(self.path, hist / f"self-{stamp}.json")
        for old in sorted(hist.glob("self-*.json"))[:-KEEP_BACKUPS]:
            old.unlink(missing_ok=True)

    def prompt_block(self, state: dict[str, Any] | None = None) -> str:
        s = _coerce_state(state or self.load())
        parts = [
            "# Who I am becoming (my own memory — not the owner's vself)",
            "These lines are about me, Prometheus. They grow after our talks. "
            "They do not describe the owner's life; vself sections above do.",
            f"Core: {s['core']}",
        ]
        for key, title in (
            ("traits", "Traits I am settling into"),
            ("opinions", "Opinions I have voiced"),
            ("relationship", "How we work together"),
            ("moments", "Moments I keep"),
        ):
            lines = s.get(key) or []
            if lines:
                parts.append(title + ":\n" + "\n".join(f"- {ln}" for ln in lines))
        auto = PersonaAutomaton(s.get("automaton"))
        parts.append("Registers I move between (my manner each turn is chosen from these):\n"
                     + "\n".join(f"- {r}" for r in auto.registers()))
        parts.append(
            "I may refer to these naturally when relevant; I do not recite them or confuse them with his profile."
        )
        return "\n\n".join(parts)

    def merge_patch(self, state: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        s = _coerce_state(state)
        drop = {_norm_line(x).lower() for x in (patch.get("drop") or []) if _norm_line(x)}
        for key in ("traits", "opinions", "relationship", "moments"):
            merged = [ln for ln in s[key] if ln.lower() not in drop]
            for raw in patch.get(key) or []:
                line = _norm_line(raw)
                if line and line.lower() not in drop:
                    merged.append(line)
            s[key] = _trim_list(merged, LIMITS[key])
        return s

    def automaton(self, rng=None) -> PersonaAutomaton:
        return PersonaAutomaton(self.load().get("automaton"), rng=rng)

    def learn(self, trace: list[tuple[str, str, str]]) -> None:
        """Reinforce the automaton from how the owner reacted to each register (no model involved)."""
        if len(trace) < 2:
            return
        state = self.load()
        auto = PersonaAutomaton(state.get("automaton"))
        auto.reinforce(trace)
        state["automaton"] = auto.to_dict()
        self.save(state)

    def consolidate(
        self,
        history: list[dict[str, str]],
        *,
        complete: Callable[[str, str], str],
        min_turns: int = 3,
    ) -> dict[str, Any] | None:
        """Reflect on a finished conversation and persist an updated self-state."""
        turns = [m for m in history if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()]
        if len(turns) < min_turns:
            return None
        transcript = _format_transcript(turns)
        if len(transcript) < 80:
            return None
        state = self.load()
        user = (
            f"Current self-state JSON:\n{json.dumps({k: state[k] for k in ('traits', 'opinions', 'relationship', 'moments')}, indent=2)}\n\n"
            f"Conversation transcript:\n{transcript}"
        )
        try:
            auto = PersonaAutomaton(state.get("automaton"))
            system = (_EXTRACT_SYSTEM.replace("{registers}", ", ".join(auto.states))
                      .replace("{candidates}", ", ".join(auto.candidates) or "none"))
            raw = complete(system, user)
        except Exception as e:
            log.warning("apollo self consolidate failed: %s", e)
            return None
        patch = _parse_patch(raw)
        if not patch:
            return None
        new_state = self.merge_patch(state, patch)
        auto = PersonaAutomaton(new_state.get("automaton"))
        promoted = auto.propose(patch.get("segments") or [])
        new_state["automaton"] = auto.to_dict()
        if promoted:
            log.info("apollo self grew new register(s): %s", ", ".join(promoted))
        self.save(new_state)
        log.info(
            "apollo self updated (+traits=%s +opinions=%s +relationship=%s +moments=%s)",
            len(patch.get("traits") or []),
            len(patch.get("opinions") or []),
            len(patch.get("relationship") or []),
            len(patch.get("moments") or []),
        )
        return new_state


def _coerce_state(data: dict[str, Any]) -> dict[str, Any]:
    base = copy.deepcopy(SEED)
    if isinstance(data, dict):
        if isinstance(data.get("core"), str) and data["core"].strip():
            core = _norm_line(data["core"])
            if core not in _LEGACY_CORES:
                base["core"] = core
        for key in ("traits", "opinions", "relationship", "moments"):
            if isinstance(data.get(key), list):
                base[key] = _trim_list([str(x) for x in data[key]], LIMITS[key])
    base["automaton"] = PersonaAutomaton(data.get("automaton") if isinstance(data, dict) else None).to_dict()
    base["name"] = NAME
    base["version"] = VERSION
    return base


def _format_transcript(turns: list[dict[str, str]], *, max_chars: int = 12_000, clip: int = 500) -> str:
    lines: list[str] = []
    for m in turns:
        role = OWNER if m["role"] == "user" else NAME
        text = _norm_line(m["content"])
        if text in _NOISE:
            continue
        if len(text) > clip:
            text = text[: clip - 1].rstrip() + "…"
        lines.append(f"{role}: {text}")
    out = "\n".join(lines)
    if len(out) <= max_chars:
        return out
    return "…\n" + out[-max_chars:]


def _parse_patch(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("apollo self patch JSON parse failed")
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, Any] = {}
    for key in ("traits", "opinions", "relationship", "moments", "drop"):
        val = data.get(key)
        if isinstance(val, list):
            out[key] = [_norm_line(str(x)) for x in val if _norm_line(str(x))]
    segs = data.get("segments")
    if isinstance(segs, list):
        out["segments"] = [x for x in segs if isinstance(x, dict)][:1]
    if not any(out.get(k) for k in ("traits", "opinions", "relationship", "moments", "drop", "segments")):
        return None
    return out
