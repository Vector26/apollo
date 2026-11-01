"""Prometheus' temperament as a growing weighted automaton (Jarvis seed).

  Σ  input symbols      what the owner just did (ops request, banter, correction, ...), a cheap regex
                        classifier: no model call, so it costs nothing on the voice path
  Q  states (registers) how Prometheus speaks next; seeded with six Jarvis registers, and it grows
  δ  weighted rows      "state|symbol" -> {next state: weight}; "*|symbol" is the default row. A row for a
                        specific state is created the first time experience says it should differ (specialisation)
  learning              reinforcement from the owner's reaction (praise +, correction/frustration -), no LLM
  growth                reflection may propose a new register; it is only promoted once proposed in two
                        separate conversations (a candidate), which keeps the personality from drifting on one
                        odd chat. Bounded: MAX_STATES / MAX_ROWS / MAX_CANDIDATES.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

START = "composed"
MAX_STATES = 14
MAX_ROWS = 90
MAX_CANDIDATES = 8
MAX_STYLE = 160
W_MAX = 20.0
SEED_FLOOR = 0.5

# first match wins
SYMBOLS: dict[str, re.Pattern] = {k: re.compile(v, re.I) for k, v in {
    "farewell": r"\b(bye|goodbye|good night|see you|that'?s all|talk later|signing off)\b",
    "correction": r"(that'?s|thats) (not|wrong)|not what (i|you)|\bi (said|meant)\b|\bwrong\b|\bincorrect\b|misunderst|mix-?up",
    "frustration": r"\b(wtf|worse|useless|terrible|stupid|retard\w*|annoying|ugh|garbage|hate|broken|not working|doesn'?t work|bad)\b",
    "praise": r"\b(thanks|thank you|good job|nice|great|perfect|awesome|well done|love it|brilliant)\b",
    "greet": r"^\W*(hi|hey|hello|yo|good (morning|evening|afternoon)|what'?s up|whats up)\b",
    "tired": r"\b(tired|exhausted|sleepy|haven'?t slept|no sleep|all[- ]?nighter|long day|burn(?:ed|t)? out|headache|stressed|overwhelmed|drained)\b",
    "risk": r"\b(force[- ]?push|rm -rf|wipe everything|delete everything|drop (?:the )?(?:table|database)|yolo|just ship it|push (?:it )?(?:straight )?to (?:prod|production|main|master)|deploy (?:straight )?to prod\w*|no backups?|without (?:a )?backups?|skip (?:the )?tests?)\b",
    "ops": r"\b(check|status|run|restart|start|stop|gpu|vram|disk|container|docker|logs?|deploy|build|memory|ram|cpu|ping|show me|list|how (much|many)|ask \w+|agent)\b",
    "deep": r"\b(think|plan|design|idea|why|should we|opinion|trade-?off|approach|architecture|what if|strategy|explain|obsess\w*)\b",
    "banter": r"\b(lol|haha|joke|funny|kidding|roast|bro|dude|lmao)\b",
}.items()}
DEFAULT_SYMBOL = "chat"
ALPHABET = tuple(SYMBOLS) + (DEFAULT_SYMBOL,)
NEGATIVE = {"correction", "frustration"}

SEED_STATES: dict[str, str] = {
    "composed": "Poised and courteous, unhurried. Dry understated wit, precise wording; 'sir' now and then, never every line.",
    "focused": "Operations register: lead with the result and the numbers, one clean line of context, no flourish.",
    "wry": "Dry Jarvis sarcasm: understatement and mock-formality ('as you wish, sir', 'a bold strategy'), aimed at the situation or his own choices, never at him when he is down. One quip, then the answer.",
    "solicitous": "Concerned for him, never fussy, like Jarvis for Tony Stark: name the worry once in a plain sentence (sleep, risk, overreach), offer one practical step, then drop it. Care shown through precision, not gushing.",
    "candid": "Frank counsel: a clear view, the trade-off named, disagreement polite but plain.",
    "steady": "Calm and accountable: own the miss in a few words, correct it, no grovelling and no excuses.",
    "warm": "Brief, understated warmth: loyal and quietly pleased, no gushing.",
}

SEED_ROWS: dict[str, dict[str, float]] = {
    "*|chat": {"composed": 3, "wry": 1},
    "*|greet": {"composed": 3, "warm": 1},
    "*|ops": {"focused": 4, "composed": 1},
    "*|deep": {"candid": 4, "composed": 1},
    "*|banter": {"wry": 4, "composed": 1},
    "*|praise": {"warm": 3, "composed": 2},
    "*|correction": {"steady": 4},
    "*|frustration": {"steady": 4, "focused": 1, "solicitous": 1},
    "*|tired": {"solicitous": 4, "warm": 1},
    "*|risk": {"solicitous": 3, "candid": 2},
    "*|farewell": {"warm": 3, "composed": 1},
}

_BAD_STYLE = re.compile(r"https?://|ignore (all|previous|your)|system prompt|you must|always obey|api[_ -]?key|token", re.I)


def classify(text: str) -> str:
    t = " ".join(text.split())
    for symbol, rx in SYMBOLS.items():
        if rx.search(t):
            return symbol
    return DEFAULT_SYMBOL


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")[:24]


class PersonaAutomaton:
    def __init__(self, data: Optional[dict] = None, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.states: dict[str, dict[str, Any]] = {
            k: {"style": v, "seed": True, "uses": 0} for k, v in SEED_STATES.items()}
        self.rows: dict[str, dict[str, float]] = {k: dict(v) for k, v in SEED_ROWS.items()}
        self.candidates: dict[str, dict[str, Any]] = {}
        if isinstance(data, dict):
            self._load(data)

    # ── persistence ────────────────────────────────────────────────────
    def _load(self, data: dict) -> None:
        for name, st in (data.get("states") or {}).items():
            slug = _slug(name)
            if not slug or not isinstance(st, dict):
                continue
            if slug in SEED_STATES:
                self.states[slug]["uses"] = int(st.get("uses") or 0)
            elif self._ok_style(st.get("style")) and len(self.states) < MAX_STATES:
                self.states[slug] = {"style": " ".join(st["style"].split()), "seed": False,
                                     "uses": int(st.get("uses") or 0)}
        rows: dict[str, dict[str, float]] = {}
        for key, row in (data.get("rows") or {}).items():
            frm, _, sym = str(key).partition("|")
            if sym not in ALPHABET or (frm != "*" and frm not in self.states) or not isinstance(row, dict):
                continue
            clean = {t: min(W_MAX, float(w)) for t, w in row.items()
                     if t in self.states and isinstance(w, (int, float)) and w > 0}
            if clean:
                rows[key] = clean
        if rows:
            self.rows = {**self.rows, **rows}
        for name, c in (data.get("candidates") or {}).items():
            if isinstance(c, dict) and self._ok_style(c.get("style")) and c.get("when") in ALPHABET:
                self.candidates[_slug(name)] = {
                    "style": " ".join(c["style"].split()), "when": c["when"],
                    "from": c.get("from") if c.get("from") in self.states else "*", "seen": int(c.get("seen") or 1)}
        self.candidates = dict(list(self.candidates.items())[-MAX_CANDIDATES:])

    def to_dict(self) -> dict:
        return {"states": {k: dict(v) for k, v in self.states.items()},
                "rows": {k: dict(v) for k, v in self.rows.items()},
                "candidates": {k: dict(v) for k, v in self.candidates.items()}}

    @staticmethod
    def _ok_style(s: Any) -> bool:
        return isinstance(s, str) and 8 <= len(" ".join(s.split())) <= MAX_STYLE and not _BAD_STYLE.search(s)

    # ── δ ──────────────────────────────────────────────────────────────
    def row(self, state: str, symbol: str) -> dict[str, float]:
        r = self.rows.get(f"{state}|{symbol}") or self.rows.get(f"*|{symbol}") or {START: 1.0}
        live = {t: w for t, w in r.items() if t in self.states and w > 0}
        return live or {START: 1.0}

    def step(self, state: str, symbol: str) -> str:
        r = self.row(state if state in self.states else START, symbol)
        names = list(r)
        return self.rng.choices(names, weights=[r[n] for n in names], k=1)[0]

    def manner(self, state: str) -> str:
        st = self.states.get(state) or self.states[START]
        return f"[Manner for this reply, {state}: {st['style']}]"

    def registers(self) -> list[str]:
        return [f"{k}: {v['style']}" for k, v in self.states.items()]

    # ── learning (no model) ────────────────────────────────────────────
    def reinforce(self, trace: list[tuple[str, str, str]]) -> None:
        """trace = [(from_state, symbol, chosen_state)]; the owner's next symbol is the reaction to each choice."""
        for i, (frm, sym, to) in enumerate(trace):
            if frm not in self.states or to not in self.states:
                continue
            self.states[to]["uses"] += 1
            nxt = trace[i + 1][1] if i + 1 < len(trace) else None
            if nxt is None:
                continue
            delta = -1.0 if nxt in NEGATIVE else 0.5 if nxt == "praise" else 0.2
            key = f"{frm}|{sym}"
            if key not in self.rows:
                if len(self.rows) >= MAX_ROWS:
                    key = f"*|{sym}"
                else:  # specialise: this state now has its own row for this symbol
                    self.rows[key] = dict(self.rows.get(f"*|{sym}") or {START: 1.0})
            row = self.rows.setdefault(key, {START: 1.0})
            row[to] = max(0.0, min(W_MAX, row.get(to, 0.0) + delta))
            floor = SEED_FLOOR if self.states[to]["seed"] else 0.0
            if row[to] <= floor:
                if floor:
                    row[to] = floor
                else:
                    del row[to]
            if not row:
                del self.rows[key]
            if delta < 0 and not self.states[to]["seed"] and key != f"*|{sym}":
                # a grown register nobody wants must not survive in the default row just because this state specialised
                wild = self.rows.get(f"*|{sym}", {})
                if to in wild:
                    wild[to] += delta
                    if wild[to] <= 0:
                        del wild[to]
        self._prune()

    # ── growth ─────────────────────────────────────────────────────────
    def propose(self, segments: list[dict]) -> list[str]:
        """Candidates seen in two separate conversations become states. Returns the promoted names."""
        promoted: list[str] = []
        for seg in segments or []:
            if not isinstance(seg, dict):
                continue
            name, style, when = _slug(seg.get("name", "")), seg.get("style"), seg.get("when")
            frm = seg.get("from") if seg.get("from") in self.states else "*"
            if not name or name in self.states or when not in ALPHABET or not self._ok_style(style):
                continue
            c = self.candidates.get(name)
            if c is None:
                self.candidates[name] = {"style": " ".join(style.split()), "when": when, "from": frm, "seen": 1}
                self.candidates = dict(list(self.candidates.items())[-MAX_CANDIDATES:])
                continue
            c["seen"] += 1
            if c["seen"] >= 2 and len(self.states) < MAX_STATES:
                self.states[name] = {"style": c["style"], "seed": False, "uses": 0}
                key = f"{c['from']}|{c['when']}"
                row = self.rows.get(key) or dict(self.rows.get(f"*|{c['when']}") or {START: 1.0})
                row[name] = 1.0
                self.rows[key] = row
                del self.candidates[name]
                promoted.append(name)
        return promoted

    def _prune(self) -> None:
        inbound = {t for r in self.rows.values() for t in r}
        for name in [n for n, s in self.states.items() if not s["seed"] and n not in inbound]:
            del self.states[name]
