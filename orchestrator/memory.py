"""Persistent conversational memory — facts + rolling summary + notes search.

Not RAG. Facts are extracted by the same local LLM, retrieved with lexical
overlap (fast, no extra model). Session history is compacted into a summary
so the live context stays short.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

_STOP = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for", "at",
    "is", "are", "was", "were", "be", "been", "i", "you", "we", "they", "he",
    "she", "it", "my", "your", "our", "me", "us", "this", "that", "with",
    "from", "as", "if", "so", "do", "did", "does", "have", "has", "had",
    "just", "about", "what", "when", "who", "how", "why", "can", "will",
    "would", "could", "should", "please", "okay", "ok", "yeah", "yes", "no",
}

RECALL_RE = re.compile(
    r"\b(remember|last time|last week|you said|we (decided|talked|discussed)|"
    r"what did we|remind me|from (the )?notes|previous session|my name|"
    r"do you (know|recall))\b",
    re.I,
)

FORGET_ALL = {"forget everything", "clear memory", "wipe memory", "reset memory"}
FORGET_LAST = {"forget that", "forget the last thing", "scratch that memory"}

FACT_EXTRACT_PROMPT = """Extract durable facts worth remembering next week from this conversation turn.
Return a JSON array only. Each item: {"text": "one sentence fact", "kind": "preference|decision|person|project|constraint|other"}
Rules:
- Names, preferences, decisions, constraints, projects, relationships.
- Skip greetings, jokes, one-off chit-chat, and the assistant's filler.
- One fact per item. Short. Third person ("The user prefers…").
- Empty array [] if nothing durable."""

SUMMARY_PROMPT = """Rewrite the running session summary after new turns.
Keep 6–12 sentences. Third person. No markdown. Facts and decisions only.

Current summary:
{summary}

New turns:
{turns}"""


def tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9']{2,}", text.lower()) if w not in _STOP]


def lexical_score(query_toks: List[str], text: str) -> float:
    if not query_toks:
        return 0.0
    doc = tokens(text)
    if not doc:
        return 0.0
    qset, dset = set(query_toks), set(doc)
    overlap = len(qset & dset)
    if overlap == 0:
        return 0.0
    return overlap * 2.0 + sum(1.0 for w in doc if w in qset) / max(len(doc), 1)


def parse_fact_payload(raw: str) -> List[dict]:
    if not raw or not raw.strip():
        return []
    blob = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", blob, re.S)
    if fence:
        blob = fence.group(1).strip()
    match = re.search(r"\[.*\]", blob, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, str) and item.strip():
            out.append({"text": item.strip(), "kind": "other"})
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("fact") or "").strip()
        if not text:
            continue
        kind = str(item.get("kind") or "other").strip().lower()
        if kind not in {"preference", "decision", "person", "project", "constraint", "other"}:
            kind = "other"
        out.append({"text": text, "kind": kind})
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class MemoryStore:
    def __init__(self, root: str, notes_dir: str, keep_turns: int = 8, fact_limit: int = 8):
        self.root = Path(root)
        self.notes_dir = Path(notes_dir)
        self.keep_turns = keep_turns
        self.fact_limit = fact_limit
        self.root.mkdir(parents=True, exist_ok=True)
        self.facts_path = self.root / "facts.jsonl"
        self.state_path = self.root / "state.json"
        self.session_summary = ""
        self._facts: List[dict] = []
        self._load()

    def _load(self):
        if self.facts_path.exists():
            for line in self.facts_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._facts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text())
                self.session_summary = str(state.get("session_summary") or "")
            except json.JSONDecodeError:
                pass

    def _save_state(self):
        self.state_path.write_text(json.dumps({
            "session_summary": self.session_summary,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fact_count": len(self._facts),
        }, indent=2))

    def add_facts(self, facts: Iterable[dict]) -> int:
        added = 0
        existing = {_norm(f.get("text", "")) for f in self._facts}
        with open(self.facts_path, "a") as fh:
            for fact in facts:
                text = str(fact.get("text", "")).strip()
                if not text or _norm(text) in existing:
                    continue
                row = {
                    "id": uuid.uuid4().hex[:12],
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                    "kind": fact.get("kind") or "other",
                }
                self._facts.append(row)
                existing.add(_norm(text))
                fh.write(json.dumps(row) + "\n")
                added += 1
        if added:
            self._save_state()
            logger.info("Memory: stored %d new fact(s) (total %d)", added, len(self._facts))
        return added

    def forget_last(self) -> Optional[str]:
        if not self._facts:
            return None
        dropped = self._facts.pop()
        self._rewrite_facts()
        return dropped.get("text")

    def forget_all(self):
        self._facts.clear()
        self.session_summary = ""
        self._rewrite_facts()
        self._save_state()

    def _rewrite_facts(self):
        tmp = self.facts_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as fh:
            for row in self._facts:
                fh.write(json.dumps(row) + "\n")
        tmp.replace(self.facts_path)

    def retrieve_facts(self, query: str, k: Optional[int] = None) -> List[dict]:
        k = k or self.fact_limit
        if not self._facts:
            return []
        q = tokens(query)
        scored = [(lexical_score(q, f.get("text", "")), i, f) for i, f in enumerate(self._facts)]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if scored[0][0] <= 0:
            return list(reversed(self._facts[-k:]))
        return [f for s, _, f in scored if s > 0][:k]

    def search_notes(self, query: str, k: int = 3) -> List[str]:
        if not self.notes_dir.exists():
            return []
        q = tokens(query)
        if not q:
            return []
        hits: List[tuple[float, str]] = []
        for path in self.notes_dir.rglob("*"):
            if path.suffix.lower() not in {".md", ".jsonl", ".txt"}:
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            score = lexical_score(q, text)
            if score <= 0:
                continue
            snippet = _best_snippet(text, q)
            hits.append((score, f"{path.name}: {snippet}"))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits[:k]]

    def context_for(self, user_text: str) -> str:
        parts: List[str] = ["## Memory"]
        if self.session_summary:
            parts.append("Session so far:\n" + self.session_summary.strip())
        facts = self.retrieve_facts(user_text)
        if facts:
            parts.append("Known facts:\n" + "\n".join(f"- {f['text']}" for f in facts))
        if RECALL_RE.search(user_text):
            notes = self.search_notes(user_text)
            if notes:
                parts.append("Past notes:\n" + "\n".join(f"- {n}" for n in notes))
        if len(parts) == 1:
            return ""
        parts.append(
            "Use these only when relevant. Do not mention this memory list. "
            "Do not recite it. Speak naturally."
        )
        return "\n\n".join(parts)

    def observe(self, llm, user_text: str, assistant_text: str):
        """Extract facts from a finished turn. Call after speech, not during."""
        turn = f"USER: {user_text}\nASSISTANT: {assistant_text}"
        try:
            raw = llm.complete(FACT_EXTRACT_PROMPT, turn, max_tokens=400)
        except Exception as e:
            logger.warning("Memory extract failed: %s", e)
            return
        facts = parse_fact_payload(raw)
        if facts:
            self.add_facts(facts)

    def compact(self, llm, history: List[dict]) -> List[dict]:
        """Fold old turns into session_summary; return trimmed history."""
        keep = self.keep_turns * 2
        if len(history) <= keep:
            return history
        old, recent = history[:-keep], history[-keep:]
        turns = "\n".join(f"{m.get('role','?').upper()}: {m.get('content','')}" for m in old)
        prompt = SUMMARY_PROMPT.format(
            summary=self.session_summary or "(empty)",
            turns=turns,
        )
        try:
            summary = llm.complete(
                "You compress conversation history into a durable summary.",
                prompt,
                max_tokens=600,
            ).strip()
        except Exception as e:
            logger.warning("Memory compact failed: %s", e)
            return recent
        if summary:
            self.session_summary = summary
            self._save_state()
            logger.info("Memory: compacted %d turns into summary", len(old) // 2)
        return recent

    def is_forget_all(self, text: str) -> bool:
        return text.lower().strip().rstrip(".!") in FORGET_ALL

    def is_forget_last(self, text: str) -> bool:
        return text.lower().strip().rstrip(".!") in FORGET_LAST


def _best_snippet(text: str, query_toks: List[str], width: int = 280) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    best, best_s = "", -1.0
    for ln in lines:
        s = lexical_score(query_toks, ln)
        if s > best_s:
            best, best_s = ln, s
    snippet = best or (lines[0] if lines else text[:width])
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if len(snippet) > width:
        snippet = snippet[: width - 1] + "…"
    return snippet
