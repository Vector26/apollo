"""Live voice as a front end of Virtual Self (vself).

vself owns who is speaking: persona, memories, time context, style exemplars,
sessions and memory extraction. This client only streams the reply:

  1. POST /v1/voice:prepare  → system + messages (no writes; safe to call speculatively)
  2. local brain first (via modelq), cloud only if local is busy, too slow to its
     first words, errors, or opens with a defect
  3. POST /v1/voice:commit   → once the turn is confirmed and spoken
  4. POST /v1/voice:end      → when the conversation is over: vself writes the summary,
     facts, tasks and the session file

A reply vself already decided on (note saved, task added, "what's open?") arrives as
``direct_reply``: spoken as is, no model involved.
"""

import collections
from pathlib import Path
import logging
import queue
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable, Generator, List, Optional

import httpx
from openai import OpenAI

from agent_tools import (
    HEDGE_START,
    AgentTools,
    Plan,
    find_call,
    find_plane_call,
    find_remind_call,
    parse_call,
    parse_plane_call,
    parse_remind_call,
    promised_agent,
)
from apollo_self import OWNER, ApolloSelf
from initiative import aside_prompts
from persona_automaton import START, classify
from cursor_brain import CursorRoute
from llm_client import LOCAL_EXTRA_BODY, LLMClient, read_env_var

logger = logging.getLogger(__name__)

GUARD_CHARS = 16
HISTORY_MESSAGES = 12
TOOL_REPLY_KEEP = 160
CONVO_MESSAGES = 80  # whole-conversation buffer for Apollo's self-reflection (history is capped for latency)
ANSWER_ALL = ("\n\nIf the owner's message holds several separate statements or questions (they often speak in bursts), "
              "respond to every one of them, briefly and in order, before moving on. Never answer only the last part.")
# Openings that mean the local model went off the rails; judged on the first GUARD_CHARS.
_DEFECT = re.compile(r"^\s*(<think|<\|im_|\*\*|#|```|[-*] |the user\b|as an ai\b)", re.I)


RETRY_PAUSE_S = 0.8


class BrainUnavailable(RuntimeError):
    """No route produced a reply (twice): say so out loud instead of going quiet."""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clauses(text: str) -> List[str]:
    """Split spoken text at commas and sentence ends, keeping the spaces so it joins back unchanged."""
    return re.findall(r"[^,.;:!?]+[,.;:!?]*\s*|[,.;:!?]+\s*", text) or [text]


def local_opening_ok(text: str) -> bool:
    return bool(text.strip()) and not _DEFECT.search(text)


class VselfBrain:
    """Drop-in for LLMClient on the talk path (chat_stream / commit / complete)."""

    cursor: Optional[CursorRoute] = None  # Composer route; set by __init__ when brains.vself.cursor is enabled
    hedge_after_s: float = 2.2
    tools: Optional[AgentTools] = None  # set by from_config when brains.vself.tools is configured
    _tool_plan: Optional[tuple[str, Plan]] = None  # plan of the turn being generated; applied at commit
    apollo_self: Optional[ApolloSelf] = None  # companion personality growth (not vself owner memory)
    _session_t0: Optional[float] = None       # first committed turn of this conversation (monotonic)

    def __init__(self, api_url: str, token: str, local: LLMClient, cloud: Optional[LLMClient],
                 local_first_token_ms: int = 1200, queue_url: Optional[str] = None,
                 prepare_timeout_s: float = 4.0, prefer: str = "local",
                 cursor: Optional[CursorRoute] = None, hedge_after_s: float = 2.2):
        self.api = httpx.Client(base_url=api_url.rstrip("/"), timeout=prepare_timeout_s,
                                headers={"Authorization": f"Bearer {token}"})
        self.local = local
        self.cloud = cloud
        self.threshold_s = local_first_token_ms / 1000.0
        self.queue_url = queue_url
        self.cursor = cursor
        self.hedge_after_s = hedge_after_s
        if prefer == "cursor" and cursor is None:
            prefer = "cloud"  # Composer is switched off: the next in line
        self.prefer = prefer if (prefer == "cursor" or cloud is not None) else "local"
        routes = [local.model] + ([cloud.model] if cloud else [])
        if self.prefer in ("cloud", "cursor"):
            routes.reverse()
        if self.prefer == "cursor":
            routes.insert(0, cursor.model)
        self.model = "vself → " + " | ".join(routes)
        self.history: List[dict] = []
        self._convo: List[dict] = []
        self._pstate = START                      # Prometheus' current register (persona_automaton)
        self._trace: List[tuple] = []             # (from, symbol, chosen) per committed turn
        self._step: Optional[tuple] = None        # (message, from, symbol, chosen): pure until commit
        self.last_route: Optional[str] = None
        self.session_id: Optional[str] = None
        self._last_system: Optional[str] = None
        self._local_misses: collections.deque = collections.deque(maxlen=3)
        self.owns_memory = True
        self.last_activity = time.monotonic()
        self._pending: List[threading.Thread] = []
        self._lock = threading.Lock()
        self.tools: Optional[AgentTools] = None
        self._tool_plan: Optional[tuple[str, Plan]] = None  # the plan of the turn being generated, applied at commit

    @classmethod
    def from_config(cls, cfg: dict, brains: dict) -> "VselfBrain":
        token = read_env_var(cfg["token_env"], cfg.get("token_file"))
        if not token:
            raise RuntimeError(f"{cfg['token_env']} not set and not found in {cfg.get('token_file')}")
        lc = cfg["local"]
        local = LLMClient(
            base_url=lc["base_url"], model=lc["model"], system_prompt="",
            temperature=lc.get("temperature", 0.4), max_tokens=lc.get("max_tokens", 200),
            timeout_s=lc.get("timeout_s", 20), max_retries=0,
        )
        local.client = local.client.with_options(default_headers={
            "X-Fortress-Agent": "vox-voice", "X-Fortress-Priority": "interactive"})
        cloud = None
        if cfg.get("cloud"):
            cc = dict(brains[cfg["cloud"]])
            cc.pop("fallback_to_local", None)
            cc.setdefault("system_prompt", "")  # vself supplies the system prompt per turn
            cloud = LLMClient.from_config(cc)
        cursor = None
        cur = cfg.get("cursor") or {}
        if cur.get("enabled"):
            cursor = CursorRoute(
                sandbox_dir=Path(cur.get("sandbox_dir", "~/.local/state/apollo/composer-sandbox")).expanduser(),
                state_dir=Path(cur.get("state_dir", "~/.local/state/apollo")).expanduser(),
                agent_bin=cur.get("agent_bin", "cursor-agent"), model_id=cur.get("model_id", "composer-2.5[fast=true]"),
                give_up_s=float(cur.get("give_up_s", 6.0)), max_words=int(cur.get("max_words", 140)),
                rotate_turns=int(cur.get("rotate_turns", 24)))
        brain = cls(cfg["api_url"], token, local, cloud,
                    local_first_token_ms=int(cfg.get("local_first_token_ms", 1200)),
                    queue_url=cfg.get("queue_url"),
                    prepare_timeout_s=float(cfg.get("prepare_timeout_s", 4.0)),
                    prefer=str(cfg.get("prefer", "local")), cursor=cursor,
                    hedge_after_s=float(cur.get("hedge_after_s", 2.2)))
        if cfg.get("tools"):
            try:
                brain.tools = AgentTools.from_config(cfg["tools"])
            except Exception as e:  # talking still works without the tools
                logger.warning("agent tools unavailable: %s", e)
        brain.apollo_self = ApolloSelf.from_config(cfg.get("personality"))
        return brain

    def _personality_block(self) -> str:
        if self.apollo_self is None:
            return ""
        try:
            return "\n\n" + self.apollo_self.prompt_block()
        except Exception as e:
            logger.warning("apollo self prompt failed: %s", e)
            return ""

    def _manner(self, user_message: str) -> str:
        """The register for this reply, from the automaton. Computed once per message (retries reuse it) and
        only committed with the turn, so a speculative or repeated prepare never moves the state."""
        if self.apollo_self is None:
            return ""
        try:
            if self._step is None or self._step[0] != user_message:
                auto = self.apollo_self.automaton()
                symbol = classify(user_message)
                chosen = auto.step(self._pstate, symbol)
                self._step = (user_message, self._pstate, symbol, chosen)
            return self.apollo_self.automaton().manner(self._step[3])
        except Exception as e:
            logger.warning("persona automaton failed: %s", e)
            return ""

    def _reflect_self(self, history: List[dict], trace: Optional[List[tuple]] = None) -> None:
        if self.apollo_self is None:
            return
        if trace:
            try:
                self.apollo_self.learn(trace)
            except Exception as e:
                logger.warning("persona learning failed: %s", e)
        ref = self.cloud  # never the local brain: it is shared with DUM-E
        if ref is None or not history:
            return

        def complete(system: str, user: str) -> str:
            return ref.complete(system, user, max_tokens=900)

        self.apollo_self.consolidate(history, complete=complete)

    # ── vself ──────────────────────────────────────────────────────────
    def prepare_full(self, user_message: str) -> tuple[List[dict], Optional[str]]:
        """(messages, direct_reply). ``direct_reply`` is a finished answer: speak it, skip the model."""
        try:
            r = self.api.post("/v1/voice:prepare", json={
                "query": user_message, "history": self.history[-HISTORY_MESSAGES:], "now": _now_iso()})
            r.raise_for_status()
            body = r.json()
            self._last_system = body["system"] + self._personality_block() + self._tools_block() + ANSWER_ALL
            logger.info("vself prepare %s web=%s action=%s", body.get("timings_ms"), body.get("web"),
                        body.get("action"))
            manner = self._manner(user_message)
            tail = [{"role": "user", "content": manner}] if manner else []
            return ([{"role": "system", "content": self._last_system}] + body["messages"] + tail,
                    body.get("direct_reply") or None)
        except Exception as e:
            if self._last_system is None:
                raise RuntimeError(f"vself unreachable: {e}") from e
            # persona is stable across turns; without fresh memories is better than silence
            logger.warning("vself prepare failed (%s); using last persona without memories", e)
            manner = self._manner(user_message)
            tail = [{"role": "user", "content": manner}] if manner else []
            return ([{"role": "system", "content": self._last_system}] + self.history[-HISTORY_MESSAGES:]
                    + [{"role": "user", "content": user_message}] + tail, None)

    def _tools_block(self) -> str:
        try:
            block = self.tools.capabilities_block() if self.tools else ""
        except Exception:
            logger.exception("capabilities block failed")
            return ""
        return "\n\n" + block if block else ""

    def prepare(self, user_message: str) -> List[dict]:
        return self.prepare_full(user_message)[0]

    def commit(self, user_message: str, reply_text: str) -> None:
        planned, self._tool_plan = self._tool_plan, None
        if planned and planned[0] == user_message and planned[1].apply:
            # a confirmed turn: now the side effect may happen (off the voice thread)
            threading.Thread(target=planned[1].apply, daemon=True, name="tool-apply").start()
        before = self.history[-HISTORY_MESSAGES:]
        kept = reply_text
        if planned and planned[0] == user_message and len(kept) > TOOL_REPLY_KEEP:
            # a long tool answer left whole in history gets echoed into unrelated chat; keep the gist
            kept = kept[:TOOL_REPLY_KEEP].rsplit(" ", 1)[0] + "…"
        self.history += [{"role": "user", "content": user_message},
                         {"role": "assistant", "content": kept}]
        del self.history[:-HISTORY_MESSAGES]
        if self._step and self._step[0] == user_message:
            _, frm, symbol, chosen = self._step
            self._trace.append((frm, symbol, chosen))
            self._pstate, self._step = chosen, None
        self._convo += [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply_text}]
        del self._convo[:-CONVO_MESSAGES]
        self.last_activity = time.monotonic()
        if self._session_t0 is None:
            self._session_t0 = self.last_activity
        t = threading.Thread(target=self._commit_remote, args=(user_message, reply_text, before, _now_iso()),
                             daemon=True, name="vself-commit")
        with self._lock:
            self._pending = [p for p in self._pending if p.is_alive()] + [t]
        t.start()

    def note_background(self, text: str) -> None:
        """Something was said unprompted (a late agent answer): put it in the thread so follow-ups can refer to it."""
        kept = text if len(text) <= TOOL_REPLY_KEEP * 2 else text[:TOOL_REPLY_KEEP * 2].rsplit(" ", 1)[0] + "…"
        self.history += [{"role": "user", "content": "(an agent you asked has just answered)"},
                         {"role": "assistant", "content": kept}]
        del self.history[:-HISTORY_MESSAGES]
        self.last_activity = time.monotonic()

    # ── speaking up unprompted (initiative.py decides when; this only supplies the signals and the line) ──
    def cloud_in_use(self) -> bool:
        """The last reply came from Composer or OpenRouter (not the local brain, not a tool or a direct reply)."""
        return bool(self.last_route) and self.last_route.startswith(("cursor:", "cloud:"))

    def owner_symbols(self) -> List[str]:
        return [t[1] for t in self._trace[-4:]]

    def session_age_s(self) -> float:
        return 0.0 if self._session_t0 is None else time.monotonic() - self._session_t0

    def aside(self, tone: str, facts: str) -> Optional[str]:
        """One short line Apollo might say unprompted, or NONE, from the cloud model. Never the local brain
        (shared with DUM-E). Sees the persona and the last few turns only: no vself memories leave the host."""
        if self.cloud is None or self.apollo_self is None or not self._convo:
            return None
        exchange = "\n".join(f"{OWNER if m['role'] == 'user' else 'You'}: {m['content'][:200]}"
                             for m in self._convo[-6:])
        system, user = aside_prompts(tone, self.apollo_self.prompt_block(), OWNER, facts, exchange)
        return self.cloud.complete(system, user, max_tokens=90)

    def note_aside(self, text: str) -> None:
        """Said unprompted: keep it in the thread so "what do you mean?" has something to refer to."""
        self.history += [{"role": "user", "content": "(you spoke up unprompted)"},
                         {"role": "assistant", "content": text}]
        del self.history[:-HISTORY_MESSAGES]
        self.last_activity = time.monotonic()

    def _commit_remote(self, user_message: str, reply_text: str, before: List[dict], now: str) -> None:
        try:
            r = self.api.post("/v1/voice:commit", json={
                "user": user_message, "assistant": reply_text, "session_id": self.session_id,
                "history": before, "now": now})
            r.raise_for_status()
            self.session_id = r.json()["session_id"]
        except Exception as e:
            logger.warning("vself commit failed: %s", e)

    def idle_s(self) -> float:
        return time.monotonic() - self.last_activity

    def has_session(self) -> bool:
        return bool(self.session_id) or any(p.is_alive() for p in self._pending)

    def end(self, keep_history: bool = False) -> bool:
        """Close the conversation: vself consolidates it (summary, facts, tasks, files).

        The next turn opens a fresh session. Returns True if vself accepted it.
        """
        with self._lock:
            pending = list(self._pending)
        for t in pending:  # the first commit is what creates the session id
            t.join(timeout=5.0)
        sid = self.session_id
        snapshot, self._convo = list(self._convo), []
        trace, self._trace, self._pstate, self._step = list(self._trace), [], START, None
        self._session_t0 = None
        if snapshot:
            # non-daemon: a quick exit waits for the update instead of losing it (the cloud call has a timeout)
            threading.Thread(target=self._reflect_self, args=(snapshot, trace), name="apollo-self").start()
        if not sid:
            return False
        self.session_id = None
        if not keep_history:
            self.history.clear()
            if self.cursor is not None:
                self.cursor.new_conversation()
        try:
            r = self.api.post("/v1/voice:end", json={"session_id": sid})
            r.raise_for_status()
            logger.info("vself session %s ended", sid)
            return True
        except Exception as e:
            logger.warning("vself end failed (the idle sweep will pick it up): %s", e)
            return False

    def reset(self):
        self.history.clear()
        if self.cursor is not None:
            self.cursor.new_conversation()

    # ── routing ────────────────────────────────────────────────────────
    def local_available(self) -> bool:
        if not self.queue_url:
            return True
        try:
            q = httpx.get(self.queue_url, timeout=0.3).json()
        except Exception:
            return True  # modelq unreachable: let the request itself decide
        busy = q.get("brain_paused") or q.get("active_count", 0) >= q.get("max_in_flight", 2)
        if busy:
            logger.info("local brain busy/paused (%s) → cloud", q)
        return not busy

    def chat_stream(self, user_message: str, memory_block: str = "",
                    commit: bool = True) -> Generator[str, None, None]:
        self._tool_plan = None
        plan = self.tools.plan(user_message, resolve=False) if self.tools else None
        notice = self.tools.take_notice() if self.tools else None
        reply = []
        if notice:
            reply.append(notice + " ")
            yield notice + " "
        if plan:
            yield from self._speak_plan(user_message, plan, reply, commit)
            return
        messages, direct = self.prepare_full(user_message)
        if direct:
            self.last_route = "vself:direct"
            reply.append(direct)
            yield direct
        else:
            call = None
            stream = self._route_with_retry(messages)
            for kind, piece in (self._peel_call(stream, user_message) if self.tools else (("text", d) for d in stream)):
                if kind == "call":
                    call = piece
                    break
                reply.append(piece)
                yield piece
            if call:
                if call[0] == "plane":
                    plan = self.tools.plane_command(call[1], call[2])
                elif call[0] == "remind":
                    # the model may only touch reminders when the owner actually spoke of them
                    plan = (self.tools.reminder_command(call[1], call[2]) if self.tools.mentions_reminders(user_message)
                            else self.tools.plan(user_message))
                else:
                    plan = self.tools.call_agent(*call, by=self.last_route or "model")
                if not plan:
                    msg = "I can't reach that right now."
                    reply.append(msg)
                    yield msg
                else:
                    yield from self._speak_plan(user_message, plan, reply, commit)
                    return
        if commit:
            self.commit(user_message, "".join(reply))

    def _speak_plan(self, user_message: str, plan, reply: list, commit: bool) -> Generator[str, None, None]:
        self.last_route = f"tool:{plan.tool}"
        if plan.ack:  # a slow agent is working: say so now instead of going quiet
            reply.append(plan.ack + " ")
            yield plan.ack + " "
        plan = self.tools.finish(plan)
        self._tool_plan = (user_message, plan)
        for piece in _clauses(plan.speak):  # short pieces, like a streamed reply: the voice starts sooner
            reply.append(piece)
            yield piece
        if commit:
            self.commit(user_message, "".join(reply))

    @staticmethod
    def _peel_call(deltas, user_message: str = "") -> Generator[tuple[str, Any], None, None]:
        """Pass a reply through, except for what is really a request for an agent. Never delays a normal reply.
        - a call line ([[ask:dume|...]]) is a request, not speech: it is held back wherever it appears, and run;
        - a reply that opens like a promise or an offer ("I'll ask DUM-E", "I can't check, ask DUM-E") is held for
          its first sentence or two; if it does say it will ask an agent, the call is made with the owner's own words,
          and a call line in that held text replaces the hedging sentence entirely."""
        def hold_hedges():
            buf, mode = "", "start"
            for delta in deltas:
                if mode == "text":
                    yield "text", delta
                    continue
                buf += delta
                head = buf.lstrip()
                if not head:
                    continue
                if mode == "start":
                    if len(head) < 24 and not re.search(r"[.!?]\s", head):
                        continue
                    mode = "hedge" if HEDGE_START.match(head) else "text"
                    if mode == "text":
                        yield "text", buf
                        continue
                rc = find_remind_call(head)
                if rc:
                    yield "call", ("remind", rc[0], rc[1])
                    return
                pc = find_plane_call(head)
                if pc:
                    yield "call", ("plane", pc[0], pc[1])
                    return
                call = find_call(head)
                if call:
                    yield "call", call
                    return
                if len(re.findall(r"[.!?](?:\s|$)", head)) >= 2 or len(head) > 220:
                    agent = promised_agent(head)
                    if agent and user_message:
                        yield "call", (agent, user_message)
                        return
                    mode = "text"
                    yield "text", buf
            if mode != "text" and buf:
                rc = find_remind_call(buf)
                pc = find_plane_call(buf)
                call = find_call(buf)
                agent = None if call or pc or rc else promised_agent(buf)
                if rc:
                    yield "call", ("remind", rc[0], rc[1])
                elif pc:
                    yield "call", ("plane", pc[0], pc[1])
                elif call:
                    yield "call", call
                elif agent and user_message:
                    yield "call", (agent, user_message)
                else:
                    yield "text", buf

        held = ""  # a possible call line being held back (it starts with "[[")
        for kind, piece in hold_hedges():
            if kind == "call":
                yield kind, piece
                return
            s = held + piece
            held = ""
            while s:
                i = s.find("[[")
                if i < 0:
                    if s.endswith("["):  # might be the first half of "[["
                        held, s = "[", s[:-1]
                    if s:
                        yield "text", s
                    break
                if i:
                    yield "text", s[:i]
                s = s[i:]
                end = s.find("]]")
                if end < 0 and len(s) <= 300:
                    held = s
                    break
                chunk, s = (s[:end + 2], s[end + 2:]) if end >= 0 else (s, "")
                rc = parse_remind_call(chunk)
                if rc:
                    yield "call", ("remind", rc[0], rc[1])
                    return
                pc = parse_plane_call(chunk)
                if pc:
                    yield "call", ("plane", pc[0], pc[1])
                    return
                call = parse_call(chunk)
                if call:
                    yield "call", call
                    return
                yield "text", chunk
        if held:
            yield "text", held

    def _route_with_retry(self, messages: List[dict]) -> Generator[str, None, None]:
        """One more try when nothing came back (a network blip); then BrainUnavailable."""
        got = False
        for attempt in (0, 1):
            try:
                for delta in self._route(messages):
                    got = True
                    yield delta
                return
            except Exception as e:
                if got:
                    raise  # mid-sentence: the caller keeps what was already said
                if attempt == 1:
                    raise BrainUnavailable(str(e)) from e
                logger.warning("no route answered (%s); retrying once", e)
                time.sleep(RETRY_PAUSE_S)

    def _set_route(self, label: str) -> None:
        self.last_route = label

    def _cloud_first(self, messages: List[dict], on_open: Optional[Callable] = None,
                     on_route: Optional[Callable[[str], None]] = None) -> Generator[str, None, None]:
        """Cloud answers; the local brain only if the cloud fails before its first word."""
        on_route = on_route or self._set_route
        got = False
        try:
            for delta in self.cloud._race(messages, on_open=on_open):
                if not got:
                    got = True
                    on_route(f"cloud:{self.cloud.model}")
                yield delta
            if got:
                return
            logger.warning("cloud returned nothing → local")
        except Exception as e:
            if got:
                raise
            logger.warning("cloud failed (%s) → local", e)
            if not self.local_available():
                raise  # local is paused or full: its half-open stream is worse than a clear failure
        on_route(f"local:{self.local.model}")
        yield from self.local._race(messages, on_open=on_open)

    def _fallback_chain(self, messages: List[dict], on_open: Optional[Callable] = None,
                        on_route: Optional[Callable[[str], None]] = None) -> Generator[str, None, None]:
        """What answers when Composer does not: OpenRouter, then the local brain."""
        if self.cloud is not None:
            yield from self._cloud_first(messages, on_open, on_route)
            return
        (on_route or self._set_route)(f"local:{self.local.model}")
        yield from self.local._race(messages, on_open=on_open)

    def _cursor_first(self, messages: List[dict]) -> Generator[str, None, None]:
        """Composer 2.5 Fast answers. If it has not spoken by hedge_after_s, the fallback chain (OpenRouter, then
        local) starts beside it and the first to speak wins; if Composer fails outright, the fallback starts at
        once. The loser is cut. Raises only if every route started has failed before a first word."""
        events: queue.Queue = queue.Queue()
        stop = {"cursor": threading.Event(), "fallback": threading.Event()}
        handles: dict = {"cursor": [], "fallback": []}
        labels: dict = {}
        started: set = set()

        def pump(name: str, factory: Callable[[], Generator[str, None, None]]) -> None:
            try:
                for delta in factory():
                    if stop[name].is_set():
                        return
                    events.put(("tok", name, delta))
                events.put(("done", name, None))
            except Exception as e:
                events.put(("err", name, e))

        def start(name: str) -> None:
            if name in started:
                return
            started.add(name)
            if name == "cursor":
                factory = lambda: self.cursor._race(messages, on_open=handles["cursor"].append)  # noqa: E731
            else:
                factory = lambda: self._fallback_chain(  # noqa: E731
                    messages, on_open=handles["fallback"].append, on_route=lambda label: labels.update(fallback=label))
            threading.Thread(target=pump, args=(name, factory), daemon=True, name=f"vself-{name}").start()

        def cut(name: str) -> None:
            stop[name].set()
            for h in list(handles[name]):
                try:
                    h.close()
                except Exception:
                    pass

        t0 = time.monotonic()
        if self.cursor.available():
            start("cursor")
        else:
            self.cursor.warm()  # still starting or down: do not make this turn wait for it
            start("fallback")
        failed: dict = {}
        winner: Optional[str] = None
        first = ""
        while winner is None:
            if "fallback" not in started and time.monotonic() - t0 >= self.hedge_after_s:
                logger.info("composer silent for %.1fs: starting the fallback beside it", self.hedge_after_s)
                start("fallback")
            try:
                kind, name, val = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == "tok":
                winner, first = name, val
            else:
                failed[name] = val if kind == "err" else RuntimeError(f"{name} returned nothing")
                if name == "cursor" and "fallback" not in started:
                    start("fallback")
                if started <= set(failed):
                    raise failed.get("fallback") or failed["cursor"]
        loser = "fallback" if winner == "cursor" else "cursor"
        if loser in started:
            cut(loser)
        if winner == "cursor":
            self.last_route = f"cursor:{self.cursor.model}"
        else:
            self.last_route = labels.get("fallback", "cloud")  # only the winner names the route
            self.cursor.note_missed()
        yield first
        while True:
            kind, name, val = events.get()
            if name != winner:
                continue
            if kind == "tok":
                yield val
            elif kind == "done":
                return
            else:
                raise val

    def close(self) -> None:
        """Stop Apollo's background helpers (the Composer agent process)."""
        if self.cursor is not None:
            self.cursor.close()

    def _route(self, messages: List[dict]) -> Generator[str, None, None]:
        if self.prefer == "cursor":
            yield from self._cursor_first(messages)
            return
        if self.prefer == "cloud":
            yield from self._cloud_first(messages)
            return
        events: queue.Queue = queue.Queue()
        abandoned = {"local": threading.Event(), "cloud": threading.Event()}
        streams: dict = {"local": [], "cloud": []}
        started = set()

        def cut(name: str) -> None:
            """Abandon a route: close its open streams so a read blocked on the network ends now."""
            abandoned[name].set()
            for stream in list(streams[name]):
                try:
                    stream.close()
                except Exception:
                    pass

        t0 = time.monotonic()

        def pump(name: str, open_stream: Callable[[], Generator[str, None, None]]) -> None:
            try:
                for delta in open_stream():
                    if abandoned[name].is_set():
                        return
                    events.put(("tok", name, delta))
                events.put(("done", name, None))
            except Exception as e:
                events.put(("err", name, e))

        def start(name: str) -> None:
            if name in started or (name == "cloud" and self.cloud is None):
                return
            client = self.local if name == "local" else self.cloud
            started.add(name)
            threading.Thread(target=pump, args=(name, lambda: client._race(messages, on_open=streams[name].append)),
                             daemon=True, name=f"vself-{name}").start()

        def start_cloud_now() -> bool:
            if self.cloud is None:
                return False
            start("cloud")
            return True

        local_ok = self.local_available() or self.cloud is None
        if local_ok:
            start("local")
        # local missed its budget three turns running: give the cloud no head start,
        # but local still wins if it opens cleanly within the threshold
        if not local_ok or (len(self._local_misses) == 3 and all(self._local_misses)):
            start_cloud_now()

        buf = {"local": [], "cloud": []}
        done = set()
        failed = {}
        winner = None
        local_judged = False
        try:
            while winner is None:
                now = time.monotonic() - t0
                if "cloud" not in started and now >= self.threshold_s:
                    logger.info("local missed %.0fms first-token budget → cloud", self.threshold_s * 1000)
                    start_cloud_now()
                # wake at the threshold even if nothing arrives, to start or pick the cloud
                wait = self.threshold_s - now if now < self.threshold_s else None
                try:
                    kind, name, val = events.get(timeout=wait)
                    if kind == "tok":
                        buf[name].append(val)
                    elif kind == "done":
                        done.add(name)
                    else:
                        failed[name] = val
                        logger.warning("%s route failed: %s", name, val)
                        if name == "local":
                            start_cloud_now()
                except queue.Empty:
                    pass

                elapsed = time.monotonic() - t0
                local_text = "".join(buf["local"])
                if "local" not in failed and not local_judged and (
                        len(local_text) >= GUARD_CHARS or "local" in done):
                    local_judged = True
                    if not local_opening_ok(local_text):
                        logger.info("local opening rejected: %r → cloud", local_text[:GUARD_CHARS])
                        failed["local"] = ValueError("defective local opening")
                        cut("local")
                        start_cloud_now()
                local_good = local_judged and "local" not in failed
                if local_good and (elapsed <= self.threshold_s or not buf["cloud"]):
                    winner = "local"
                elif buf["cloud"] and (elapsed > self.threshold_s or "local" not in started
                                       or "local" in failed):
                    winner = "cloud"
                else:
                    exhausted = {n for n in started if n in failed or (n in done and not buf[n])}
                    cloud_left = self.cloud is not None and "cloud" not in started
                    if started and exhausted == started and not cloud_left:
                        raise failed.get("cloud") or failed.get("local") or RuntimeError(
                            "no route produced a reply")

            loser = "cloud" if winner == "local" else "local"
            cut(loser)
            if "local" in started:
                self._local_misses.append(winner != "local")
            self.last_route = f"{winner}:{(self.local if winner == 'local' else self.cloud).model}"
            yield from buf[winner]
            if winner in done:
                return
            while True:
                kind, name, val = events.get()
                if name != winner:
                    continue
                if kind == "tok":
                    yield val
                elif kind == "done":
                    return
                else:
                    raise val
        finally:
            for name in abandoned:  # also when the turn is dropped mid-reply (barge-in, speech resumed)
                cut(name)

    def complete(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        """Notes and warm-up stay on the local brain (they carry the conversation)."""
        return self.local.complete(system, user, max_tokens=max_tokens)

    def warm(self) -> None:
        """Load vself's embedder, cache the persona prefix on the local brain, open the cloud TLS, start Composer."""
        if self.cursor is not None:
            self.cursor.warm()
        try:
            messages = self.prepare("Hello.")
            if self.cursor is not None:
                self.cursor.prime(messages[0]["content"])
            r = self.local.client.chat.completions.create(
                model=self.local.model, messages=messages, max_tokens=1,
                extra_body=LOCAL_EXTRA_BODY)
            del r
        except Exception as e:
            logger.warning("vself warm-up failed: %s", e)
        if self.cloud is not None:
            try:
                self.cloud.complete("Reply with OK.", "ping", max_tokens=1)
            except Exception as e:
                logger.warning("cloud warm-up failed: %s", e)
