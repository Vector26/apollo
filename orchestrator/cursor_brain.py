"""Composer 2.5 Fast as Apollo's brain: one long-lived `cursor-agent acp` process, one session, text in / text out.

Why ACP and not `cursor-agent --print`: a fresh process per turn takes ~5 s to first word; a persistent session
takes ~1.5 s. The agent is boxed in: Ask mode (no edits, no commands), an empty sandbox directory, and every
permission request is refused. It is given no tools; when it should use one of Apollo's agents it writes the
same `[[ask:agent|question]]` line every other brain writes, and the brain turns that into the real call.

Voice needs speed, so around the model: a short style contract every turn, markdown stripped, a word cap that
stops at a sentence end, a hard first-word deadline, and an abort the moment it wanders into a tool. Any failure
raises before the first word, so the caller can fall back (OpenRouter, then local) without the owner noticing.

The persona is static, so it is sent once per session; each later turn carries only the owner's message
(vself already folds memories into it) plus a one-line reminder.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import queue
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional

logger = logging.getLogger(__name__)

MODEL_ID = "composer-2.5[fast=true]"
LABEL = "composer-2.5-fast"

VOICE_RULES = (
    "You are the voice brain of Apollo. Everything you write is spoken aloud, so: answer at once, do not "
    "deliberate or show reasoning; at most two short plain sentences unless asked for detail (then at most "
    "six); no markdown, lists, headings, code, emoji or links. You have no tools, no files and no shell: never "
    "search or run anything. Follow the persona below exactly."
)
REMINDER = "(Spoken reply: two short plain sentences, no markdown, no tools, no deliberating.)"


# cursor-agent reports a network or backend failure as ordinary reply text ("Error: RetriableError: [unavailable]
# getaddrinfo EAI_AGAIN ..."), which would be spoken and filed as Prometheus' words. It is a route failure.
_ERROR_REPLY = re.compile(r"\s*(?:(?:\w*Error|\w*Exception)\s*:|\[(?:unavailable|internal|unauthenticated)\])", re.I)


class CursorUnavailable(Exception):
    """The Composer route cannot take a turn right now (not started, crashed, still starting)."""


class CursorSlow(Exception):
    """It took a turn but gave no word in time, or wandered into tools. Raised before anything was spoken."""


class _Clean:
    """Streaming markdown stripper: what is left can be spoken. Never touches [[ask:agent|...]] lines."""

    def __init__(self) -> None:
        self._lead = True
        self._space = False

    def feed(self, delta: str) -> str:
        out: list[str] = []
        t = delta.replace("`", "").replace("*", "")
        t = re.sub(r"(?m)^\s*#{1,6}\s*", "", t)
        t = re.sub(r"(?m)^\s*(?:[-•]|\d+[.)])\s+", " ", t)
        for ch in t.replace("\r", ""):
            if ch in "\n\t ":
                if not self._lead:
                    self._space = True
                continue
            if self._space:
                out.append(" ")
                self._space = False
            self._lead = False
            out.append(ch)
        return "".join(out)


class _Conn:
    """One `cursor-agent acp` process: newline-delimited JSON-RPC, a reader thread, refuses every permission."""

    def __init__(self, cmd: list[str], cwd: str, popen: Callable[..., Any] = subprocess.Popen,
                 stderr_path: Optional[Path] = None) -> None:
        self._stderr = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        self.proc = popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, text=True,
                          cwd=cwd, start_new_session=True)
        self.updates: queue.Queue = queue.Queue()
        self._pending: dict[int, tuple[threading.Event, dict]] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self.dead = threading.Event()
        threading.Thread(target=self._read, daemon=True, name="acp-read").start()

    @property
    def alive(self) -> bool:
        return not self.dead.is_set() and self.proc.poll() is None

    def _write(self, obj: dict) -> None:
        with self._lock:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def _read(self) -> None:
        try:
            for line in self.proc.stdout:
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if "method" in m and "id" in m:
                    self._on_agent_request(m)
                elif "method" in m:
                    if m["method"] == "session/update":
                        self.updates.put(m.get("params", {}).get("update", {}))
                elif "id" in m:
                    slot = self._pending.get(m["id"])
                    if slot:
                        slot[1].update(m)
                        slot[0].set()
        except Exception:
            logger.exception("acp reader failed")
        finally:
            self.dead.set()
            for ev, box in list(self._pending.values()):
                box.setdefault("error", {"message": "agent process ended"})
                ev.set()
            self.updates.put({"sessionUpdate": "_dead"})

    def _on_agent_request(self, m: dict) -> None:
        """Nothing the agent asks for is ever allowed: no tools, no files, no terminal."""
        try:
            if m["method"] == "session/request_permission":
                opts = m.get("params", {}).get("options", [])
                reject = next((o.get("optionId") for o in opts if "reject" in str(o.get("kind", ""))), None)
                result = ({"outcome": {"outcome": "selected", "optionId": reject}} if reject
                          else {"outcome": {"outcome": "cancelled"}})
                self.updates.put({"sessionUpdate": "_permission_refused"})
                self._write({"jsonrpc": "2.0", "id": m["id"], "result": result})
            else:
                self._write({"jsonrpc": "2.0", "id": m["id"], "error": {"code": -32601, "message": "not supported"}})
        except Exception:
            logger.exception("could not answer an agent request")

    def request_async(self, method: str, params: dict) -> tuple[threading.Event, dict]:
        with self._lock:
            self._next_id += 1
            rid = self._next_id
        slot = (threading.Event(), {})
        self._pending[rid] = slot
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return slot

    def request(self, method: str, params: dict, timeout: float = 20.0) -> dict:
        ev, box = self.request_async(method, params)
        if not ev.wait(timeout):
            raise CursorUnavailable(f"{method} timed out")
        if "error" in box:
            raise CursorUnavailable(f"{method}: {box['error'].get('message', box['error'])}")
        return box.get("result", {})

    def notify(self, method: str, params: dict) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def kill(self) -> None:
        pgid = self.proc.pid
        for sig, wait in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 0.5)):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError, OSError):
                break
            end = time.monotonic() + wait
            while time.monotonic() < end and self.proc.poll() is None:
                time.sleep(0.05)
            if self.proc.poll() is not None:
                break
        self.dead.set()


class _TurnHandle:
    """What the brain closes to cut this turn (a hedge lost to the other route, or a barge-in)."""

    def __init__(self, turn: dict) -> None:
        self._turn = turn

    def close(self) -> None:
        self._turn["superseded"] = True


class CursorRoute:
    """Same interface as LLMClient for the brain's router: ``model`` and ``_race(messages, on_open)``."""

    def __init__(self, sandbox_dir: str | Path, state_dir: str | Path, agent_bin: str = "cursor-agent",
                 model_id: str = MODEL_ID, give_up_s: float = 6.0, ready_wait_s: float = 1.0, max_words: int = 140,
                 rotate_turns: int = 24, popen: Callable[..., Any] = subprocess.Popen) -> None:
        self.model = LABEL
        self.sandbox = Path(sandbox_dir)
        self.state_dir = Path(state_dir)
        self.agent_bin = agent_bin
        self.model_id = model_id
        self.give_up_s = give_up_s
        self.ready_wait_s = ready_wait_s
        self.max_words = max_words
        self.rotate_turns = rotate_turns
        self._popen = popen
        self._conn: Optional[_Conn] = None
        self._session: Optional[str] = None
        self._ready = threading.Event()
        self._booting = False
        self._next_boot = 0.0
        self._backoff = 2.0
        self._lock = threading.Lock()
        self._primed_hash: Optional[str] = None
        self._pending_session: Optional[str] = None
        self._rotating = False
        self._turns = 0
        self._last_reply: Optional[str] = None  # what this session last said, to spot turns it never saw
        self._active: Optional[dict] = None
        atexit.register(self.close)

    # ── lifecycle ──────────────────────────────────────────────────────
    def warm(self) -> None:
        """Start the agent in the background (idempotent, rate-limited). Never blocks, never raises."""
        with self._lock:
            if self._booting or (self._conn is not None and self._conn.alive and self._ready.is_set()):
                return
            if time.monotonic() < self._next_boot:
                return
            self._booting = True
        threading.Thread(target=self._boot, daemon=True, name="acp-boot").start()

    def _sweep_stale(self) -> None:
        """A previous Apollo that was killed hard leaves its agent behind: end that group before starting ours."""
        pidfile = self.state_dir / "acp.pgid"
        try:
            old = int(pidfile.read_text().strip())
            if Path(self.agent_bin).name in Path(f"/proc/{old}/cmdline").read_text().replace("\0", " "):
                os.killpg(old, signal.SIGTERM)
                logger.warning("ended a stale cursor-agent group %d from an earlier run", old)
        except (OSError, ValueError):
            pass

    def _boot(self) -> None:
        conn: Optional[_Conn] = None
        try:
            self.sandbox.mkdir(parents=True, exist_ok=True)
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self._sweep_stale()
            old, self._conn = self._conn, None
            if old is not None:
                old.kill()
            self._ready.clear()
            conn = _Conn([self.agent_bin, "acp"], str(self.sandbox), self._popen, self.state_dir / "acp.stderr.log")
            (self.state_dir / "acp.pgid").write_text(str(conn.proc.pid))
            init = conn.request("initialize", {"protocolVersion": 1, "clientCapabilities": {}}, 15)
            methods = [a.get("id") for a in init.get("authMethods", [])]
            if methods:
                conn.request("authenticate", {"methodId": methods[0]}, 20)
            self._conn = conn
            self._session, self._primed_hash, self._turns = self._create_session(conn), None, 0
            self._backoff = 2.0
            self._ready.set()
            logger.info("composer route ready (model %s)", self.model_id)
        except Exception as e:
            logger.warning("composer route failed to start: %s", e)
            if conn is not None:
                conn.kill()
            self._conn = None
            self._next_boot = time.monotonic() + self._backoff
            self._backoff = min(self._backoff * 2, 60.0)
        finally:
            self._booting = False

    def _create_session(self, conn: _Conn) -> str:
        res = conn.request("session/new", {"cwd": str(self.sandbox), "mcpServers": []}, 20)
        sid = res["sessionId"]
        conn.request("session/set_mode", {"sessionId": sid, "modeId": "ask"}, 10)  # no edits, no commands
        current = (res.get("models") or {}).get("currentModelId", "")
        if current != self.model_id:
            conn.request("session/set_model", {"sessionId": sid, "modelId": self.model_id}, 10)
        return sid

    def _rotate_bg(self) -> None:
        """Build a fresh session off to the side; the next turn swaps to it. Never delays a turn."""
        conn = self._conn
        if conn is None or not conn.alive or self._rotating:
            return
        self._rotating = True

        def run() -> None:
            try:
                sid = self._create_session(conn)
                with self._lock:
                    self._pending_session = sid
            except Exception:
                logger.warning("could not build a fresh composer session; keeping the old one")
            finally:
                self._rotating = False

        threading.Thread(target=run, daemon=True, name="acp-rotate").start()

    def available(self) -> bool:
        return self._ready.is_set() and self._conn is not None and self._conn.alive

    def close(self) -> None:
        conn, self._conn = self._conn, None
        self._ready.clear()
        if conn is not None:
            conn.kill()
        try:
            (self.state_dir / "acp.pgid").unlink()
        except OSError:
            pass

    def prime(self, system: str) -> None:
        """Send the persona once in the background, right after start-up, so the owner's first turn is as fast as
        the rest (the persona turn is the slow one). Never blocks, never raises."""
        def run() -> None:
            try:
                end = time.monotonic() + 20
                while not self.available() and time.monotonic() < end:
                    self.warm()
                    time.sleep(0.3)
                if self.available():
                    "".join(self._race([{"role": "system", "content": system},
                                         {"role": "user", "content": "(setup) Reply with the single word: ready."}]))
                    self._last_reply = None
            except Exception as e:
                logger.warning("composer priming failed: %s", e)

        threading.Thread(target=run, daemon=True, name="acp-prime").start()

    def note_missed(self) -> None:
        """Another brain answered a turn: the next prompt re-sends the recent conversation."""
        self._last_reply = None

    def new_conversation(self) -> None:
        """A new conversation starts clean: a fresh session (built in the background), not a re-primed old one."""
        self._last_reply = None
        self._rotate_bg()

    # ── one turn ───────────────────────────────────────────────────────
    def _compose(self, messages: list[dict], system_hash: str) -> str:
        # vself ends a turn with a run of user messages: the owner's words with their memories folded in, then its own
        # reminder. All of that is the turn; what came before it is history.
        cut = len(messages)
        while cut > 1 and messages[cut - 1].get("role") == "user":
            cut -= 1
        cut = min(cut, len(messages) - 1)
        user = "\n\n".join(m["content"] for m in messages[cut:] if m.get("content"))
        history = [m for m in messages[1:cut] if m.get("content")]
        parts: list[str] = []
        recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-8:])
        if self._primed_hash != system_hash:
            parts.append(VOICE_RULES + "\n\n<persona>\n" + messages[0]["content"] + "\n</persona>")
            if history:
                parts.append("Conversation so far:\n" + recent)
        else:
            if history and self._unseen(history):
                parts.append("Conversation so far (some of it happened without you):\n" + recent)
            parts.append(REMINDER)
        parts.append(f"Owner: {user}")
        return "\n\n".join(parts)

    def _unseen(self, history: list[dict]) -> bool:
        """True when the conversation moved on without this session: the last thing said (a tool's answer, a note,
        another brain's reply) is not what this session last said."""
        last = next((m["content"] for m in reversed(history) if m.get("role") == "assistant"), None)
        return last is None or self._last_reply is None or " ".join(last.split()) != self._last_reply

    def _ready_conn(self) -> _Conn:
        if not self.available():
            if self._conn is not None and not self._conn.alive:
                self._ready.clear()
            self.warm()
            if not self._ready.wait(self.ready_wait_s) or not self.available():
                raise CursorUnavailable("composer route is not ready")
        assert self._conn is not None
        return self._conn

    def _stop_previous(self, conn: _Conn) -> None:
        """A newer turn replaces one still running (a speculative turn the owner talked over)."""
        prev = self._active
        if prev is None:
            return
        prev["superseded"] = True
        try:
            conn.notify("session/cancel", {"sessionId": prev["session"]})
        except Exception:
            pass
        prev["finished"].wait(2.0)

    def _race(self, messages: list[dict], on_open: Optional[Callable[[Any], None]] = None) -> Generator[str, None, None]:
        conn = self._ready_conn()
        with self._lock:
            self._stop_previous(conn)
            if self._pending_session:
                self._session, self._pending_session = self._pending_session, None
                self._primed_hash, self._turns = None, 0
            turn = {"session": self._session, "superseded": False, "finished": threading.Event()}
            self._active = turn
        sid = turn["session"]
        if on_open is not None:
            on_open(_TurnHandle(turn))
        system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        system_hash = hashlib.sha1(system.encode()).hexdigest()
        prompt = self._compose(messages, system_hash)
        while not conn.updates.empty():
            conn.updates.get_nowait()  # nothing left over from an earlier turn
        ev, box = conn.request_async("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": prompt}]})
        clean, spoken, words, t0, cancelled, said = _Clean(), False, 0, time.monotonic(), False, []
        probe = ""  # the raw start of the reply, until a first word has been spoken

        def cancel() -> None:
            nonlocal cancelled
            if not cancelled:
                cancelled = True
                try:
                    conn.notify("session/cancel", {"sessionId": sid})
                except Exception:
                    pass

        try:
            while True:
                if turn["superseded"]:
                    return
                try:
                    u = conn.updates.get(timeout=0.1)
                except queue.Empty:
                    u = None
                if u is not None:
                    kind = u.get("sessionUpdate")
                    if kind == "_dead":
                        raise CursorUnavailable("agent process ended")
                    if kind in ("tool_call", "tool_call_update", "_permission_refused"):
                        cancel()  # it has no tools; reaching for one means it is wandering
                        if not spoken:
                            raise CursorSlow("composer reached for a tool")
                        return
                    if kind == "agent_message_chunk":
                        raw = (u.get("content") or {}).get("text", "")
                        if not spoken:
                            probe += raw
                            if _ERROR_REPLY.match(probe):
                                cancel()
                                raise CursorUnavailable(f"agent reported: {' '.join(probe.split())[:160]}")
                        text = clean.feed(raw)
                        if text:
                            if not spoken:
                                self._primed_hash = system_hash  # the persona has landed
                            spoken = True
                            said.append(text)
                            words += len(text.split())
                            yield text
                            if words >= self.max_words and re.search(r"[.!?]\s*$", text):
                                cancel()  # long enough for speech: stop at a sentence end
                                return
                    continue
                if ev.is_set():
                    if "error" in box:
                        if not spoken:
                            raise CursorUnavailable(str(box["error"].get("message", box["error"])))
                        return
                    if conn.updates.empty():
                        return
                    continue
                if not spoken and time.monotonic() - t0 > self.give_up_s:
                    cancel()
                    raise CursorSlow(f"no word in {self.give_up_s:.0f}s")
        finally:
            if said:
                self._last_reply = " ".join("".join(said).split())
            if not ev.is_set():
                cancel()  # barge-in, a lost hedge, or an error: free the session now
                ev.wait(1.0)
            turn["finished"].set()
            with self._lock:
                if self._active is turn:
                    self._active = None
            self._turns += 1
            if self._turns >= self.rotate_turns and self.available():
                self._rotate_bg()  # a fresh session keeps the context short; the persona is re-sent on it
