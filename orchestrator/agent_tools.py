"""Apollo's tools: what the live assistant may do besides talk.

Apollo is the hub. The specialist agents (Detective, Dexter, ODA, DUM-E, flowsmith) each do one job behind
the agent gateway; Apollo only routes to them, by name first and then by what the request is about.
There is no shell here. Every tool is a fixed, named action:

  notify       one message to the owner's phone through the notifier
  notes        list / read / create / append / delete (to .trash) inside one directory, never outside it
  ask agent    a chat turn with a specialist, answer spoken back
  run job      start, check or cancel a gateway job (long ones are confirmed first and remembered on disk)
  status       what the agents and jobs are doing

Two phases, because the voice loop generates speculatively and may throw a turn away:
  plan(text)  reads only: decides what to say and what would be done. Safe to call and discard.
  apply()     the side effect, run once the turn is confirmed (see VselfBrain.commit).

Risky actions (delete a note, start a job, anything on flowsmith, and by default a message to the phone or words
added to an existing note: see CONFIRM_DEFAULT and the tools.confirm config) are never done on the first ask: Apollo
says what it would do and waits for a spoken "yes". DUM-E stays propose-only: Apollo can read its status,
but a change is something the owner asks DUM-E for in their own terminal.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import httpx
import yaml

from halt import is_halted
from llm_client import read_env_var
from plane_client import PlaneClient, PlaneError
from reminder_client import ReminderClient, ReminderError

logger = logging.getLogger(__name__)

CONFIRM_TTL_S = 45.0
# actions that are spoken back and wait for "yes" even though they are not destructive: they leave the host (a
# message to the phone) or change something that cannot be undone (words added to an existing note)
CONFIRM_DEFAULT = frozenset({"notify", "notes_write"})
CHAT_TIMEOUT_S = 100.0
READ_JOB_WAIT_S = 25.0
INLINE_WAIT_S = 8.0       # an agent that answers within this is this turn's reply; a slower one is announced when it lands
JOB_POLL_S = 5.0
JOB_WATCH_MAX_S = 3 * 3600
JOB_RESUME_MAX_AGE_S = 24 * 3600
AGENT_SPEAK_CHARS = 900   # an agent's own answer is read out whole (markdown stripped), not summarised
SPEAK_CHARS = 380
NOTIFY_CHARS = 500
VOICE_AGENTS = ("detective", "dexter", "oda", "dume", "flowsmith")  # virtual-self is Apollo's own memory
# Job types the voice may start. Destructive or open-ended ones (delete_workflow, migrate_pipeline, tool, task)
# are not here on purpose: those need the owner at a terminal.
VOICE_JOB_TYPES = {
    "detective": ("run", "resume", "nightly", "status"),
    "oda": ("status", "previews"),
    "flowsmith": ("list_workflows",),
    "dume": ("snapshot",),
}
# How each agent sounds to the speech recogniser. Extend in config.yaml under tools.aliases (agent: [spellings]).
ALIASES = {
    "detective": ("the detective", "detective"),
    "dexter": ("dexter", "dexta"),
    "oda": ("oda", "o d a", "odah", "o.d.a"),
    "dume": ("dume", "dum-e", "dum e", "dummy", "dumb e", "dumb-e", "doom e", "doom-e", "diom", "dium", "dom e", "dumi"),
    "flowsmith": ("flowsmith", "flow smith", "flow-smith", "floe smith"),
}
BLURBS = {
    "detective": "web research with cited sources: it browses the web through its own scouts, for news, facts, films and anything to look up",
    "dexter": "finance and markets research",
    "oda": "design and previews",
    "dume": "server status; it only proposes changes, in your terminal",
    "flowsmith": "n8n workflows, which is experimental",
}
CHANGE_WORDS = re.compile(r"\b(restart|reboot|stop|start|kill|delete|remove|clean|prune|install|update|upgrade|fix|change|"
                          r"shut ?down|enable|disable|stop|mount|unmount|clear|wipe|reset)\b")
PRONOUN_ASK = re.compile(r"^(?:ask|tell|check with) (?:it|them|him|her|that|the agent)(?: for me| to do that| about that| about it)?$")
OPS = (r"(?:machine|server|fortress|computer|system|gpu|vram|disks?|storage|memory|ram|containers?|services?|"
       r"ai brain|brain|cpu|load|uptime)")
STRONG_MACHINE = re.compile(r"\b(?:gpu|vram|ram|cpu|disks?|hard ?(?:disk|drive)s?|ssd|nvme|docker|containers?|uptime|ai brain|"
                            r"memory (?:usage|used|use|left|free|available)|how much memory|(?:using|eating|hogging) (?:the |my )?memory)\b")
WEAK_MACHINE = re.compile(r"\b(?:machine|server|fortress|computer|system|storage|load|services?)\b")
STATE_WORDS = re.compile(r"\b(?:status|state|health|healthy|doing|going on|holding up|holding|running|ok|okay|fine|up|down|busy|"
                         r"free|full|left|using|used|usage|space|slow|hot|temperature|crashed|broken|alive|online)\b")
ADVICE = re.compile(r"\b(?:would you|should (?:i|we)|do you think|how (?:do|can|could|should) (?:i|we)|how to|recommend|advice|"
                    r"suggest|i keep|any ideas|what if)\b")
OPINION = re.compile(r"\b(?:think|opinion|feel|believe|government|economy|economic|political|society|philosophy|love|story)\b")
ASKY = re.compile(r"\b(?:what|what's|which|how|how's|tell|check|show|list|give|is|are|any|does|do|whats)\b")
FINANCE = r"(?:(?<!in )stocks?|share price|shares|markets?|nifty|sensex|crypto|bitcoin|ethereum|earnings|etfs?|portfolio)"
# DUM-E is asked for plain speech, then read out as it says it: nothing is summarised on the way.
VOICE_HINT = " (This will be read aloud: answer in plain spoken sentences, no tables, lists, markdown or symbols.)"
QUESTION_SHAPE = re.compile(r"\b(?:what|which|how|why|is|are|can|could|list|show|tell|check|status|any|do|does|give)\b")
ROSTER = re.compile(r"^(?:what|which)\b.*\b(?:agents?|tools)\b.*\b(?:you|your|there|available|handles?)\b"
                    r"|^(?:do you|are you)\b.*\b(?:agents?|tools)\b|^(?:list|name) (?:the |all |your )?agents")
CONFIRM_YES = re.compile(r"^(?:yes|yeah|yep|yup|sure|ok|okay|confirm|confirmed|do it|go ahead|go for it|please do|affirmative)\b")
CONFIRM_NO = re.compile(r"^(?:no|nope|nah|cancel|don'?t|do not|never ?mind|stop|abort|forget it)\b")
DAILY_NOTE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")  # written by Virtual Self; Apollo reads these, never edits them
LEAD = re.compile(r"^(?:(?:hey|ok|okay|so|and|but|also|then|well|now|just|actually|alright|anyways?|anyhow|um|uh|yo|"
                  r"apollo|jarvis|prometheus|please|can you|could you|would you|will you|i want you to|"
                  r"i would like you to|i'd like you to|i want to know|i need to know|i wanna know|no,? i mean|i mean|you can|you could|"
                  r"you should|go ahead and|if you can),?\s+)+")


CALL_RE = re.compile(r"^\s*\[\[\s*ask\s*:\s*([a-z]+)\s*\|\s*(.+?)\s*\]\]", re.S | re.I)
PLANE_CALL = re.compile(r"^\s*\[\[\s*plane\s*:\s*(\w+)\s*\|\s*(.*?)\s*\]\]", re.S | re.I)
PLANE_ANY = re.compile(r"\[\[\s*plane\s*:\s*(\w+)\s*\|\s*(.*?)\s*\]\]", re.S | re.I)
CALLABLE_AGENTS = ("dume", "detective", "dexter", "oda")  # flowsmith is confirm-gated: never started by a bare model line
PLANE_WORDS = re.compile(r"\b(plane|ticketing|ticketing system|work items?)\b")
REMIND_CALL = re.compile(r"^\s*\[\[\s*remind\s*:\s*(\w+)\s*\|\s*(.*?)\s*\]\]", re.S | re.I)
REMIND_ANY = re.compile(r"\[\[\s*remind\s*:\s*(\w+)\s*\|\s*(.*?)\s*\]\]", re.S | re.I)
REMIND_WORDS = re.compile(r"\b(remind(?:er)?s?|timed reminder|alarm)\b")
VOICE_POLL_S = 2.0  # how often the live loop asks the notifier for a due spoken reminder
PLANE_LIST_ASK = re.compile(r"\b(what|which|list|show|any|how many|tell me|give me)\b")
PLANE_NOUNS = re.compile(r"\b(tickets?|issues|work items?|projects?|board|backlog|open|todo|to-do)\b")
PLANE_WRITE_VERB = re.compile(r"\b(create|add|delete|remove|update|rename|move|mark|set|close)\b")


# A reply that promises, offers or tells the owner to ask an agent is a call that was never made: make it.
_AGENT_WORD = r"(dum[\s-]?e|dume|dummy|dexter|detective|oda)"
PROMISE_RE = re.compile(
    r"^\W*(?:(?:i(?:'ll|\u2019ll| will| can| could| am going to|'m going to|\u2019m going to)|let me|let's|you can|you could|"
    r"you should|you'd need to|please|try to|just|go)\s+)*(?:ask|check with|query|call)\s+(?:the\s+)?" + _AGENT_WORD + r"\b", re.I)
HEDGE_START = re.compile(
    r"^\W*(?:i(?:'ll|\u2019ll| will| can(?:'|\u2019)?t?| cannot| don(?:'|\u2019)t| do not| could| am going|(?:'|\u2019)m going| am not|"
    r"(?:'|\u2019)m not)|let me|let's|sorry|unfortunately|ask |you can|you could|you should|please|try |just )", re.I)
_AGENT_IDS = {"dum": "dume", "dexter": "dexter", "detective": "detective", "oda": "oda"}


def promised_agent(text: str) -> Optional[str]:
    """The agent a reply says it will ask (or tells the owner to ask), if any of its first sentences does."""
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip())[:2]:
        m = PROMISE_RE.match(sentence)
        if m:
            word = m.group(1).lower()
            return _AGENT_IDS["dum"] if word.startswith("dum") else _AGENT_IDS.get(word)
    return None


CALL_ANY = re.compile(r"\[\[\s*ask\s*:\s*([a-z]+)\s*\|\s*(.+?)\s*\]\]", re.S | re.I)


def find_call(text: str) -> Optional[tuple[str, str]]:
    """The first valid agent call line anywhere in a reply (a model sometimes writes a sentence before it)."""
    for m in CALL_ANY.finditer(text):
        call = parse_call(m.group(0))
        if call:
            return call
    return None


def parse_plane_call(text: str) -> Optional[tuple[str, str]]:
    m = PLANE_CALL.match(text.strip())
    if not m:
        return None
    action, args = m.group(1).lower(), re.sub(r"\s+", " ", m.group(2)).strip()
    if action not in ("list", "get", "read", "create", "update", "delete", "move", "comment"):
        return None
    return action, args


def find_plane_call(text: str) -> Optional[tuple[str, str]]:
    for m in PLANE_ANY.finditer(text):
        call = parse_plane_call(m.group(0))
        if call:
            return call
    return None


def parse_remind_call(text: str) -> Optional[tuple[str, str]]:
    m = REMIND_CALL.match(text.strip())
    if not m:
        return None
    action, args = m.group(1).lower(), re.sub(r"\s+", " ", m.group(2)).strip()
    if action not in ("list", "get", "in", "at", "daily", "weekly", "update", "delete", "enable", "disable", "cancel"):
        return None
    return action, args


def find_remind_call(text: str) -> Optional[tuple[str, str]]:
    for m in REMIND_ANY.finditer(text):
        call = parse_remind_call(m.group(0))
        if call:
            return call
    return None


def parse_call(text: str) -> Optional[tuple[str, str]]:
    """A reply that is really a request for an agent: [[ask:dume|how much RAM is in use]]."""
    m = CALL_RE.match(text)
    if not m:
        return None
    agent, question = m.group(1).lower(), re.sub(r"\s+", " ", m.group(2)).strip()
    if agent not in CALLABLE_AGENTS or not question or len(question) > 400:
        return None
    return agent, question


class ToolError(Exception):
    """A failure with a sentence Apollo can say out loud."""

    def __init__(self, spoken: str) -> None:
        super().__init__(spoken)
        self.spoken = spoken


@dataclass
class Plan:
    speak: str
    tool: str
    apply: Optional[Callable[[], None]] = None
    ack: str = ""  # said at once while a slow agent works
    lazy: Optional[Callable[[], "Plan"]] = None  # the slow part: finish() runs it after the ack


@dataclass
class Pending:
    tool: str
    prompt: str
    apply: Callable[[], None]
    replan: Optional[Callable[[], "Plan"]] = None  # "did you mean X?" -> yes -> do the original request with X
    expires: float = field(default_factory=lambda: time.monotonic() + CONFIRM_TTL_S)


def skeleton(word: str) -> str:
    """A rough sound-alike key: dume, dummy, dum-e and diom all become 'dm'; dexter and 'decks ter' become 'dkstr'."""
    w = re.sub(r"[^a-z]", "", word.lower())
    w = w.replace("ph", "f").replace("ck", "k").replace("x", "ks").replace("q", "k").replace("c", "k").replace("z", "s")
    w = re.sub(r"mb$", "m", w)
    w = re.sub(r"[aeiouyhw]", "", w)
    return re.sub(r"(.)\1+", r"\1", w)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def summarize_snapshot(text: str, cores: int | None = None) -> Optional[str]:
    """DUM-E's report as a few plain sentences: a verdict first, no decimals. None if it can't be read."""
    cores = cores or os.cpu_count() or 1
    facts: list[str] = []
    issues: list[str] = []
    if m := re.search(r"load average: ([\d.]+)", text):
        ratio = float(m.group(1)) / cores
        if ratio >= 1:
            issues.append("it is under heavy load")
        else:
            facts.append("the load is " + ("light" if ratio < 0.5 else "moderate"))
    if m := re.search(r"^Mem:\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+([\d.]+)(Gi|Mi|Ti)", text, re.M):
        gb = float(m.group(1)) * {"Ti": 1024, "Gi": 1, "Mi": 1 / 1024}[m.group(2)]
        if gb < 4:
            issues.append("memory is running low")
        else:
            facts.append("about %d gigabytes of memory are free" % round(gb))
    if m := re.search(r"^\S+\s+\S+\s+\S+\s+\S+\s+(\d+)%\s+/$", text, re.M):
        pct = int(m.group(1))
        if pct >= 90:
            issues.append(f"the main disk is {pct} percent full")
        else:
            facts.append(f"the main disk is {pct} percent full")
    if m := re.search(r"VRAM used (\d+) / (\d+) MiB", text):
        facts.append(f"the GPU is using about {round(int(m.group(1)) / 1024)} of its {round(int(m.group(2)) / 1024)} gigabytes")
    if m := re.search(r"^AI brain: (.+?)(?: \(|,|$)", text, re.M):
        if m.group(1).strip().lower() != "running":
            issues.append(f"the AI brain is {m.group(1).strip()}")
        else:
            facts.append("the AI brain is running")
    if not facts and not issues:
        return None
    verdict = "The machine looks fine." if not issues else "The machine is up, but " + " and ".join(issues) + "."
    if not facts:
        return verdict
    body = ", ".join(facts)
    return f"{verdict} {body[0].upper() + body[1:]}."


UNITS = ((r"\b(\d+(?:\.\d+)?)\s?(?:GiB|GB|G)\b", r"\1 gigabytes"), (r"\b(\d+(?:\.\d+)?)\s?(?:MiB|MB)\b", r"\1 megabytes"),
         (r"\b(\d+(?:\.\d+)?)\s?%", r"\1 percent"), (r"~\s?(?=\d)", "about "))


def smooth_numbers(text: str) -> str:
    """Say numbers the way a person would: units spelled out, long decimals trimmed to one place."""
    for pat, rep_ in UNITS:
        text = re.sub(pat, rep_, text)
    return re.sub(r"\b(\d+)\.(\d)\d+\b", lambda m: f"{m.group(1)}.{m.group(2)}", text)


def speakable(text: str, limit: int = SPEAK_CHARS) -> str:
    """Markdown-free, short, ends on a sentence. Agent output is data to be spoken, never instructions."""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"[`*_#>|]+", "", t)
    t = re.sub(r"https?://\S+", "a link", t)
    t = smooth_numbers(re.sub(r"\s+", " ", t).strip())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end > limit // 2:
        return cut[: end + 1]
    return cut.rsplit(" ", 1)[0] + "…"


class AgentTools:
    def __init__(
        self,
        gateway_url: str,
        token: str,
        registry_path: str | Path,
        notifier_url: str,
        notes_dir: str | Path,
        state_dir: str | Path,
        http: httpx.Client | None = None,
        aliases: dict[str, list[str]] | None = None,
        confirm: Iterable[str] | None = None,
        plane: PlaneClient | None = None,
        reminders: ReminderClient | None = None,
    ) -> None:
        self.confirm: frozenset[str] = CONFIRM_DEFAULT if confirm is None else frozenset(confirm)
        self.plane = plane
        self.reminders = reminders
        self._last_voice_poll = 0.0
        self.aliases: dict[str, tuple[str, ...]] = {
            aid: tuple(dict.fromkeys([*names, *[a.lower() for a in (aliases or {}).get(aid, [])]]))
            for aid, names in ALIASES.items()
        }
        self.gateway_url = gateway_url.rstrip("/")
        self.token = token
        self.registry_path = Path(registry_path)
        self.notifier_url = notifier_url
        self.notes_dir = Path(notes_dir)
        self.state_dir = Path(state_dir)
        self.http = http or httpx.Client(timeout=10.0)
        self.pending: Optional[Pending] = None
        self.outbox: "queue.Queue[str]" = queue.Queue()  # things to say when the owner next is free: late answers, job results
        self.inline_wait_s = INLINE_WAIT_S
        self.job_poll_s = JOB_POLL_S
        self._prev: Optional[str] = None  # what the owner said last, for "ask it for me"
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: dict) -> "AgentTools":
        token = Path(cfg["token_file"]).expanduser().read_text().strip()
        plane = None
        pc = cfg.get("plane") or {}
        if pc.get("enabled", True):
            key = read_env_var(pc.get("api_key_env", "PLANE_API_KEY"), pc.get("api_key_file"))
            slug, pid = pc.get("workspace_slug"), pc.get("project_id")
            if key and slug and pid:
                plane = PlaneClient(
                    base_url=pc.get("base_url", "http://127.0.0.1:8082/api/v1"),
                    api_key=key,
                    workspace_slug=slug,
                    project_id=pid,
                    project_label=pc.get("project_label", "Plane"),
                )
            elif pc.get("required"):
                logger.warning("plane.enabled but PLANE_API_KEY or project ids missing")
        reminders = None
        rc = cfg.get("reminders") or {}
        if rc.get("enabled", True):
            base = rc.get("base_url") or cfg.get("notifier_url", "http://127.0.0.1:8001/notify").rsplit("/notify", 1)[0]
            reminders = ReminderClient(base_url=base, tz=rc.get("tz", "UTC"))
        return cls(
            gateway_url=cfg.get("gateway_url", "http://127.0.0.1:8190"),
            token=token,
            registry_path=Path(cfg["registry"]).expanduser(),
            notifier_url=cfg.get("notifier_url", "http://127.0.0.1:8001/notify"),
            notes_dir=Path(cfg["notes_dir"]).expanduser(),
            state_dir=Path(cfg["state_dir"]).expanduser(),
            aliases=cfg.get("aliases"),
            confirm=cfg.get("confirm"),
            plane=plane,
            reminders=reminders,
        )

    # ── entry points ───────────────────────────────────────────────────
    def announce(self, text: str) -> None:
        """Queue something to say unprompted (the voice loop speaks it when the owner is not mid-conversation)."""
        self.outbox.put(text)

    def has_announcement(self) -> bool:
        return not self.outbox.empty()

    def take_announcement(self) -> Optional[str]:
        try:
            return self.outbox.get_nowait()
        except queue.Empty:
            return None

    def take_notice(self) -> Optional[str]:
        """The brain asks before each answer, so a late result is never lost if the loop has not spoken it yet."""
        return self.take_announcement()

    def finish(self, plan: Plan) -> Plan:
        """Run the slow part of a plan (a specialist's answer). Failures become something sayable."""
        if not plan.lazy:
            return plan
        try:
            return plan.lazy()
        except ToolError as e:
            return Plan(e.spoken, plan.tool)
        except Exception:
            logger.exception("tool %s failed", plan.tool)
            return Plan("That didn't work.", plan.tool)

    def call_agent(self, agent: str, question: str, by: str = "model") -> Optional[Plan]:
        """The language model asked for an agent itself. Same path, same limits, as when the owner names it.
        ``by`` says which brain made the call; the audit log keeps it with the agent's full reply."""
        if agent not in CALLABLE_AGENTS or agent not in self._present() or is_halted():
            return None
        return self._ask(agent, question, by)

    def plan(self, text: str, resolve: bool = True) -> Optional[Plan]:
        """None when this is just conversation. Reads only; the returned apply() does the side effect.

        With resolve=False a slow agent call comes back as an ack plus lazy(), so the voice can speak at once."""
        plan = self._plan(text)
        return self.finish(plan) if plan and resolve else plan

    def _plan(self, text: str) -> Optional[Plan]:
        t = self._normalize(text)
        if not t:
            return None
        if PRONOUN_ASK.match(t):
            target, self._prev = self._prev, None
            return self._plan(target) if target else None
        if not (CONFIRM_YES.match(t) or CONFIRM_NO.match(t)):
            self._prev = t
        try:
            with self._lock:
                pending = self.pending
            if pending is not None:
                if time.monotonic() > pending.expires:
                    self._clear_pending()
                elif CONFIRM_YES.match(t):
                    self._clear_pending()
                    if pending.replan:
                        return pending.replan()
                    return Plan("On it.", pending.tool, pending.apply)
                elif CONFIRM_NO.match(t):
                    self._clear_pending()
                    return Plan("Okay, dropped it.", pending.tool)
                else:
                    self._clear_pending()  # the owner moved on; never trap them in a question
            if is_halted():
                return None
            for parse in (self._p_roster, self._p_notify, self._p_notes, self._p_plane, self._p_remind, self._p_status,
                          self._p_cancel, self._p_job, self._p_ask, self._p_mention, self._p_auto):
                plan = parse(t)
                if plan:
                    return plan
        except ToolError as e:
            return Plan(e.spoken, "error")
        except Exception:
            logger.exception("tool planning failed for %r", text)
        return None

    # ── plumbing ───────────────────────────────────────────────────────
    @staticmethod
    def _normalize(text: str) -> str:
        t = text.lower().strip().strip(".,!?;: ")
        t = re.sub(r"\s+", " ", t)
        prev = None
        while prev != t:
            prev, t = t, LEAD.sub("", t)
        return t

    def _clear_pending(self) -> None:
        with self._lock:
            self.pending = None

    def _confirm(self, tool: str, prompt: str, action: Callable[[], None]) -> Plan:
        """Speak the question now; the question itself is registered when the turn commits."""
        def register() -> None:
            with self._lock:
                self.pending = Pending(tool, prompt, action)

        return Plan(prompt + " Say yes to go ahead.", tool, register)

    def _deferred(self, tool: str, agent: str, summary: str, action: Callable[[], str | None]) -> Callable[[], None]:
        """Wrap a side effect: audit it, and keep a failure to report on the next turn."""
        def run() -> None:
            try:
                action()
                self._audit(tool, agent, summary, "ok")
            except ToolError as e:
                self._audit(tool, agent, summary, f"failed: {e.spoken}")
                self.announce(f"The last thing I tried didn't work. {e.spoken}")
            except Exception as e:
                logger.exception("tool %s failed", tool)
                self._audit(tool, agent, summary, f"failed: {type(e).__name__}")
                self.announce("The last thing I tried failed unexpectedly.")

        return run

    def _audit(self, tool: str, agent: str, summary: str, outcome: str, **extra: str) -> None:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            with (self.state_dir / "tool_audit.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": datetime.now().astimezone().isoformat(timespec="seconds"), "tool": tool,
                                    "agent": agent, "what": summary[:160], "outcome": outcome, **extra}) + "\n")
        except OSError:
            logger.warning("tool audit not written")

    def _registry(self) -> dict[str, dict]:
        data = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        return {a["id"]: a for a in data.get("agents", []) if a.get("id")}

    def _agent(self, agent_id: str) -> dict:
        try:
            agent = self._registry().get(agent_id)
        except (OSError, yaml.YAMLError):
            agent = None
        if not agent or (agent.get("api") or {}).get("enabled") is False:
            raise ToolError(f"I don't have {agent_id} available right now.")
        return agent

    def _gw(self, method: str, path: str, *, json_body: Any = None, timeout: float = 10.0,
            headers: dict[str, str] | None = None, agent: str = "the agent") -> Any:
        try:
            r = self.http.request(method, self.gateway_url + path, json=json_body, timeout=timeout,
                                  headers={"Authorization": f"Bearer {self.token}", **(headers or {})})
        except httpx.TimeoutException:
            raise ToolError(f"{agent} is taking too long. I'll leave it for now.")
        except httpx.HTTPError:
            raise ToolError("I can't reach the agent gateway right now.")
        if r.status_code == 503:
            detail = _detail(r)
            if "halted" in detail:
                raise ToolError("The agents are halted. Run fortress-resume to bring them back.")
            wait = r.headers.get("Retry-After", "a minute")
            raise ToolError(f"{agent} keeps failing, so I'm leaving it alone for about {wait} seconds.")
        if r.status_code == 429:
            raise ToolError(f"{agent} is busy right now. Try again shortly.")
        if r.status_code == 422:
            raise ToolError(f"That isn't something {agent} is allowed to do.")
        if r.status_code >= 400:
            raise ToolError(f"{agent} couldn't do that.")
        return r.json()

    def _agent_for(self, alias: str) -> str:
        for aid, names in self.aliases.items():
            if alias in names:
                return aid
        raise ToolError("I don't know that agent.")

    def _resolve(self, raw: str) -> tuple[Optional[str], str]:
        """(agent, how sure). 'exact': a known spelling. 'close': sounds like one name only. 'maybe': plausible, ask first."""
        word = raw.strip().lower()
        tight = re.sub(r"[\s.\-]", "", word)
        for aid, names in self.aliases.items():
            if word in names or tight in {re.sub(r"[\s.\-]", "", n) for n in names}:
                return aid, "exact"
        sk = skeleton(word)
        hits = {aid for aid, names in self.aliases.items() if sk and any(skeleton(n) == sk for n in names)}
        if len(hits) == 1:
            return next(iter(hits)), "close" if len(sk) >= 4 else "maybe"
        flat = {a: aid for aid, ns in self.aliases.items() for a in ns if " " not in a}
        near = difflib.get_close_matches(tight, list(flat), n=1, cutoff=0.8)
        if near:
            return flat[near[0]], "close" if len(tight) >= 5 else "maybe"
        return None, ""

    def _split_name(self, text: str) -> tuple[Optional[str], str, str]:
        """Read an agent name off the front of text: (agent, how sure, the rest). Tries two words, then one."""
        words = re.sub(r"^(?:the |that |my )", "", text).split()
        # Known spellings win, longest first; only then sound-alikes, one word before two, so
        # "dummy how the server is" is DUM-E plus a question, not one odd two-word name.
        for sizes, want_exact in (((3, 2, 1), True), ((1, 2), False)):
            for n in sizes:
                if len(words) >= n:
                    agent, how = self._resolve(" ".join(words[:n]))
                    if agent and (how == "exact" or not want_exact):
                        return agent, how, " ".join(words[n:])
        return None, "", text

    @staticmethod
    def _label(agent_id: str, agent: dict | None = None) -> str:
        return {"dume": "DUM-E", "oda": "ODA"}.get(agent_id, agent_id.capitalize())

    def _p_mention(self, t: str) -> Optional[Plan]:
        """An agent named at the very start, however the opening was garbled: "Kenya's dummy, what models are loaded?"."""
        words = [w.strip(",.;:!?") for w in t.split()]
        if not words or words[0] in ("what", "which", "who", "how", "why", "did", "does", "was", "were", "is", "are"):
            return None
        for i in range(0, min(3, len(words))):
            for n in (2, 1):
                agent, how = self._resolve(" ".join(words[i:i + n]))
                if agent and how == "exact":
                    body = " ".join(w for w in t.split()[i + n:]).strip()
                    if body and QUESTION_SHAPE.search(body):
                        return self._ask(agent, body)
                    return None
        return None

    # ── natural routing: no need to say "ask" ──────────────────────────
    def _machine_question(self, t: str) -> bool:
        """A question about the machine, however it is phrased. Words that only mean the machine in context
        (system, server, load...) need a state word beside them; unmistakable ones (GPU, RAM, disk...) do not."""
        if CHANGE_WORDS.search(t) or ADVICE.search(t) or len(t.split()) > 32 or not ASKY.search(t):
            return False
        if STRONG_MACHINE.search(t):
            return True
        return bool(WEAK_MACHINE.search(t) and STATE_WORDS.search(t) and not OPINION.search(t))

    def _p_auto(self, t: str) -> Optional[Plan]:
        """Route by what is being asked. Only clear questions about the machine, markets, news, design or automation."""
        if self._machine_question(t):
            return self._ask("dume", t)
        personal = re.search(r"\b(?:my|me|i|we|our)\b", t) and not re.search(rf"\b(?:my|our) {OPS}\b", t)
        if re.search(rf"^(?:what(?:'s| is| are)|how(?:'s| is| are| much| many)|tell me|give me|show me|check|list|which|is|are|any)\b.*\b{OPS}\b", t) and (
                re.search(r"\b(?:status|state|health|healthy|doing|going on|holding up|running|ok|okay|fine|up|down|busy|"
                          r"free|full|left|using|eating|hogging|usage|space)\b", t)
                or re.match(rf"^how much .*\b{OPS}\b", t)):
            if not CHANGE_WORDS.search(t):
                return self._ask("dume", t)
        if re.match(r"^(?:is|are) (?:everything|all) .*\b(?:system|machine|server|fortress)\b", t):
            return self._ask("dume", t)
        if re.match(rf"^(?:what|how|why|should|is|are|tell me|give me|any)\b.*\b{FINANCE}\b", t) and not personal:
            return self._ask("dexter", t)
        if (m := re.match(r"^(?:what(?:'s| is) the latest (?:on|about|with|in)|any news (?:on|about|from)|"
                          r"(?:the )?news (?:on|about)|what(?:'s| is) happening (?:with|in|to|on)|what happened (?:with|in|to|at))\s+(.+)$", t)) \
                and not re.search(r"\b(?:my|me|i|we|our|you|your)\b", m.group(1)):
            return self._ask("detective", t)
        if re.match(r"^(?:make|design|draw|create|build|give) (?:me )?(?:a |an )?(?:new )?(?:landing page|poster|logo|web ?site|web page|"
                    r"mock-?up|slide deck|deck|prototype|banner)\b", t):
            return self._ask("oda", t)
        if re.match(r"^automate\b", t):
            return self._ask("flowsmith", t)
        return None

    # ── what Apollo can do ─────────────────────────────────────────────
    def _present(self) -> list[str]:
        try:
            reg = self._registry()
        except (OSError, yaml.YAMLError):
            return []
        return [a for a in VOICE_AGENTS if a in reg and (reg[a].get("api") or {}).get("enabled") is not False]

    def _p_roster(self, t: str) -> Optional[Plan]:
        if not ROSTER.search(t):
            return None
        here = self._present()
        if not here:
            return Plan("I can't see the agent list right now.", "roster")
        names = [f"{self._label(a)} for {BLURBS[a]}" for a in here]
        return Plan("I'm the hub. I can reach " + "; ".join(names) + ". I can also send you notifications and manage your notes. "
                    "Say ask the detective about something, or what are the agents doing.", "roster")

    def capabilities_block(self) -> str:
        """What the language model is told about the tools, so it never has to guess or deny them."""
        lines: list[str] = []
        if self.reminders:
            lines += [
                "## Timed reminders (Telemon; phone + voice at due time)",
                "Use these lines ONLY when the owner's own words say remind, reminder or alarm. Questions about Plane, "
                "tickets, projects, notes or agents are never reminder requests.",
                "List and read happen immediately; create, update, delete, enable and disable ask the owner to say yes.",
                "Reply with ONLY one line: [[remind:list|]], [[remind:get|5]], [[remind:in|10m|drink water]], "
                "[[remind:at|2026-11-01 18:30|call mom]], [[remind:daily|09:00|standup]], "
                "[[remind:weekly|Mon,Wed|18:30|gym]], [[remind:update|5|in:20m]], [[remind:update|5|text:new words]], "
                "[[remind:disable|5]], [[remind:delete|5]].",
            ]
        if self.plane:
            lines += [
                "## Plane ticketing (live CRUD on the self-hosted board)",
                f"You have direct access to {self.plane.project_label} on Plane. List and read happen immediately; "
                "create, update, and delete ask the owner to say yes after you read the question back.",
                "Reply with ONLY one line (no prose): [[plane:list|]], [[plane:get|LP-5]], "
                "[[plane:create|title|optional description]], [[plane:update|LP-5|state:Done]], "
                "[[plane:delete|LP-5]]. Never say you cannot reach Plane or ticketing when this section is present.",
            ]
        here = self._present()
        if not here and not lines:
            return ""
        if here:
            lines += ["## Your agents and tools (real; this is what you can actually do)",
                      "You are the hub of specialist agents. The owner reaches them through you, by voice:"]
            for a in here:
                hears = ", ".join([n for n in self.aliases[a] if n not in (a, self._label(a).lower())][:6])
                lines.append(f"- {self._label(a)} ({BLURBS[a]}): \"ask {self._label(a) if a != 'dume' else 'dume'} about ...\""
                             + (f" (heard as: {hears})" if a == "dume" else ""))
            lines.append("The speech recogniser often mangles these names (for example dummy, diom or dumb e for DUM-E). "
                         "Treat any close sound-alike as the agent; if you cannot tell, ask which agent they meant.")
        lines += ["Also: \"notify me ...\", reminders (\"remind me in 10 minutes to …\", \"list my reminders\"), "
                  "notes (\"list my notes\", \"read note X\", \"create a note called X saying Y\", "
                  "\"delete the note X\"), Plane (\"list plane tickets\", \"read ticket LP-5\", \"create a plane ticket called X\"), "
                  "\"run the detective nightly\", \"what are the agents doing\", \"cancel the detective job\".",
                  "You can use the agents yourself. When the owner wants live information from one, or asks you to ask one, reply "
                  "with ONLY this one line and nothing else: [[ask:AGENT|the question in plain words]] (AGENT is dume, detective, "
                  "dexter or oda). The system runs it and reads the agent's own answer to the owner. Use it for anything about the "
                  "machine (RAM, GPU, disks, containers, logs, which models are loaded, status) -> dume; anything "
                  "that needs looking up on the web (news, facts, films, people, prices, anything recent or that you are not sure "
                  "of) -> detective, which browses and cites sources and takes 15 to 40 seconds; markets -> dexter; design -> oda. "
                  "Do not use an agent for chit-chat, opinions or things you already know for certain. "
                  "Do not guess machine numbers yourself. Never say you cannot reach "
                  "an agent, never tell the owner what to say instead, and never claim you asked unless you used that line. "
                  "Never offer (\"I can ask DUM-E...\", \"I don't have live access\"): if you were about to, use the line instead. "
                  "A question or complaint about what you did earlier (\"did you ask dummy?\") is answered in words, never with a new call. "
                  "Advice, opinions and how-to questions (\"what would you do\", \"how do I...\") are answered by you, "
                  "not sent to an agent: only questions about the machine's current state or live information. "
                  "Examples (your whole reply is just the line): owner: \"is my graphics card okay?\" -> "
                  "[[ask:dume|how is the GPU doing: memory use and temperature]]; owner: \"how much space is left?\" -> "
                  "[[ask:dume|how much disk space is left on each drive]]; owner: \"what's nvidia doing today?\" -> "
                  "[[ask:dexter|how is nvidia stock doing today]]; owner: \"anything new on the port strike?\" -> "
                  "[[ask:detective|the latest on the port strike]]. "
                  "DUM-E only proposes changes, in the owner's terminal; you never change the server.",
                  "A tool answer is finished business once spoken: do not bring up its numbers or topics (disk, GPU, agent "
                  "reports) again unless the owner asks, and never tie them to an unrelated subject."]
        return "\n".join(lines)

    # ── notify ─────────────────────────────────────────────────────────
    def _p_notify(self, t: str) -> Optional[Plan]:
        m = re.match(r"^(?:send me a (?:notification|notif|message|text)|notify me|ping me|text me|message me)"
                     r"(?: (?:that|to say|saying|about|to|:))?\s*(.*)$", t)
        if not m:
            return None
        body = m.group(1).strip()
        if not body:
            return Plan("What should the message say?", "notify")

        def send() -> None:
            try:
                r = self.http.post(self.notifier_url, json={"message": body[:NOTIFY_CHARS], "source": "apollo"}, timeout=8.0)
            except httpx.HTTPError:
                raise ToolError("I couldn't reach the notifier.")
            if r.status_code >= 400:
                raise ToolError("The notifier refused the message.")

        action = self._deferred("notify", "notifier", body, send)
        if "notify" in self.confirm:  # a misheard word would otherwise go straight to the owner's phone
            return self._confirm("notify", f"Send this to your phone: {speakable(body[:NOTIFY_CHARS], 160)}?", action)
        return Plan("Sending that to your phone.", "notify", action)

    # ── notes ──────────────────────────────────────────────────────────
    def _note_path(self, name: str) -> Path:
        slug = slugify(name)
        if not slug:
            raise ToolError("I need a name for the note.")
        path = self.notes_dir / f"{slug}.md"
        self._inside(path)
        return path

    def _inside(self, path: Path) -> None:
        """Every note path is checked against the notes directory; links are refused outright."""
        root = self.notes_dir.resolve()
        parent = path.parent.resolve()
        if parent != root and root not in parent.parents:
            raise ToolError("That is outside my notes folder.")
        if path.is_symlink():
            raise ToolError("I won't touch a linked file.")

    def _all_notes(self) -> list[Path]:
        if not self.notes_dir.is_dir():
            return []
        return sorted((p for p in self.notes_dir.glob("*.md") if p.is_file() and not p.is_symlink()),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    def _find_note(self, query: str) -> Path:
        notes = self._all_notes()
        if not notes:
            raise ToolError("You don't have any notes yet.")
        if re.fullmatch(r"(?:the )?(?:last|latest|newest|most recent)(?: one| note)?", query):
            return notes[0]
        slug = slugify(query)
        exact = [p for p in notes if p.stem == slug]
        if exact:
            return exact[0]
        near = [p for p in notes if slug and (slug in p.stem or p.stem in slug)]
        if len(near) == 1:
            return near[0]
        if not near:
            raise ToolError(f"I couldn't find a note called {query}.")
        raise ToolError("Which one? " + ", ".join(p.stem.replace("-", " ") for p in near[:4]) + ".")

    def _p_notes(self, t: str) -> Optional[Plan]:
        if re.match(r"^(?:list|show|what are)(?: me)? (?:my |the )?notes\b|^what notes do i have|^do i have any notes", t):
            notes = self._all_notes()
            if not notes:
                return Plan("You don't have any notes yet.", "notes")
            names = ", ".join(p.stem.replace("-", " ") for p in notes[:5])
            if len(notes) == 1:
                return Plan(f"You have one note, {names}.", "notes")
            return Plan(f"You have {len(notes)} notes. The newest are {names}.", "notes")
        m = re.match(r"^(?:read|open|show|what(?:'s| is) in)(?: me)? (?:my |the )?(?:note(?: called| named| about| on)?|(last|latest|newest) note)\s*(.*)$", t)
        if m:
            path = self._find_note(m.group(1) or m.group(2))
            body = re.sub(r"^---.*?---\s*", "", path.read_text(encoding="utf-8", errors="replace"), flags=re.S)
            return Plan(f"{path.stem.replace('-', ' ')}. {speakable(body) or 'It is empty.'}", "notes")
        m = re.match(r"^(?:create|start|make|new) (?:a )?note (?:called|named) (.+?)(?: (?:saying|with|that says|says|:) (.+))?$", t)
        if m:
            path, body = self._note_path(m.group(1)), (m.group(2) or "").strip()
            if DAILY_NOTE.match(path.name):
                raise ToolError("That name is reserved for the daily notes.")
            title = m.group(1).strip()

            def create() -> None:
                self.notes_dir.mkdir(parents=True, exist_ok=True)
                text = f"\n{body}\n" if path.exists() else f"# {title}\n\n{body}\n"
                with path.open("a", encoding="utf-8") as f:
                    f.write(text)

            action = self._deferred("notes", "notes", f"create {path.name}", create)
            if path.exists() and "notes_write" in self.confirm:
                return self._confirm("notes", f"Add this to the existing note {title}: {speakable(body, 160) or 'nothing'}?", action)
            verb = "Adding to the existing note" if path.exists() else "Creating the note"
            return Plan(f"{verb} {title}.", "notes", action)
        m = re.match(r"^(?:add|append) (?:this )?to (?:my |the )?note (?:called |named )?(.+?)(?: (?:saying|that says|that|:) (.+))$", t)
        if m:
            path = self._find_note(m.group(1))
            if DAILY_NOTE.match(path.name):
                raise ToolError("The daily notes are filled in for you; I won't edit them.")
            self._inside(path)
            body = m.group(2).strip()

            def append() -> None:
                with path.open("a", encoding="utf-8") as f:
                    f.write(f"\n{body}\n")

            action = self._deferred("notes", "notes", f"append {path.name}", append)
            if "notes_write" in self.confirm:
                return self._confirm("notes", f"Add this to {path.stem.replace('-', ' ')}: {speakable(body, 160)}?", action)
            return Plan(f"Adding that to {path.stem.replace('-', ' ')}.", "notes", action)
        m = re.match(r"^(?:delete|remove|trash) (?:my |the )?note (?:called |named |about )?(.+)$", t)
        if m:
            path = self._find_note(m.group(1))
            if DAILY_NOTE.match(path.name):
                raise ToolError("The daily notes belong to Virtual Self; I won't delete them.")
            self._inside(path)

            def trash() -> None:
                dest = self.notes_dir / ".trash"
                dest.mkdir(parents=True, exist_ok=True)
                path.rename(dest / f"{path.stem}-{datetime.now():%Y%m%d-%H%M%S}.md")

            action = self._deferred("notes", "notes", f"delete {path.name}", trash)
            return self._confirm("notes", f"Delete the note {path.stem.replace('-', ' ')}? It goes to the trash folder.", action)
        return None

    # ── Plane (self-hosted ticketing) ────────────────────────────────
    def _plane_missing(self) -> Plan:
        return Plan(
            "Plane isn't wired yet. Export PLANE_API_KEY or set the api_key_file in config.yaml, "
            "then restart Apollo.",
            "plane",
        )

    def _plane_run(self, fn: Callable[[], str]) -> Plan:
        try:
            return Plan(fn(), "plane")
        except PlaneError as e:
            return Plan(e.spoken, "plane")
        except Exception:
            logger.exception("plane tool failed")
            return Plan("Something went wrong talking to Plane.", "plane")

    def plane_command(self, action: str, args: str) -> Optional[Plan]:
        if not self.plane:
            return self._plane_missing()
        action = {"read": "get", "move": "update"}.get(action, action)
        args = args.strip()
        if action == "list":
            return self._plane_run(lambda: self.plane.speak_list())
        if action in ("get", "comment"):
            if not args:
                return Plan("Which ticket? Say LP-5 or the UUID.", "plane")
            return self._plane_run(lambda a=args: self.plane.describe_item(self.plane.get_work_item(a)))
        if action == "create":
            name, _, desc = args.partition("|")
            name = (name.strip() or args).strip()
            desc = desc.strip()
            if not name:
                return Plan("What should the ticket be called?", "plane")

            def apply_create() -> None:
                try:
                    item = self.plane.create_work_item(name, desc)
                    self._audit("plane", "plane", f"create {name}", "ok")
                    self.outbox.put(f"Created {self.plane.ticket_label(item)} on {self.plane.project_label}.")
                except PlaneError as e:
                    self._audit("plane", "plane", f"create {name}", f"failed: {e.spoken}")
                    self.outbox.put(e.spoken)

            return self._confirm(
                "plane",
                f"Create Plane ticket {name}" + (f" — {speakable(desc, 120)}" if desc else "") + "?",
                apply_create,
            )
        if action == "update":
            ref, _, rest = args.partition("|")
            ref, rest = ref.strip(), rest.strip()
            if not ref:
                return Plan("Which ticket should I update?", "plane")
            if rest.lower().startswith("state:") or rest.lower().startswith("state="):
                state = re.split(r"[:=]", rest, 1)[1].strip()

                def apply_state() -> None:
                    try:
                        sid = self.plane.state_id(state)
                        item = self.plane.update_work_item(ref, state=sid)
                        self._audit("plane", "plane", f"update {ref}", "ok")
                        self.outbox.put(f"Moved {self.plane.ticket_label(item)} to {state}.")
                    except PlaneError as e:
                        self.outbox.put(e.spoken)

                return self._confirm("plane", f"Move {ref} to {state} on Plane?", apply_state)
            if rest.lower().startswith("name:") or rest.lower().startswith("title="):
                title = re.split(r"[:=]", rest, 1)[1].strip()

                def apply_name() -> None:
                    try:
                        item = self.plane.update_work_item(ref, name=title)
                        self._audit("plane", "plane", f"rename {ref}", "ok")
                        self.outbox.put(f"Renamed to {self.plane.ticket_label(item)}.")
                    except PlaneError as e:
                        self.outbox.put(e.spoken)

                return self._confirm("plane", f"Rename {ref} to {title}?", apply_name)
            return Plan("Update needs state or name, like [[plane:update|LP-5|state:Done]].", "plane")
        if action == "delete":
            if not args:
                return Plan("Which ticket should I delete?", "plane")
            ref = args.split("|", 1)[0].strip()
            item = self.plane.get_work_item(ref)
            label = self.plane.ticket_label(item)

            def apply_delete() -> None:
                try:
                    self.plane.delete_work_item(ref)
                    self._audit("plane", "plane", f"delete {ref}", "ok")
                    self.outbox.put(f"Deleted {label} from Plane permanently.")
                except PlaneError as e:
                    self.outbox.put(e.spoken)

            return self._confirm("plane", f"Delete {label} from Plane permanently?", apply_delete)
        return Plan(f"I don't know the Plane action {action!r}.", "plane")

    def _p_plane(self, t: str) -> Optional[Plan]:
        if not PLANE_WORDS.search(t) and not re.search(r"\bLP-\d+\b", t, re.I):
            return None
        if not self.plane:
            if re.match(r"^(?:list|show|create|delete|update|read|move|open|get|what)", t):
                return self._plane_missing()
            return None
        if re.match(r"^(?:list|show|what(?:'s| are)|open|see)(?: me)? (?:the )?(?:plane )?"
                    r"(?:tickets?|issues|work items|board)(?:\?)?$", t) or re.match(
                        r"^(?:plane )?(?:tickets|issues|work items)(?:\?)?$", t):
            return self._plane_run(lambda: self.plane.speak_list())
        if (PLANE_LIST_ASK.search(t) and PLANE_NOUNS.search(t) and not PLANE_WRITE_VERB.search(t)
                and not re.search(r"\bLP-\d+\b", t, re.I)):
            return self._plane_run(lambda: self.plane.speak_list())  # "what are the plane projects that are open"
        if m := re.match(r"^(?:read|show|open|get|describe) (?:the )?(?:plane )?(?:ticket )?(LP-\d+|[0-9a-f-]{36})$", t, re.I):
            ref = m.group(1)
            return self._plane_run(lambda r=ref: self.plane.describe_item(self.plane.get_work_item(r)))
        if m := re.match(
                r"^(?:create|add|open) (?:a )?(?:plane )?(?:ticket|issue|work item) "
                r"(?:called|named|titled)? (.+?)(?: (?:saying|with(?: description| body| text| details)?|:) (.+))?$", t):
            name, desc = m.group(1).strip(), (m.group(2) or "").strip()

            def apply_create() -> None:
                try:
                    item = self.plane.create_work_item(name, desc)
                    self._audit("plane", "plane", f"create {name}", "ok")
                    self.outbox.put(f"Created {self.plane.ticket_label(item)}.")
                except PlaneError as e:
                    self.outbox.put(e.spoken)

            return self._confirm(
                "plane",
                f"Create Plane ticket {name}" + (f" — {speakable(desc, 120)}" if desc else "") + "?",
                apply_create,
            )
        if m := re.match(r"^(?:move|set|mark) (?:plane )?(?:ticket )?(LP-\d+) (?:to|as) (?:state )?(.+)$", t, re.I):
            ref, state = m.group(1), m.group(2).strip()

            def apply_move() -> None:
                sid = self.plane.state_id(state)
                item = self.plane.update_work_item(ref, state=sid)
                self._audit("plane", "plane", f"update {ref}", "ok")
                self.outbox.put(f"Moved {self.plane.ticket_label(item)} to {state}.")

            return self._confirm("plane", f"Move {ref} to {state} on Plane?", apply_move)
        if m := re.match(r"^update (?:plane )?(?:ticket )?(LP-\d+) (?:title|name) (?:to )?(.+)$", t, re.I):
            ref, title = m.group(1), m.group(2).strip()

            def apply_title() -> None:
                item = self.plane.update_work_item(ref, name=title)
                self._audit("plane", "plane", f"rename {ref}", "ok")
                self.outbox.put(f"Renamed to {self.plane.ticket_label(item)}.")

            return self._confirm("plane", f"Rename {ref} to {title}?", apply_title)
        if m := re.match(r"^delete (?:plane )?(?:ticket )?(LP-\d+)$", t, re.I):
            ref = m.group(1)
            item = self.plane.get_work_item(ref)
            label = self.plane.ticket_label(item)

            def apply_delete() -> None:
                self.plane.delete_work_item(ref)
                self._audit("plane", "plane", f"delete {ref}", "ok")
                self.outbox.put(f"Deleted {label} from Plane permanently.")

            return self._confirm("plane", f"Delete {label} from Plane permanently?", apply_delete)
        return None

    # ── timed reminders (Telemon) ─────────────────────────────────────
    def _remind_missing(self) -> Plan:
        return Plan(
            "Reminders aren't reachable. Make sure the notifier is running on port 8001, then restart Apollo.",
            "remind",
        )

    def _remind_run(self, fn: Callable[[], str]) -> Plan:
        try:
            return Plan(fn(), "remind")
        except ReminderError as e:
            return Plan(e.spoken, "remind")
        except Exception:
            logger.exception("reminder tool failed")
            return Plan("Something went wrong with reminders.", "remind")

    @staticmethod
    def mentions_reminders(user_message: str) -> bool:
        return bool(REMIND_WORDS.search(user_message.lower()))

    def poll_voice_reminders(self) -> Optional[str]:
        """One due reminder line for the voice loop (Telegram is handled by the notifier scheduler)."""
        if not self.reminders:
            return None
        now = time.monotonic()
        if now - self._last_voice_poll < VOICE_POLL_S:  # the voice loop ticks many times a second
            return None
        self._last_voice_poll = now
        msgs = self.reminders.dequeue_voice(limit=1)
        if not msgs:
            return None
        text = msgs[0].strip()
        return f"Reminder. {text}" if text else None

    def reminder_command(self, action: str, args: str) -> Optional[Plan]:
        if not self.reminders:
            return self._remind_missing()
        action = {"cancel": "delete", "off": "disable", "on": "enable"}.get(action, action)
        args = args.strip()
        if action == "list":
            return self._remind_run(lambda: self.reminders.speak_list())
        if action == "get":
            if not args or not args.split("|", 1)[0].strip().isdigit():
                return Plan("Which reminder? Say the number from the list.", "remind")
            rid = int(args.split("|", 1)[0].strip())
            return self._remind_run(lambda i=rid: self.reminders.speak_one(self.reminders.get(i)))
        if action == "in":
            duration, _, text = args.partition("|")
            duration, text = duration.strip(), text.strip()
            if not duration or not text:
                return Plan("Say remind me in 10 minutes to do something.", "remind")

            def apply_in() -> None:
                r = self.reminders.add_in(duration, text)
                self._audit("remind", "reminders", f"in {duration}", "ok")
                self.outbox.put(f"Set reminder {r.id} for {duration} from now: {text}.")

            return self._confirm("remind", f"Remind you in {duration}: {speakable(text, 120)}?", apply_in)
        if action == "at":
            when, _, text = args.partition("|")
            when, text = when.strip(), text.strip()
            if not when or not text:
                return Plan("Say remind me at six thirty to … or give a date and time.", "remind")

            def apply_at() -> None:
                r = self.reminders.add_at(when, text)
                self._audit("remind", "reminders", f"at {when}", "ok")
                self.outbox.put(f"Set reminder {r.id} for {when}: {text}.")

            return self._confirm("remind", f"Remind you at {when}: {speakable(text, 120)}?", apply_at)
        if action == "daily":
            hhmm, _, text = args.partition("|")
            hhmm, text = hhmm.strip(), text.strip()
            if not hhmm or not text:
                return Plan("Daily reminders need a time like 09:00 and what to remember.", "remind")

            def apply_daily() -> None:
                r = self.reminders.add_daily(hhmm, text)
                self._audit("remind", "reminders", f"daily {hhmm}", "ok")
                self.outbox.put(f"Daily reminder {r.id} at {hhmm}: {text}.")

            return self._confirm("remind", f"Every day at {hhmm}, remind you: {speakable(text, 120)}?", apply_daily)
        if action == "weekly":
            chunk = [p.strip() for p in args.split("|")]
            if len(chunk) < 3:
                return Plan("Weekly needs days, time, and text, like Mon,Wed at 18:30.", "remind")
            days, hhmm, text = chunk[0], chunk[1], "|".join(chunk[2:]).strip()

            def apply_weekly() -> None:
                r = self.reminders.add_weekly(days, hhmm, text)
                self._audit("remind", "reminders", f"weekly {days} {hhmm}", "ok")
                self.outbox.put(f"Weekly reminder {r.id} on {days} at {hhmm}: {text}.")

            return self._confirm("remind", f"Every {days} at {hhmm}, remind you: {speakable(text, 120)}?", apply_weekly)
        if action in ("enable", "disable"):
            rid_s = args.split("|", 1)[0].strip()
            if not rid_s.isdigit():
                return Plan("Which reminder number?", "remind")
            rid = int(rid_s)
            on = action == "enable"

            def apply_toggle() -> None:
                r = self.reminders.patch(rid, enabled=on)
                self._audit("remind", "reminders", f"{action} {rid}", "ok")
                self.outbox.put(f"Reminder {r.id} is {'on' if on else 'off'}.")

            verb = "Turn on" if on else "Turn off"
            return self._confirm("remind", f"{verb} reminder {rid}?", apply_toggle)
        if action == "update":
            rid_s, _, rest = args.partition("|")
            if not rid_s.strip().isdigit():
                return Plan("Update needs a reminder number and in, at, or text.", "remind")
            rid = int(rid_s.strip())
            rest = rest.strip()
            in_d = at_l = new_text = None
            if rest.lower().startswith("in:"):
                in_d = rest[3:].strip()
            elif rest.lower().startswith("at:"):
                at_l = rest[3:].strip()
            elif rest.lower().startswith("text:"):
                new_text = rest[5:].strip()
            else:
                return Plan("Use in, at, or text after the id, like [[remind:update|5|in:20m]].", "remind")

            def apply_update() -> None:
                r = self.reminders.patch(rid, text=new_text, in_duration=in_d, at_local=at_l)
                self._audit("remind", "reminders", f"update {rid}", "ok")
                self.outbox.put(self.reminders.speak_one(r))

            detail = in_d or at_l or new_text or ""
            return self._confirm("remind", f"Update reminder {rid} to {speakable(detail, 120)}?", apply_update)
        if action == "delete":
            rid_s = args.split("|", 1)[0].strip()
            if not rid_s.isdigit():
                return Plan("Which reminder should I delete?", "remind")
            rid = int(rid_s)

            def apply_delete() -> None:
                self.reminders.delete(rid)
                self._audit("remind", "reminders", f"delete {rid}", "ok")
                self.outbox.put(f"Deleted reminder {rid}.")

            return self._confirm("remind", f"Delete reminder {rid} permanently?", apply_delete)
        return Plan(f"I don't know the reminder action {action!r}.", "remind")

    def _p_remind(self, t: str) -> Optional[Plan]:
        if not REMIND_WORDS.search(t) and not re.search(r"\bremind me\b", t):
            return None
        if not self.reminders:
            if re.match(r"^(?:list|show|remind|delete|cancel|update)", t):
                return self._remind_missing()
            return None
        if re.match(r"^(?:list|show|what(?:'s| are)|do i have)(?: me)? (?:my )?reminders?\b", t):
            return self._remind_run(lambda: self.reminders.speak_list())
        if m := re.match(r"^(?:read|show|describe|get) (?:reminder )?(?:number )?(\d+)\b", t):
            rid = int(m.group(1))
            return self._remind_run(lambda i=rid: self.reminders.speak_one(self.reminders.get(i)))
        if m := re.match(
                r"^remind me in (\d+\s*(?:minutes?|mins?|hours?|hrs?|seconds?|secs?|days?)|\d+[mhd])\s+(?:to )?(.+)$", t):
            duration, text = m.group(1).replace(" ", ""), m.group(2).strip()
            return self.reminder_command("in", f"{duration}|{text}")
        if m := re.match(r"^remind me (?:at|on) (.+?) to (.+)$", t):
            return self.reminder_command("at", f"{m.group(1).strip()}|{m.group(2).strip()}")
        if m := re.match(r"^(?:every day|daily) at (\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+(?:remind me to )?(.+)$", t, re.I):
            hhmm = m.group(1).strip()
            return self.reminder_command("daily", f"{hhmm}|{m.group(2).strip()}")
        if m := re.match(r"^delete (?:reminder )?(?:number )?(\d+)\b", t):
            return self.reminder_command("delete", m.group(1))
        if m := re.match(r"^cancel (?:reminder )?(?:number )?(\d+)\b", t):
            return self.reminder_command("delete", m.group(1))
        if m := re.match(r"^(?:turn off|disable) (?:reminder )?(?:number )?(\d+)\b", t):
            return self.reminder_command("disable", m.group(1))
        if m := re.match(r"^(?:turn on|enable) (?:reminder )?(?:number )?(\d+)\b", t):
            return self.reminder_command("enable", m.group(1))
        return None

    # ── asking agents ──────────────────────────────────────────────────
    def _p_ask(self, t: str) -> Optional[Plan]:
        if m := re.match(r"^(?:ask|tell|have|get|check with|talk to|message) (.+)$", t):
            agent, how, rest = self._split_name(m.group(1))
            if agent:
                body = re.sub(r"^(?:to|about|if|whether|for|that)\s+", "", rest).strip()
                if how == "maybe":
                    label = self._label(agent)

                    def register() -> None:
                        with self._lock:
                            self.pending = Pending("ask", "", lambda: None, replan=lambda: self._ask(agent, body))

                    return Plan(f"Did you mean {label}? Say yes and I'll ask it.", "ask", register)
                return self._ask(agent, body)
        # by what the request is about, only for plain imperative openings
        if m := re.match(r"^(?:research|investigate|look into|dig into|find out about) (.+)$", t):
            topic = m.group(1)
            agent = "dexter" if re.search(r"\b(stock|stocks|share price|market|markets|earnings|valuation|crypto|bitcoin|nifty|sensex|invest\w*)\b", topic) else "detective"
            return self._ask(agent, topic)
        if m := re.match(r"^(?:design|mock ?up|prototype) (.+)$", t):
            return self._ask("oda", "design " + m.group(1))
        if re.match(r"^(?:check|how(?:'s| is)) (?:on )?(?:the )?(?:server|fortress|gpu|disks?|containers?|services|machine)\b", t) or t in ("server status", "fortress status"):
            return self._dume_status()
        if re.match(r"^(?:build|create|make|set up) (?:me )?(?:a|an) (?:n8n )?(?:workflow|automation)\b", t):
            return self._ask("flowsmith", t)
        return None

    def _ask(self, agent_id: str, message: str, by: str = "rules") -> Plan:
        if not message:
            return Plan(f"What should I ask {self._label(agent_id)}?", "ask")
        agent = self._agent(agent_id)
        label = self._label(agent_id)
        if agent_id == "dume":
            change = bool(CHANGE_WORDS.search(message))

            def work_dume() -> str:
                try:
                    said = self._chat("dume", label, message + VOICE_HINT)
                except ToolError:
                    snap = self._dume_snapshot()  # DUM-E would not answer; the quick snapshot is better than silence
                    return f"{label} did not answer, so here is the quick snapshot. {snap.speak}"
                self._audit("ask", "dume", message, "ok", reply=said[:1500], by=by)
                return f"{label} says: {said}" + (" Nothing changes until you approve it in your terminal." if change else "")

            return Plan("", "ask", ack="Asking DUM-E.", lazy=lambda: self._race("ask", label, work_dume))
        if agent_id == "flowsmith":
            return self._confirm("ask", "Flowsmith is experimental. Ask it for a workflow preview? It won't change anything.",
                                 self._deferred("ask", agent_id, message, lambda: self._chat_to_notice(agent_id, label, message)))
        def work() -> str:
            said = self._chat(agent_id, label, message)
            self._audit("ask", agent_id, message, "ok", reply=said[:1500], by=by)
            return f"{label} says: {said}"

        return Plan("", "ask", ack=f"Asking {label}.", lazy=lambda: self._race("ask", label, work))

    def _race(self, tool: str, label: str, work: Callable[[], str]) -> Plan:
        """Call an agent without holding the conversation hostage. An answer inside inline_wait_s is this
        turn's reply; a slower one ends the turn now and is announced the moment it lands."""
        box: dict[str, str] = {}
        done = threading.Event()
        abandoned = [False]

        def run() -> None:
            try:
                out = work()
            except ToolError as e:
                out = e.spoken
            except Exception:
                logger.exception("tool %s failed", tool)
                out = "That didn't work."
            with self._lock:
                box["out"] = out
                done.set()
                late = abandoned[0]
            if late:
                self.announce(out)

        threading.Thread(target=run, name=f"ask-{label}", daemon=True).start()
        if done.wait(self.inline_wait_s):
            return Plan(box["out"], tool)
        with self._lock:
            if not done.is_set():
                abandoned[0] = True
                return Plan(f"{label} is still working on it. I'll tell you the moment it answers.", tool)
        return Plan(box["out"], tool)

    def _chat(self, agent_id: str, label: str, message: str) -> str:
        body = {"message": message, "options": {"timeout_s": int(CHAT_TIMEOUT_S) - 5}}
        data = self._gw("POST", f"/agents/{agent_id}/chat", json_body=body, timeout=CHAT_TIMEOUT_S, agent=label)
        reply = data.get("reply") if isinstance(data, dict) else None
        if isinstance(reply, str) and reply.strip().startswith("{") and '"refused"' in reply:
            try:
                refusal = json.loads(reply)
                who = refusal.get("hand_off_to")
                return speakable(f"{refusal.get('refused', 'That is outside its job.')}" + (f" Try {who}." if who else ""))
            except ValueError:
                pass
        text = speakable(str(reply or ""), AGENT_SPEAK_CHARS)
        if not text:
            raise ToolError(f"{label} had nothing to say.")
        return text

    def _chat_to_notice(self, agent_id: str, label: str, message: str) -> None:
        answer = self._chat(agent_id, label, message)
        self.announce(f"{label} replied: {answer}")

    def _dume_status(self) -> Plan:
        """Status comes in DUM-E's own words, like any other question."""
        return self._ask("dume", "How is the machine doing right now? Give me the overall status.")

    def _dume_snapshot(self) -> Plan:
        self._agent("dume")
        job = self._gw("POST", "/agents/dume/jobs", json_body={"payload": {"action": "snapshot"}}, agent="DUM-E")
        row = self._wait_job(job["job_id"], "DUM-E")
        out = (row.get("result") or {}).get("stdout", "")
        gist = summarize_snapshot(out)
        self._audit("status", "dume", "snapshot", "ok")
        if gist:
            return Plan(gist, "status")
        return Plan("DUM-E reports: " + (speakable(out, 300) or "it answered, but there was nothing to read out."), "status")

    def _wait_job(self, job_id: str, label: str) -> dict:
        end = time.monotonic() + READ_JOB_WAIT_S
        while time.monotonic() < end:
            row = self._gw("GET", f"/jobs/{job_id}", agent=label)
            if row["status"] == "completed":
                return row
            if row["status"] in ("failed", "cancelled"):
                raise ToolError(f"{label} couldn't finish that.")
            time.sleep(0.5)
        raise ToolError(f"{label} is taking too long.")

    # ── jobs ───────────────────────────────────────────────────────────
    def _jobs_file(self) -> Path:
        return self.state_dir / "pending_jobs.json"

    def _load_jobs(self) -> list[dict]:
        try:
            return json.loads(self._jobs_file().read_text())
        except (OSError, ValueError):
            return []

    def _save_jobs(self, jobs: list[dict]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._jobs_file().with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs[-30:]))
        tmp.replace(self._jobs_file())

    def _p_job(self, t: str) -> Optional[Plan]:
        m = re.match(r"^(?:run|start|kick off|trigger) (.+)$", t)
        if not m:
            return None
        agent_id, how, rest = self._split_name(m.group(1))
        if not agent_id or how == "maybe":
            return None  # "start the car" is not ours; a doubtful name is never enough to start a job
        tail = re.match(r"^(?:(?:job|run|pipeline|sleuth) ?)?(nightly|run|resume|status|previews|snapshot)?$", rest.strip())
        if not tail:
            return None

        agent = self._agent(agent_id)
        label = self._label(agent_id)
        default = (agent.get("api") or {}).get("default_job_type") or "run"
        job_type = tail.group(1) or default
        if job_type not in VOICE_JOB_TYPES.get(agent_id, ()):
            raise ToolError(f"I can't start that one by voice. Do it from your terminal.")
        read_only = job_type in ((agent.get("api") or {}).get("read_only_job_types") or [])
        if agent_id == "dume":
            return self._dume_status()
        if read_only:
            job = self._gw("POST", f"/agents/{agent_id}/jobs", json_body={"payload": {"type": job_type}}, agent=label)
            row = self._wait_job(job["job_id"], label)
            self._audit("job", agent_id, job_type, "ok")
            result = row.get("result") or {}
            gist = speakable(str(result.get("stdout") or result.get("reply") or json.dumps(result)), 300)
            return Plan(f"{label} {job_type.replace('_', ' ')}: {gist}", "job")
        key = uuid.uuid4().hex  # one confirmed action = one job, even if the request is repeated

        def start() -> None:
            job = self._gw("POST", f"/agents/{agent_id}/jobs", json_body={"payload": {"type": job_type}},
                           headers={"Idempotency-Key": key}, agent=label)
            record = {"job_id": job["job_id"], "agent": agent_id, "type": job_type,
                      "started": datetime.now().astimezone().isoformat(timespec="seconds")}
            jobs = self._load_jobs()
            jobs.append(record)
            self._save_jobs(jobs)
            self._watch_job(record)

        action = self._deferred("job", agent_id, f"start {job_type}", start)
        return self._confirm("job", f"Start the {label} {job_type} job? It can run for a long time and I'll tell you when it's done.", action)

    def _watch_job(self, record: dict) -> None:
        threading.Thread(target=self._watch_job_run, args=(record,), daemon=True, name=f"job-{record['job_id'][:8]}").start()

    def _watch_job_run(self, record: dict) -> None:
        label = self._label(record["agent"])
        deadline = time.monotonic() + JOB_WATCH_MAX_S
        while time.monotonic() < deadline:
            try:
                row = self._gw("GET", f"/jobs/{record['job_id']}", agent=label)
            except ToolError:
                row = {}
            if row.get("status") in ("completed", "failed", "cancelled"):
                self._announce_job(record, row)
                return
            time.sleep(self.job_poll_s)

    def _announce_job(self, record: dict, row: dict) -> None:
        with self._lock:
            jobs = self._load_jobs()
            mine = next((j for j in jobs if j["job_id"] == record["job_id"]), None)
            if mine is not None:
                if mine.get("announced"):
                    return
                mine["announced"] = True
                self._save_jobs(jobs)
        label, kind = self._label(record["agent"]), record["type"].replace("_", " ")
        status = row.get("status")
        if status == "completed":
            result = row.get("result") or {}
            gist = speakable(str(result.get("stdout") or result.get("reply") or ""), AGENT_SPEAK_CHARS)
            self.announce(f"The {label} {kind} job finished." + (f" {gist}" if gist else ""))
        elif status == "failed":
            self.announce(f"The {label} {kind} job failed.")

    def resume_jobs(self) -> None:
        """Jobs started in an earlier session may have finished while nobody was listening: say so."""
        now = datetime.now().astimezone()
        for j in self._load_jobs():
            if j.get("announced"):
                continue
            try:
                age = (now - datetime.fromisoformat(j["started"])).total_seconds()
            except (KeyError, ValueError):
                continue
            if age <= JOB_RESUME_MAX_AGE_S:
                self._watch_job(j)

    def _p_status(self, t: str) -> Optional[Plan]:
        if not re.match(r"^(?:agent status|status of (?:the )?(?:agents|jobs)|how are the agents|what are the agents doing|"
                        r"(?:are )?(?:any|there any) (?:jobs|agents) (?:running|going)|job status|what(?:'s| is) running|"
                        r"how(?:'s| is) the \w+ job)\b", t):
            return None
        parts: list[str] = []
        for aid in VOICE_AGENTS:
            try:
                h = self._gw("GET", f"/agents/{aid}/health", agent=self._label(aid))
            except ToolError as e:
                return Plan(e.spoken, "status")
            if h.get("breaker") != "closed":
                parts.append(f"{self._label(aid)} is failing and paused")
            elif h.get("running") or h.get("queued"):
                parts.append(f"{self._label(aid)} has {h['running']} running and {h['queued']} waiting")
        recent = []
        for j in self._load_jobs()[-5:]:
            try:
                row = self._gw("GET", f"/jobs/{j['job_id']}")
            except ToolError:
                continue
            recent.append(f"the {self._label(j['agent'])} {j['type']} job is {row['status']}")
        said = "; ".join(parts + recent)
        self._audit("status", "all", "status", "ok")
        return Plan((said[0].upper() + said[1:] + ".") if said else "All the agents are idle and healthy.", "status")

    def _p_cancel(self, t: str) -> Optional[Plan]:
        m = re.match(r"^(?:cancel|stop|kill) (?:the |that |my )?(.*?) ?job$", t)
        if not m:
            return None
        want = None
        if m.group(1):
            want, how, rest = self._split_name(m.group(1))
            if not want or how == "maybe" or rest:
                return None
        jobs = [j for j in self._load_jobs() if not want or j["agent"] == want]
        for j in reversed(jobs):
            row = self._gw("GET", f"/jobs/{j['job_id']}")
            if row["status"] in ("queued", "running"):
                label = self._label(j["agent"])

                def cancel(j=j) -> None:
                    self._gw("POST", f"/jobs/{j['job_id']}/cancel", agent=label)

                return self._confirm("job", f"Cancel the {label} {j['type']} job?", self._deferred("job", j["agent"], "cancel", cancel))
        return Plan("There's no job running that I started.", "job")


def _detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail", ""))
    except ValueError:
        return ""
