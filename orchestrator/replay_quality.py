#!/usr/bin/env python3
"""Replay the conversations that went wrong, through the real brain, and grade the replies.

    PYTHONPATH=. .venv/bin/python replay_quality.py            # all cases
    PYTHONPATH=. .venv/bin/python replay_quality.py name write # cases whose id starts with these

Live and read-only: it asks vself to prepare each turn and streams the reply, but never
commits, so nothing is remembered or written. Checks are deterministic string rules, not
a model's opinion: therapy-speak, filler closers, refusals of harmless play, walls of text,
markdown, and "did it actually say something".
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

MAX_WORDS = 70

THERAPY = re.compile(
    r"how (does|did) that (make|feel)|it sounds like you|i hear you|that must (be|feel)|"
    r"your feelings|valid(ate|ating)? (your|that)|i'?m here for you|take a deep breath",
    re.I,
)
FILLER_CLOSER = re.compile(
    r"(is there anything else|let me know if|feel free to|anything else i can|"
    r"don'?t hesitate|hope that helps|happy to help)[^.?!]*[.?!]?\s*$",
    re.I,
)
REFUSAL = re.compile(
    r"\b(i can'?t|i cannot|i'?m not able to|i won'?t|i'?m unable to|i must decline)\b.{0,60}"
    r"\b(help|say|repeat|assist|do that|comply|threat|harm)",
    re.I,
)
SELF_AS_AI = re.compile(r"as an ai\b|language model|i'?m (just )?an? (ai|assistant)\b", re.I)
MARKDOWN = re.compile(r"(^|\n)\s*([-*•]|\d+\.)\s|\*\*|^#|`", re.M)


def words(text: str) -> int:
    return len(text.split())


def general_problems(reply: str) -> list[str]:
    """Rules every spoken reply must meet."""
    out = []
    if not reply.strip():
        return ["empty reply"]
    if THERAPY.search(reply):
        out.append("therapy-speak")
    if FILLER_CLOSER.search(reply.strip()):
        out.append("filler closer")
    if SELF_AS_AI.search(reply):
        out.append("talks about itself as an AI model")
    if MARKDOWN.search(reply):
        out.append("markdown in a spoken reply")
    if words(reply) > MAX_WORDS:
        out.append(f"{words(reply)} words (limit {MAX_WORDS})")
    return out


@dataclass
class Case:
    id: str
    why: str                      # the real failure this guards against
    user: str
    history: list[dict] = field(default_factory=list)
    check: Optional[Callable[[str], Optional[str]]] = None   # None = pass, else the problem
    direct: Optional[str] = None  # expect vself to answer itself; this must be in the reply (regex)
    context: Optional[str] = None  # expect vself to hand the model real data: this must be in the prompt (regex)
    route: Optional[str] = None    # the turn must be answered by this route (regex), e.g. an agent: "tool:ask"
    no_route: Optional[str] = None  # ...and must NOT take this one (conversation hijacked by a tool)


def _has(pattern: str, what: str) -> Callable[[str], Optional[str]]:
    rx = re.compile(pattern, re.I)
    return lambda reply: None if rx.search(reply) else f"missing {what}"


def _substantive(min_words: int) -> Callable[[str], Optional[str]]:
    return lambda reply: None if words(reply) >= min_words else f"too thin ({words(reply)} words, need {min_words})"


def _no_refusal(reply: str) -> Optional[str]:
    return "refused harmless play" if REFUSAL.search(reply) else None


def _all(*checks):
    def run(reply):
        for c in checks:
            problem = c(reply)
            if problem:
                return problem
        return None
    return run


def _turn(role: str, text: str) -> dict:
    return {"role": role, "content": text}


CASES: list[Case] = [
    Case("name", "asked who it is, it dodged or sounded like a stock assistant",
         "Who are you? What's your name?",
         check=_has(r"prometheus", "its name (Prometheus)")),
    Case("write-help", "asked for writing help, got a refusal / generic advice",
         "Help me write the opening line of an essay about grief.",
         check=_all(_no_refusal, _substantive(14))),
    Case("yeah-continues", "answered 'yeah' with 'what do you mean' instead of continuing the thread",
         "Yeah.",
         history=[_turn("user", "Should I rebuild the rig or buy a new one?"),
                  _turn("assistant", "Rebuild, honestly. The board is fine and the fan is the weak link. Want the steps?")],
         check=_all(_substantive(15), lambda r: "what do you mean" if re.search(r"what do you mean|could you clarify|which one", r, re.I) else None)),
    Case("repeat-after-me", "treated a movie line as a threat",
         "Repeat after me: I'll be back.",
         check=_all(_no_refusal, _has(r"i'?ll be back", "the line itself"))),
    Case("conundrum", "chat answer to a real problem: no actual options",
         "I have sixteen gigs of VRAM and I keep running out when two models load. What would you do?",
         no_route=r"^tool:",
         check=_all(_substantive(20), _has(r"quantiz|offload|one at a time|sequential|unload|smaller|queue|swap", "a concrete option"))),
    Case("recall", "claimed to remember nothing despite a full memory",
         "What do you remember about me?",
         check=_all(_substantive(8),
                    lambda r: "claims no memory" if re.search(r"don'?t (have any|remember anything|know anything)|no memor|nothing (about you|yet)", r, re.I) else None)),
    Case("thread-followup", "short follow-up lost the topic",
         "And what about the second one?",
         history=[_turn("user", "Give me two names for a hobby project about tracking my sleep."),
                  _turn("assistant", "Nightledger, and Dozeline. Nightledger sounds more serious, Dozeline more playful.")],
         check=_has(r"dozeline|second|playful", "a reference to the second name")),
    # ── the owner's own words, against the real gateway and DUM-E ─────────────────────────────
    Case("real-ram", "'can you tell how much RAM' never reached DUM-E; the model said it had no data",
         "Can you tell how much RAM is being used right now?", route=r"^tool:"),
    Case("real-gpu-model", "'no, I mean can you tell me the exact model running in the GPU' was not routed",
         "No, I mean can you tell me the exact model which is running right now in the GPU?", route=r"^tool:"),
    Case("real-disk", "'Anyways tell me the exact status of the hard disk' got an improvised answer",
         "Anyways tell me what's the exact status of the hard disk right now across the machine", route=r"^tool:"),
    Case("real-garbled", "a garbled opening ('Kenya's dummy') lost the agent",
         "Kenya's dummy, what are the models loaded in the GPU?", route=r"^tool:"),
    Case("model-calls-agent", "wording no phrase rule knows: the model itself must reach DUM-E",
         "Is anything weird going on with my graphics card right now?", route=r"^tool:"),
    Case("model-calls-agent-2", "another wording no rule knows",
         "Hey, how hot is the machine running right now?", route=r"^tool:"),
    Case("model-calls-agent-3", "another wording no rule knows",
         "Is the thing that runs my AI models doing okay, any trouble?", route=r"^tool:"),
    Case("model-ask-it-for-me", "'I mean you can ask it for me' got 'I can't run DUM-E right now'",
         "I mean you can ask it for me",
         history=[_turn("user", "Can you tell how much RAM is being used right now?"),
                  _turn("assistant", "I don't have that data right now. Ask DUM-E how much RAM is being used.")],
         route=r"^tool:"),
    Case("chat-not-hijacked-1", "talking about dummy is not a request to ask it",
         "Did you just call dummy for that? I don't think you did", no_route=r"^tool:"),
    Case("chat-not-hijacked-2", "a plan to research a story must not start the Detective",
         "I have to research on a story for an episodic season or a short film.", no_route=r"^tool:"),
    Case("chat-not-hijacked-3", "a feeling about the system of government is not a machine question",
         "What do you think about the system of government?", no_route=r"^tool:"),
    Case("movie-lookup-goes-to-detective", "a film summary needs a real web lookup: Detective's scout, not an invention",
         "Tell me the summary of the movie You Can See Everything that has been released this year.",
         route=r"^tool:"),
    Case("cmd-note", "a note command must be answered by vself, not improvised",
         "Take a note that the rig needs a new fan.",
         direct=r"noted|got it|saved|note"),
    Case("cmd-task", "a task command must be answered by vself",
         "Add a task to check the TAP prices on Monday.",
         direct=r"added|task"),
    Case("cmd-list", "listing tasks must come from the store, never invented",
         "What's on my task list?",
         context=r"\[Open tasks",
         check=_all(_substantive(3),
                    lambda r: "claims it cannot see tasks" if re.search(r"(don'?t|can'?t|do not) (have|see|access)", r, re.I) else None)),
]


@dataclass
class Result:
    case: Case
    reply: str
    route: str
    problems: list[str]
    elapsed: float = 0.0
    first: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.problems


def grade(case: Case, reply: str, direct_reply: Optional[str], prompt: str = "", route: str = "") -> list[str]:
    if case.context is not None and not re.search(case.context, prompt):
        return [f"vself did not put /{case.context}/ in the prompt"]
    if case.no_route and re.search(case.no_route, route):
        return [f"took the {route} route for plain conversation"]
    if "[[" in reply:
        return ["a raw agent call line leaked into the speech"]
    if case.route is not None:
        if not re.search(case.route, route):
            return [f"went to {route!r}, not an agent ({case.route})"]
        if not reply.startswith(("DUM-E", "Detective", "Dexter", "ODA", "Asking")):
            return [f"the agent's answer was not read out: {reply[:80]!r}"]
        return [m for m in ("markdown in speech" if MARKDOWN.search(reply) else "",) if m]  # an agent's own words: no length cap
    if case.direct is not None:
        if not direct_reply:
            return ["vself did not answer the command itself"]
        return [] if re.search(case.direct, direct_reply, re.I) else [f"direct reply lacks /{case.direct}/"]
    problems = general_problems(reply)
    if case.check:
        extra = case.check(reply)
        if extra:
            problems.append(extra)
    return problems


def run_case(brain, case: Case) -> Result:
    brain.history = list(case.history)
    messages, direct = brain.prepare_full(case.user)
    if case.direct is not None:
        return Result(case, direct or "", "vself:direct", grade(case, direct or "", direct))
    prompt = "\n".join(m["content"] for m in messages)
    if getattr(brain, "tools", None) is not None:
        brain.tools._prev = None  # each case stands alone: "ask it for me" must not borrow an earlier case
        while brain.tools.take_announcement():  # ...and a slow answer to an earlier case must not leak into this one
            pass
    t0 = time.monotonic()
    pieces, first = [], 0.0
    for piece in brain.chat_stream(case.user, commit=False):
        if not first:
            first = time.monotonic() - t0
        pieces.append(piece)
    reply = "".join(pieces)
    elapsed = time.monotonic() - t0
    route = brain.last_route or "?"
    return Result(case, reply, route, grade(case, reply, direct, prompt, route), elapsed, first)


def main(argv: list[str]) -> int:
    import yaml

    from vself_brain import VselfBrain

    cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    brain = VselfBrain.from_config(cfg["brains"]["vself"], cfg["brains"])
    if getattr(brain, "cursor", None) is not None:  # give Composer its ~4 s start-up so the cases use it
        brain.cursor.warm()
        for _ in range(100):
            if brain.cursor.available():
                break
            time.sleep(0.2)
        brain.cursor.prime(brain.prepare_full("Hello.")[0][0]["content"])  # what warm() does at real start-up
        time.sleep(6)
    wanted = argv[1:]
    cases = [c for c in CASES if not wanted or any(c.id.startswith(w) for w in wanted)]
    failed = 0
    for case in cases:
        try:
            res = run_case(brain, case)
        except Exception as e:  # a down brain is a failed gate, not a crash
            res = Result(case, "", "error", [f"{type(e).__name__}: {e}"])
        mark = "PASS" if res.ok else "FAIL"
        print(f"{mark}  {case.id:<22} {res.route:<34} first {res.first:4.1f}s  total {res.elapsed:4.1f}s")
        if not res.ok:
            failed += 1
            print(f"      why it matters: {case.why}")
            print(f"      problems: {'; '.join(res.problems)}")
            print(f"      reply: {res.reply.strip()[:300]!r}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    brain.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
