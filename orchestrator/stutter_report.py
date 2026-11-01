"""Summarize voice stalls per session from talk.log.

    python stutter_report.py [talk.log] [--last N]

Reads the lines the live loop writes ("clause synth ...", "speaker gap ...", "WarmSpeaker
started", "Loading RVC") and says whether synthesis is slower than playback (rtf > 1), how
often the speaker went silent mid-reply, and whether the voice ran through RVC.
"""

from __future__ import annotations

from apollo_home import apollo_path
import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LOG = Path(apollo_path("logs/talk.log"))
_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_SYNTH = re.compile(r"clause synth (\d+)ms for audio (\d+)ms \(rtf ([\d.]+)\)")
_GAP = re.compile(r"speaker gap (\d+)ms between chunks")
STALL_MS = 300  # a silence this long inside a reply is audible


@dataclass
class Session:
    started: str = ""
    rvc: bool = False
    synth_ms: list[int] = field(default_factory=list)
    audio_ms: list[int] = field(default_factory=list)
    rtf: list[float] = field(default_factory=list)
    gaps_ms: list[int] = field(default_factory=list)

    def has_data(self) -> bool:
        return bool(self.synth_ms or self.gaps_ms)


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def parse(lines) -> list[Session]:
    sessions: list[Session] = []
    cur: Session | None = None
    rvc_loaded = False  # models load during warm-up, before the speaker (and so the session) starts
    for line in lines:
        stamp = _STAMP.match(line)
        if "Loading RVC" in line:
            rvc_loaded = True
        elif "WarmSpeaker started" in line:
            cur = Session(started=stamp.group(1) if stamp else "", rvc=rvc_loaded)
            sessions.append(cur)
            rvc_loaded = False
            continue
        if cur is None:
            continue
        if m := _SYNTH.search(line):
            cur.synth_ms.append(int(m.group(1)))
            cur.audio_ms.append(int(m.group(2)))
            cur.rtf.append(float(m.group(3)))
        elif m := _GAP.search(line):
            cur.gaps_ms.append(int(m.group(1)))
    return sessions


def summarize(s: Session) -> str:
    stalls = [g for g in s.gaps_ms if g >= STALL_MS]
    slow = sum(1 for r in s.rtf if r > 1.0)
    parts = [
        f"{s.started or '?'}  rvc={'on' if s.rvc else 'off'}",
        f"clauses={len(s.synth_ms)} synth p50/p90/max={pct(s.synth_ms, 50):.0f}/{pct(s.synth_ms, 90):.0f}/{max(s.synth_ms, default=0)}ms"
        f" rtf p50/max={pct(s.rtf, 50):.2f}/{max(s.rtf, default=0):.2f} slower-than-audio={slow}",
        f"gaps={len(s.gaps_ms)} stalls(>={STALL_MS}ms)={len(stalls)} worst={max(s.gaps_ms, default=0)}ms",
    ]
    if stalls and slow:
        parts.append("reading: stalls coincide with synthesis slower than audio -> TTS/RVC is the bottleneck")
    elif stalls:
        parts.append("reading: stalls with fast synthesis -> the brain's token cadence (or the sink), not TTS")
    elif s.has_data():
        parts.append("reading: no audible mid-reply stalls")
    return "\n  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", nargs="?", default=str(DEFAULT_LOG))
    ap.add_argument("--last", type=int, default=3, help="how many recent sessions with data to show")
    args = ap.parse_args(argv)
    path = Path(args.log)
    if not path.exists():
        print(f"no log at {path}", file=sys.stderr)
        return 1
    sessions = [s for s in parse(path.read_text(errors="replace").splitlines()) if s.has_data()]
    if not sessions:
        print("no clause/gap data yet: run a conversation on v2.0.0 first")
        return 0
    for s in sessions[-args.last:]:
        print(summarize(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
