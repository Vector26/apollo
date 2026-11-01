"""Mic and speaker discovery that survives a cold boot: wait for PipeWire and the devices, prefer what is
configured, fall back to anything that works, bring Bluetooth earbuds up if they are in range, and never
leave a device muted. Light on purpose (no models): the boot job and `apollo talk` both use it.

    python -m audio_devices ensure      # what the boot service runs; always exits 0
    python -m audio_devices status      # what would be picked right now
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

_WPCTL_SECTION = re.compile(r"^[\s│├└─]*([A-Z][A-Za-z ]+):\s*$")
_WPCTL_NODE = re.compile(r"^[\s│]*(\*)?\s*(\d+)\.\s+(.+?)\s*(?:\[[^\]]*\])?\s*$")
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def read_status() -> str:
    return _run(["wpctl", "status"], timeout=5.0)


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    default: bool


def list_nodes(status_text: str, kind: str) -> list[Node]:
    """Audio nodes of one section ("sinks" or "sources") from `wpctl status`."""
    nodes: list[Node] = []
    in_audio, section = False, ""
    for line in status_text.splitlines():
        if line and not line[0].isspace() and line[0] not in "│├└":
            in_audio, section = line.strip() == "Audio", ""
            continue
        sm = _WPCTL_SECTION.match(line)
        if sm:
            section = sm.group(1).strip().lower()
            continue
        if in_audio and section == kind and (nm := _WPCTL_NODE.match(line)):
            nodes.append(Node(nm.group(2), nm.group(3), bool(nm.group(1))))
    return nodes


def parse_wpctl_id(status_text: str, name: str, kind: str = "sinks") -> Optional[str]:
    """Node id for ``name`` in the Audio ``kind`` section. Exact name beats substring; the starred (default)
    node breaks ties."""
    want = name.lower().strip()
    hits = [(int(n.name.lower() == want), int(n.default), n.id) for n in list_nodes(status_text, kind)
            if want in n.name.lower()]
    return sorted(hits, reverse=True)[0][2] if hits else None


def find_wpctl_id(name: str, kind: str = "sinks") -> Optional[str]:
    text = read_status()
    return parse_wpctl_id(text, name, kind) if text else None


def audio_up(status_text: str) -> bool:
    """PipeWire answers and has at least one output and one input."""
    return bool(list_nodes(status_text, "sinks") and list_nodes(status_text, "sources"))


def has_any(status_text: str, names: list[str], kind: str) -> bool:
    return any(parse_wpctl_id(status_text, n, kind) for n in names if n)


def wait_for_audio(
    preferred_mics: list[str],
    timeout_s: float,
    poll_s: float = 1.0,
    read: Optional[Callable[[], str]] = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[str, bool]:
    """Wait (bounded) for PipeWire and the FIRST-choice mic (a fallback must not win a few seconds before the
    real one enumerates). Returns (status, first_choice_present). Never raises and never waits past the timeout:
    a missing mic means a fallback is used, not a hang."""
    read = read or read_status
    first = preferred_mics[:1]
    end = clock() + max(0.0, timeout_s)
    status = read()
    while True:
        if audio_up(status) and (not first or has_any(status, first, "sources")):
            return status, True
        if clock() >= end:
            return status, bool(first) and has_any(status, first, "sources")
        sleep(poll_s)
        status = read()


@dataclass(frozen=True)
class MicChoice:
    name: str
    id: str
    how: str  # preferred | usb | default | any


def choose_mic(status_text: str, preferred: list[str]) -> Optional[MicChoice]:
    """The configured mic if it is there; else any USB mic; else the system default input; else any input."""
    for name in preferred:
        if name and (i := parse_wpctl_id(status_text, name, "sources")):
            return MicChoice(name, i, "preferred")
    sources = [n for n in list_nodes(status_text, "sources") if "monitor" not in n.name.lower()]
    for n in sources:
        if "usb" in n.name.lower():
            return MicChoice(n.name, n.id, "usb")
    for n in sources:
        if n.default:
            return MicChoice(n.name, n.id, "default")
    return MicChoice(sources[0].name, sources[0].id, "any") if sources else None


def ensure_audible(node_id: str, run: Callable[..., str] = _run) -> bool:
    """Unmute a node, and lift it if it was left at (near) zero. A level the owner chose is left alone."""
    changed = False
    out = run(["wpctl", "get-volume", node_id], 3.0)
    if "MUTED" in out:
        run(["wpctl", "set-mute", node_id, "0"], 3.0)
        changed = True
    m = re.search(r"Volume:\s*([0-9.]+)", out)
    if m and float(m.group(1)) < 0.05:
        run(["wpctl", "set-volume", node_id, "1.0"], 3.0)
        changed = True
    return changed


def _paired_bluetooth(run: Callable[..., str] = _run) -> dict[str, str]:
    """{device name: MAC} for every paired Bluetooth device."""
    out = run(["bluetoothctl", "devices", "Paired"], 4.0)
    return {m.group(2).strip(): m.group(1) for m in re.finditer(r"^Device\s+([0-9A-F:]{17})\s+(.+)$", out, re.M | re.I)}


def connect_bluetooth(names: list[str], run: Callable[..., str] = _run, timeout_s: float = 10.0) -> list[str]:
    """Bring up paired Bluetooth speakers from ``names`` that are not connected (out of range = quietly skipped)."""
    paired = _paired_bluetooth(run)
    done: list[str] = []
    for name in names:
        mac = next((m for n, m in paired.items() if n.lower() == name.lower()), None)
        if not mac or "Connected: yes" in run(["bluetoothctl", "info", mac], 4.0):
            continue
        if "successful" in run(["bluetoothctl", "connect", mac], timeout_s).lower():
            done.append(name)
    return done


class DeviceWatcher:
    """Keeps the mic right while Apollo runs: if the mic in use disappears (USB unplugged) switch to the best one
    left; if a fallback is in use and the preferred mic comes back, switch back. Quiet otherwise: a changed
    system default alone never moves it."""

    def __init__(self, mics: list[str], current_id: Optional[str], current_how: str,
                 read: Optional[Callable[[], str]] = None,
                 set_default: Optional[Callable[[str], None]] = None) -> None:
        self.mics, self.current_id, self.how = mics, current_id, current_how
        self.read = read or read_status
        self.set_default = set_default or (lambda node: _run(["wpctl", "set-default", node], 4.0))
        self._warned = False

    def step(self) -> Optional[str]:
        """One look. Returns a sentence to show when something changed, else None."""
        status = self.read()
        if not list_nodes(status, "sinks"):
            return None  # PipeWire hiccup or restarting (it always has an output): judge nothing on a blank read
        still_there = any(n.id == self.current_id for n in list_nodes(status, "sources"))
        choice = choose_mic(status, self.mics)
        if choice is None:
            if not self._warned:
                self._warned = True
                return "no microphone found: plug one in, I'll pick it up"
            return None
        self._warned = False
        better = self.how != "preferred" and choice.how == "preferred"
        if still_there and not better:
            return None
        if choice.id == self.current_id:
            return None
        self.set_default(choice.id)
        ensure_audible(choice.id)
        before, self.current_id, self.how = self.current_id, choice.id, choice.how
        why = "back on" if better and still_there else "mic lost, now using"
        return f"{why} '{choice.name}' ({choice.how})" if before else f"using '{choice.name}' ({choice.how})"


def preferences(cfg: dict) -> tuple[list[str], list[str]]:
    """(speaker names in priority order, mic names in priority order) from the config."""
    bt = cfg.get("bluetooth") or {}
    speakers = [str(n) for n in (bt.get("speaker_names") or [bt.get("speaker_name", "Dash 4")])]
    mics = [str(n) for n in ([bt.get("mic_name", "USB Composite Device")] + list(bt.get("mic_fallbacks") or []))]
    return speakers, mics


def ensure(cfg: dict, wait_s: float = 120.0) -> dict:
    """Boot-time readiness. Always returns a summary; nothing here is allowed to fail the boot."""
    speakers, mics = preferences(cfg)
    summary: dict = {"waited_for_audio": False, "bluetooth": [], "mic": None, "speaker": None, "fixed_levels": []}
    status, have_mic = wait_for_audio(mics, wait_s)
    summary["waited_for_audio"] = audio_up(status)
    try:
        summary["bluetooth"] = connect_bluetooth([s for s in speakers if not parse_wpctl_id(status, s, "sinks")])
        if summary["bluetooth"]:
            time.sleep(2.0)  # the sink appears a moment after the link is up
            status = read_status() or status
    except Exception as e:  # Bluetooth is a bonus
        summary["bluetooth_error"] = type(e).__name__
    mic = choose_mic(status, mics)
    if mic:
        summary["mic"] = {"name": mic.name, "how": mic.how}
        if ensure_audible(mic.id):
            summary["fixed_levels"].append(mic.name)
    for name in speakers:
        if sink := parse_wpctl_id(status, name, "sinks"):
            summary["speaker"] = name
            if ensure_audible(sink):
                summary["fixed_levels"].append(name)
            break
    return summary


def main(argv: list[str]) -> int:
    import yaml

    cmd = argv[1] if len(argv) > 1 else "status"
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if cmd == "ensure":
        print(json.dumps(ensure(cfg, wait_s=float((cfg.get("audio") or {}).get("boot_wait_s", 120)))))
        return 0  # a missing device must never fail the boot; Apollo falls back when it starts
    speakers, mics = preferences(cfg)
    status = read_status()
    mic = choose_mic(status, mics)
    print(json.dumps({"audio_up": audio_up(status), "mic": mic and {"name": mic.name, "how": mic.how},
                      "speaker": next((s for s in speakers if parse_wpctl_id(status, s, "sinks")), None)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
