#!/usr/bin/env python3
"""Vox live talk CLI — meter when you speak, the chosen voice replies on the speaker."""

from __future__ import annotations

from apollo_home import APOLLO_HOME, apollo_path
import argparse
import logging
import random
import os
import queue
import re
import subprocess
import sys
import threading
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import yaml

import dns_cache
from acks import Acks
from announce import Announcer
from audio_devices import DeviceWatcher, choose_mic, preferences as device_preferences, connect_bluetooth, ensure_audible, find_wpctl_id, parse_wpctl_id, read_status, wait_for_audio
from audio_io import ContinuousCapture
from carry import UnansweredCarry
from echo_gate import EchoGate
from endpoint_hold import FragmentHold, ends_dangling, is_filler_only
from halt import HALT_MESSAGE, is_halted
from initiative import Context, Initiative, clean_aside
from llm_client import LLMClient
from memory import MemoryStore
from notes import NotesManager
from speaker import WarmSpeaker
from stt_stream import STTWorker, build_stt, wav_bytes_to_f32
from tts_engine import ChatterboxEngine, ClauseSplitter
from usb_mic_linux import apply_usb_mic_linux_fixes
from version import version
from voice_interrupt import is_interrupt_command

VOX_ROOT = APOLLO_HOME
DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"

ESC = "\033"
ALT_ON = f"{ESC}[?1049h{ESC}[?25l"
ALT_OFF = f"{ESC}[?1049l{ESC}[?25h"
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
DIM = f"{ESC}[2m"
GREEN = f"{ESC}[32m"
CYAN = f"{ESC}[36m"
YELLOW = f"{ESC}[33m"
RED = f"{ESC}[31m"
WHITE = f"{ESC}[37m"


def vu_bar(level: float, width: int = 24) -> str:
    level = max(0.0, min(1.0, level))
    filled = int(round(level * width))
    return "█" * filled + "░" * (width - filled)


def speaker_preferences(cfg: dict) -> list[str]:
    bt = cfg.get("bluetooth") or {}
    names = bt.get("speaker_names") or [bt.get("speaker_name", "Dash 4")]
    return [str(n) for n in names]


def pick_speaker(cfg: dict) -> tuple[str, str | None]:
    """First preferred speaker PipeWire has as a sink (a Bluetooth sink only
    exists while the device is connected); (first name, None) if none is."""
    names = speaker_preferences(cfg)
    try:
        status = subprocess.check_output(["wpctl", "status"], text=True, errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return names[0], None
    for name in names:
        sink = parse_wpctl_id(status, name, "sinks")
        if sink:
            return name, sink
    return names[0], None


def resolve_devices(cfg: dict) -> tuple[str, str | None, str, str | None, str]:
    """(speaker name, sink id, mic name, mic source id, note). Right after a boot the devices may not exist yet:
    wait (bounded) for PipeWire and the configured mic, bring paired Bluetooth speakers up, then fall back
    (configured mic -> any USB mic -> system default input; preferred speaker -> default output). Never raises."""
    from audio_devices import preferences

    audio_cfg = cfg.get("audio") or {}
    speakers, mics = preferences(cfg)
    note = ""
    try:
        status, have_mic = wait_for_audio(mics, float(audio_cfg.get("device_wait_s", 30)))
        missing = [n for n in speakers if not parse_wpctl_id(status, n, "sinks")]
        if missing and audio_cfg.get("bluetooth_connect", True) and connect_bluetooth(missing, timeout_s=6.0):
            time.sleep(2.0)  # the sink appears a moment after the link is up
        status = read_status() or status
    except Exception:
        logging.exception("device wait failed; carrying on with what is there")
        status, have_mic = "", False
    speaker, sink = pick_speaker(cfg)
    choice = choose_mic(status, mics) if status else None
    if choice is None:
        return speaker, sink, mics[0], find_wpctl_id(mics[0], "sources"), note
    if choice.how != "preferred":
        note = f"mic '{mics[0]}' not found: using '{choice.name}' ({choice.how})"
    for node in (choice.id, sink):
        if node:
            ensure_audible(node)
    return speaker, sink, choice.name, choice.id, note


def find_input_device(name: str) -> int | None:
    """Capture via PipeWire at 48 kHz.

    Do not open the DWM-123 dongle as raw ALSA hw: — that exclusive-locks
    the full-speed USB PCM and the mic drops after overflows.
    """
    import sounddevice as sd
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        if dev["name"] == "pipewire":
            return i
    return None


class TalkUI:
    HEADER = 6

    def __init__(self):
        self.mode = "starting"
        self.level = 0.0
        self.voicing = False
        self.hint = "say something — pause when you're done"
        self.assistant_slug = "jarvis"
        self.assistant_label = "JARVIS"
        self.assistant_banner = "Jarvis on Dash 4"
        self._lock = threading.Lock()
        self._last_draw = 0.0
        self._tty_attrs = None

    def _echo(self, on: bool) -> None:
        """Keep keystrokes (and stray escape codes) from being drawn over the screen while talking."""
        try:
            import termios

            fd = sys.stdin.fileno()
            if not os.isatty(fd):
                return
            if on:
                if self._tty_attrs is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, self._tty_attrs)
                    self._tty_attrs = None
                return
            attrs = termios.tcgetattr(fd)
            self._tty_attrs = list(attrs)
            attrs[3] &= ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        except Exception:  # not a terminal (tests, pipes): nothing to do
            self._tty_attrs = None

    def enter(self):
        self._echo(False)
        sys.stdout.write(ALT_ON)
        rows = self._rows()
        sys.stdout.write(f"{ESC}[{self.HEADER + 1};{rows}r")
        sys.stdout.flush()
        self.draw(force=True)

    def leave(self):
        sys.stdout.write(f"{ESC}[r{ALT_OFF}")
        sys.stdout.flush()
        self._echo(True)

    def set(self, mode=None, level=None, voicing=None, hint=None):
        with self._lock:
            if mode is not None:
                self.mode = mode
            if level is not None:
                self.level = level
            if voicing is not None:
                self.voicing = voicing
            if hint is not None:
                self.hint = hint
        self.draw()

    def line(self, who: str, text: str):
        color = CYAN if who == "you" else GREEN
        sys.stdout.write(f"{ESC}[{self.HEADER + 1}H{ESC}[999B")
        sys.stdout.write(f"{color}{BOLD}{who:<7}{RESET} {text}\n")
        sys.stdout.flush()
        self.draw(force=True)

    def draw(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_draw < 0.05:
            return
        self._last_draw = now
        with self._lock:
            mode, level, voicing, hint = self.mode, self.level, self.voicing, self.hint
        if voicing:
            label = f"{RED}{BOLD}YOU{RESET}"
        elif mode == "assistant":
            label = f"{GREEN}{BOLD}{self.assistant_label}{RESET}"
        else:
            label = f"{GREEN}{BOLD}LIVE{RESET}"
        show_meter = voicing or mode == "live"
        bar = vu_bar(level if show_meter else 0.0)
        pct = int(level * 100) if show_meter else 0
        sys.stdout.write(f"{ESC}[H")
        sys.stdout.write(
            f"{BOLD} VOX{RESET}  {DIM}{self.assistant_banner}{RESET}          Ctrl-C or say  {BOLD}stop{RESET}\n"
        )
        sys.stdout.write(f" {label}   {GREEN}{bar}{RESET}  {pct:3d}%\n")
        sys.stdout.write(f" {DIM}{hint}{RESET}\n")
        sys.stdout.write(f"{DIM}────────────────────────────────────────────────────────────{RESET}\n")
        sys.stdout.write("\n\n")
        sys.stdout.flush()

    def _rows(self) -> int:
        try:
            return int(os.get_terminal_size().lines)
        except OSError:
            return 24


def deliver_clause(speaker, ui, clause: str, synth, cancel=None, on_start=None) -> bool:
    """Synthesize and queue audio; the caption prints when that audio starts — never text before voice."""
    pcm, sr = synth(clause)
    if cancel is not None and cancel.is_set():
        return False
    who = getattr(ui, "assistant_slug", "jarvis")

    def started():
        if cancel is not None and cancel.is_set():
            return
        ui.line(who, clause)
        if on_start is not None:
            on_start()

    speaker.write_pcm(pcm, sr, on_start=started)
    return True


def log_clause_timing(synth_s: float, pcm, sr: int) -> None:
    """One line per synthesized clause: if synthesis is slower than the audio it makes (rtf > 1),
    the next clause cannot arrive in time and the voice stutters."""
    audio_s = len(pcm) / sr if sr else 0.0
    rtf = synth_s / audio_s if audio_s > 0 else 0.0
    logging.info("clause synth %.0fms for audio %.0fms (rtf %.2f)", synth_s * 1000, audio_s * 1000, rtf)


FAIL_LINE = "I lost my connection for a second. Say that again."
STARTUP_LINES = (
    "I'm listening.",
    "Go ahead whenever you like.",
    "I'm here.",
    "Ready when you are.",
    "What do you need?",
    "Whenever you're ready.",
)
FIXED_LINES = STARTUP_LINES + ("Goodbye.", "Notes saved.", FAIL_LINE)
REPEAT_MIN_WORDS = 4


def _said_key(clause: str) -> str | None:
    """Normalized form of a clause long enough that saying it twice is a repeat."""
    words = re.findall(r"[a-z0-9']+", clause.lower().replace("\u2019", "'"))
    return " ".join(words) if len(words) >= REPEAT_MIN_WORDS else None


class SpokenLog:
    """Sentences already spoken this session; a model looping on them is cut."""

    def __init__(self):
        self._said: set[str] = set()

    def is_repeat(self, clause: str) -> bool:
        key = _said_key(clause)
        return key is not None and key in self._said

    def record(self, reply: str) -> None:
        for part in re.split(r"(?<=[.!?…;:])\s+|—|\n+", reply):
            key = _said_key(part)
            if key:
                self._said.add(key)
EXIT_WORDS = {"stop", "quit", "exit", "goodbye"}
_EXIT_PHRASES = EXIT_WORDS | {
    "bye", "bye bye", "bye for now", "good night", "goodnight", "see you", "see you later",
    "see you tomorrow", "see ya", "talk to you later", "talk later", "that's all", "that's it for now",
    "i'm done", "we're done", "we are done",
}
_LEAD_IN = {"okay", "ok", "alright", "all right", "well", "right", "so", "yeah", "yes", "um", "uh"}
_TAIL_OUT = {"prometheus", "ultron", "apollo", "vox", "buddy", "thanks", "thank you", "now", "then", "for now"}
_DUMP_PHRASES = {
    "take notes", "dump notes", "save notes", "write notes", "take the notes", "save the notes",
    "dump the notes", "write the notes", "write up the notes",
}


def _plain(text: str) -> str:
    """Lower-case words only: punctuation and hyphens gone, curly apostrophes straightened."""
    t = text.lower().replace("\u2019", "'").replace("-", " ")
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def _strip_edges(text: str, lead=_LEAD_IN, tail=_TAIL_OUT) -> str:
    t = _plain(text)
    changed = True
    while changed:
        changed = False
        for w in sorted(lead, key=len, reverse=True):
            if t.startswith(w + " "):
                t, changed = t[len(w) + 1:], True
        for w in sorted(tail, key=len, reverse=True):
            if t.endswith(" " + w):
                t, changed = t[: -len(w) - 1], True
    return t


def is_exit(text: str) -> bool:
    """A whole-utterance goodbye ("bye-bye", "okay, good night Prometheus"); never a sentence containing one."""
    return _strip_edges(text) in _EXIT_PHRASES


def is_dump_request(text: str) -> bool:
    """"Take notes" said on its own (also "please take notes now"), not "can you take notes on this idea"."""
    return _strip_edges(text, _LEAD_IN | {"please", "can you", "could you"}, _TAIL_OUT | {"please"}) in _DUMP_PHRASES


class Turn:
    """One user utterance and the reply to it.

    A speculative turn starts STT + LLM + TTS at an early pause, but nothing is
    played, captioned, or written to history until ``confirmed`` is set.
    """

    _next_id = 0

    def __init__(self, speculative: bool, t_user_stop: float):
        Turn._next_id += 1
        self.id = Turn._next_id
        self.speculative = speculative
        self.cancel = threading.Event()
        self.confirmed = threading.Event()
        self.text: str | None = None
        self.handled = False       # transcript shown / command run
        self.llm_started = False
        self.failed = False
        self.hold = False          # ended on a dangling word: wait for the rest
        self.merged = False        # continues a held fragment
        self.reply: list[str] = []
        self.route: str | None = None
        self.t_user_stop = t_user_stop
        self.t_confirm = 0.0
        self.t_stt = 0.0
        self.t_token = 0.0
        self.t_audio = 0.0
        self.t_filler = 0.0

    def wait_confirmed(self, stop: threading.Event) -> bool:
        """Block until confirmed (True) or cancelled/stopped (False)."""
        while not self.cancel.is_set() and not stop.is_set():
            if self.confirmed.wait(0.01):
                return not self.cancel.is_set()
        return False


def _warm_llm(llm: LLMClient) -> None:
    """Open the HTTPS connection (DNS + TLS) before the first real turn."""
    try:
        llm.complete("Reply with OK.", "ping", max_tokens=1)
    except Exception as e:
        logging.warning("LLM warm-up failed: %s", e)


def talk_llm_config(cfg: dict, profile) -> tuple[dict, str]:
    """LLM settings + system prompt for a voice profile (its brain and persona)."""
    llm_cfg = dict(cfg["llm"])
    brain = getattr(profile, "brain", None)
    if brain:
        brains = cfg.get("brains") or {}
        if brain not in brains:
            raise KeyError(f"voice {profile.profile_id!r} wants brain {brain!r}; add brains.{brain} to config.yaml")
        llm_cfg.update(brains[brain])
        llm_cfg.pop("fallback", None)
        if brains[brain].get("fallback_to_local"):
            llm_cfg["fallback"] = dict(cfg["llm"])
    persona = getattr(profile, "persona", None)
    return llm_cfg, persona or llm_cfg["system_prompt"]


def _load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_talk(cfg: dict, voice: str | None = None):
    if is_halted():
        print(HALT_MESSAGE, file=sys.stderr)
        sys.exit(1)
    log_dir = Path(os.environ.get("VOX_LOG_DIR", apollo_path("logs")))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=str(log_dir / "talk.log"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ui = TalkUI()
    stop = threading.Event()
    ui.enter()
    try:
        _run_talk(cfg, ui, stop, voice=voice)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        ui.leave()
        print("Vox stopped.")


def _run_talk(cfg: dict, ui: TalkUI, stop: threading.Event, voice: str | None = None):
    from voice.presets import load_profile

    apply_usb_mic_linux_fixes()
    ui.set(mode="starting", hint="checking brain…")
    tts_cfg = cfg.get("tts") or {}
    profile_id = voice or tts_cfg.get("talk_profile") or tts_cfg.get("profile", "jarvis")
    llm_cfg, system_prompt = talk_llm_config(cfg, load_profile(profile_id, tts_cfg=tts_cfg))
    dns_cache.install()
    if llm_cfg.get("kind") == "vself":
        from vself_brain import VselfBrain

        cloud_cfg = (cfg.get("brains") or {}).get(llm_cfg.get("cloud") or "", {})
        remote_url = cloud_cfg.get("base_url", "")
        llm = VselfBrain.from_config(llm_cfg, cfg.get("brains") or {})
        warm = llm.warm
    else:
        remote_url = llm_cfg["base_url"]
        llm = LLMClient.from_config(llm_cfg, system_prompt=system_prompt)
        warm = lambda: _warm_llm(llm)
    host = urlparse(remote_url).hostname if remote_url else None
    if host:
        dns_cache.prewarm(host, urlparse(remote_url).port or 443)
    owns_memory = bool(getattr(llm, "owns_memory", False))
    threading.Thread(target=warm, daemon=True).start()
    notes = NotesManager(apollo_path(cfg["notes"]["output_dir"]), llm, cfg["notes"].get("auto_dump_interval_minutes", 5))
    mem_cfg = cfg.get("memory") or {}
    memory = MemoryStore(
        root=apollo_path(mem_cfg.get("dir", "memory")),
        notes_dir=apollo_path(cfg["notes"]["output_dir"]),
        keep_turns=int(mem_cfg.get("keep_turns", 8)),
        fact_limit=int(mem_cfg.get("fact_limit", 8)),
    )

    ui.set(mode="starting", hint="finding mic and speaker…")
    speaker, sink, mic_name, src, mic_note = resolve_devices(cfg)
    if src:
        subprocess.run(["wpctl", "set-default", src], check=False)
    mic = find_input_device(mic_name)

    stt_cfg = cfg.get("stt") or {}
    audio_cfg = cfg["audio"]
    barge_in_ms = int(audio_cfg.get("barge_in_ms", 600))
    barge_in_enabled = bool(audio_cfg.get("barge_in", False))
    interrupt_keywords = bool(audio_cfg.get("interrupt_keywords", True))
    echo_tail_s = float(audio_cfg.get("echo_tail_ms", 500)) / 1000.0
    live_captions = bool(audio_cfg.get("live_captions", False))
    filler_after_s = float(audio_cfg.get("filler_after_ms", 1300)) / 1000.0
    filler_line = str(audio_cfg.get("filler_line", "One moment."))
    acks = Acks(audio_cfg.get("acks"))
    eg_cfg = audio_cfg.get("echo_gate") or {}
    echo_gate = EchoGate(window_s=float(eg_cfg.get("window_s", 8.0)), drop_at=float(eg_cfg.get("drop_at", 0.7)),
                         min_words=int(eg_cfg.get("min_words", 4))) if eg_cfg.get("enabled", False) else None
    slow_turn_ms = int(audio_cfg.get("slow_turn_ms", 1500))
    hold_s = float(audio_cfg.get("hold_fragment_ms", 2500)) / 1000.0
    hold = FragmentHold(window_s=hold_s)
    can_end = callable(getattr(llm, "end", None))  # the brain keeps sessions (vself): it files the notes
    idle_end_s = float(llm_cfg.get("idle_end_s", 600))

    stt_engine = str(stt_cfg.get("engine") or "whisper").lower()
    ui.set(mode="starting", hint=f"loading ear ({'Parakeet' if stt_engine == 'parakeet' else 'Whisper'})…")
    stt = build_stt(stt_cfg)
    ui.set(mode="starting", hint="loading voice (Chatterbox)…")
    tts = ChatterboxEngine(
        model_dir=apollo_path(tts_cfg.get("model_dir", "models/chatterbox-turbo")),
        ref_audio=apollo_path(tts_cfg.get("ref_audio", "models/refs/default.wav")),
        device=tts_cfg.get("device", "auto"),
        profile=profile_id,
        tts_cfg=tts_cfg,
    )
    slug = profile_id.replace("_", "")[:7] or "assist"
    ui.assistant_slug = slug
    ui.assistant_label = tts.display_name.upper()[:12]
    ui.assistant_banner = f"{tts.display_name} on {speaker}"

    ui.set(mode="starting", hint=f"warming {tts.display_name}…")
    tts.prerender(FIXED_LINES + (filler_line,) + (acks.all_lines() if acks.enabled else ()))
    stt.transcribe(np.zeros(16000, dtype=np.float32))

    def speaker_moved(target):
        ui.line("vox", "speaker lost: back on " + (f"output #{target}" if target else "the default output"))

    speaker_out = WarmSpeaker(sink, preroll_ms=int(audio_cfg.get("lead_silence_ms", 1000)),
                              resolve_target=lambda: pick_speaker(cfg)[1], on_change=speaker_moved)
    speaker_out.start()
    mem_lock = threading.Lock()
    spoken = SpokenLog()

    evq: queue.Queue = queue.Queue()
    capture = ContinuousCapture(
        sample_rate=audio_cfg["sample_rate"],
        capture_rate=audio_cfg.get("capture_rate", 48000),
        vad_threshold=audio_cfg["vad_threshold"],
        silence_ms=audio_cfg.get("silence_ms", 200),
        min_speech_ms=audio_cfg.get("min_speech_ms", 250),
        partial_ms=audio_cfg.get("partial_ms", 600),
        device=mic,
        event_queue=evq,
        stop_event=stop,
        muted=None if barge_in_enabled else (lambda: speaker_out.voice_active(echo_tail_s)),
        interrupt_listen=(
            (lambda: interrupt_keywords and speaker_out.voice_active(echo_tail_s))
            if not barge_in_enabled and interrupt_keywords
            else None
        ),
        interrupt_partial_ms=int(audio_cfg.get("interrupt_partial_ms", 450)),
        speculate_ms=int(audio_cfg.get("speculate_ms", 0)),
    )
    stt_worker = STTWorker(stt, evq)
    tts_q: queue.Queue = queue.Queue()

    # main thread owns mode/turn; workers only touch their own Turn's fields
    state: dict = {
        "mode": "listening",   # listening | responding
        "turn": None,          # latest Turn (speculative or confirmed)
        "barged": False,       # current utterance already barged in
        "cap_fail": 0,         # consecutive mic-stream failures (reset once it has run steadily again)
        "cap_restart_at": 0.0,
    }
    _, mic_prefs = device_preferences(cfg)
    level_box = {"v": 0.0}
    unanswered = UnansweredCarry()

    def tts_worker():
        while not stop.is_set():
            item = tts_q.get()
            if item is None:
                break
            kind, turn, clause = item
            if not turn.wait_confirmed(stop):
                continue
            if kind == "end":
                def played(turn=turn):
                    if not turn.cancel.is_set():
                        evq.put(("turn_done", turn.id))

                speaker_out.after_playback(played)
                continue

            def first_audio(turn=turn, clause=clause):
                if not turn.t_audio:
                    turn.t_audio = time.time()
                if echo_gate is not None:
                    echo_gate.heard(clause)

            def synth_when_confirmed(text, turn=turn):
                t0 = time.monotonic()
                out = tts.synthesize(text)
                log_clause_timing(time.monotonic() - t0, out[0], out[1])
                turn.wait_confirmed(stop)
                return out

            try:
                deliver_clause(speaker_out, ui, clause, synth_when_confirmed, turn.cancel,
                               on_start=first_audio)
            except Exception as e:
                logging.error("TTS failed: %s", e)
                ui.line("vox", f"voice failed: {e}")

    def start_llm(turn: Turn):
        turn.llm_started = True
        carried = unanswered.pending()
        text = unanswered.merged(turn.text) if carried else turn.text
        if carried:
            logging.info("turn %d also answers %d earlier unanswered utterance(s)", turn.id, len(carried))

        def gen():
            try:
                memory_block = ""
                if not owns_memory:
                    with mem_lock:
                        memory_block = memory.context_for(text)
                splitter = ClauseSplitter()
                for token in llm.chat_stream(text, memory_block=memory_block, commit=False):
                    if turn.cancel.is_set() or stop.is_set():
                        return
                    if not turn.t_token:
                        turn.t_token = time.time()
                        turn.route = llm.last_route
                    turn.reply.append(token)
                    clause = splitter.feed(token)
                    if clause and not spoken.is_repeat(clause):
                        tts_q.put(("clause", turn, clause))
                tail = splitter.flush()
                if tail and not spoken.is_repeat(tail):
                    tts_q.put(("clause", turn, tail))
                if turn.wait_confirmed(stop):
                    with mem_lock:
                        llm.commit(text, "".join(turn.reply))
                    unanswered.settle(carried)
            except Exception as e:
                turn.failed = True
                logging.exception("Turn failed: %s", e)
                if not turn.cancel.is_set():
                    if turn.reply:
                        # keep what was already said so the next turn doesn't lose the thread
                        if turn.wait_confirmed(stop):
                            with mem_lock:
                                llm.commit(text, "".join(turn.reply))
                            unanswered.settle(carried)
                    elif turn.wait_confirmed(stop):
                        ui.line("vox", "brain unreachable: asked you to repeat")
                        speak_ack(FAIL_LINE)
            finally:
                tts_q.put(("end", turn, None))

        threading.Thread(target=gen, name=f"llm-turn-{turn.id}", daemon=True).start()

    def barge_in():
        logging.info("Barge-in")
        t = state["turn"]
        if t is not None:
            if (not t.cancel.is_set() and t.text and t.confirmed.is_set() and t.llm_started
                    and not t.hold and not t.t_audio):
                unanswered.add(t.text)  # they kept talking before hearing anything: answer it with the next turn
            t.cancel.set()
        speaker_out.flush()
        state["barged"] = True
        state["mode"] = "listening"
        if state.get("announce_cancel") is not None:
            state["announce_cancel"].set()
        ui.set(mode="live", voicing=True, hint="listening — you interrupted")

    def speak_ack(text: str):
        """Pre-rendered acknowledgement (greeting, goodbye, notes saved)."""
        try:
            pcm, sr = tts.phrase(text)
            speaker_out.write_pcm(pcm, sr, on_start=lambda: heard_line(text))
        except Exception as e:
            logging.error("Ack TTS failed: %s", e)

    def heard_line(text: str):
        ui.line(ui.assistant_slug, text)
        if echo_gate is not None:
            echo_gate.heard(text)

    def say_unprompted(text: str):
        """An agent answered late (or a job finished): say it, as its own short turn the owner can talk over."""
        cancel = threading.Event()
        state["mode"], state["barged"], state["announce_cancel"] = "responding", False, cancel
        ui.set(mode="assistant", voicing=False, hint=tts.display_name)
        notes.add("assistant", text)
        spoken.record(text)

        def run():
            try:
                splitter = ClauseSplitter()
                clauses = [c for w in re.findall(r"\S+\s*", text) if (c := splitter.feed(w))]
                if tail := splitter.flush():
                    clauses.append(tail)
                for clause in clauses:
                    if cancel.is_set() or stop.is_set():
                        return
                    deliver_clause(speaker_out, ui, clause, tts.synthesize, cancel,
                                   on_start=(lambda c=clause: echo_gate.heard(c)) if echo_gate is not None else None)
                speaker_out.after_playback(lambda: evq.put(("announce_done", cancel)))
            except Exception as e:
                logging.error("announcement failed: %s", e)
                evq.put(("announce_done", cancel))

        threading.Thread(target=run, name="announce", daemon=True).start()

    def floor_is_free() -> bool:
        return (state["mode"] == "listening" and not state.get("in_speech") and not hold.pending
                and not speaker_out.voice_active(echo_tail_s))

    tools = getattr(llm, "tools", None)
    announcer = Announcer(tools, llm, say_unprompted, floor_is_free)

    ini_cfg = llm_cfg.get("initiative") or {}
    initiative = Initiative(ini_cfg) if ini_cfg.get("enabled") and callable(getattr(llm, "aside", None)) else None
    aside_busy = threading.Event()

    def maybe_aside():
        """Jarvis speaking up: cheap gates and one weighted roll here; the cloud call runs off the loop thread
        and the line is spoken from the loop only if nothing has happened since."""
        if initiative is None or aside_busy.is_set():
            return
        d = initiative.decide(Context(
            cloud=llm.cloud_in_use(), in_session=llm.has_session(), floor_free=floor_is_free(), halted=is_halted(),
            idle_s=llm.idle_s(), session_s=llm.session_age_s(), hour=datetime.now().hour,
            symbols=llm.owner_symbols()))
        if not d.ask:
            return
        mark = llm.last_activity
        facts = (f"Local time {datetime.now():%H:%M}. This conversation has run {llm.session_age_s() / 60:.0f} minutes. "
                 f"The owner's recent moods: {', '.join(llm.owner_symbols()) or 'unremarkable'}."
                 + (f" What prompted you: {d.why}." if d.why != "idle" else ""))
        logging.info("initiative: asking (tone=%s p=%.2f: %s)", d.tone, d.p, d.why)
        aside_busy.set()

        def run():
            try:
                line = clean_aside(llm.aside(d.tone, facts), initiative.recent)
                if line:
                    evq.put(("aside", line, mark))
                else:
                    initiative.nothing_to_say()
            except Exception as e:
                logging.warning("aside failed: %s", e)
                initiative.nothing_to_say()
            finally:
                aside_busy.clear()

        threading.Thread(target=run, name="aside", daemon=True).start()

    def remember_later(user_text: str, reply: str):
        def _run():
            try:
                with mem_lock:
                    memory.observe(llm, user_text, reply)
                    llm.history = memory.compact(llm, llm.history)
            except Exception as e:
                logging.warning("Memory update failed: %s", e)
        threading.Thread(target=_run, daemon=True).start()

    def is_command(text: str) -> bool:
        lower = text.lower().strip().rstrip(".!")
        return (
            is_exit(text)
            or (is_dump_request(text) if can_end else notes.is_dump_command(text))
            or memory.is_forget_all(text)
            or memory.is_forget_last(text)
        )

    def handle_transcript(text: str) -> bool:
        """Commands on the committed transcript. Returns False to stop."""
        if is_exit(text):
            if state["turn"] is not None:
                state["turn"].cancel.set()
            speaker_out.flush()
            notes.add("user", text)
            speak_ack("Goodbye.")
            if not can_end:
                notes.dump()  # a vself brain files the session itself, once, on the way out
            speaker_out.drain(timeout=5.0)
            return False
        if can_end and is_dump_request(text):
            notes.add("user", text)

            def checkpoint():
                saved = llm.end(keep_history=True)
                ui.line("vox", "notes → vself" if saved else "nothing to save yet")
                speak_ack("Notes saved." if saved else "Nothing to save yet.")

            threading.Thread(target=checkpoint, name="notes-checkpoint", daemon=True).start()
            return True
        if not can_end and notes.is_dump_command(text):
            notes.add("user", text)
            path = notes.dump()
            ui.line("vox", f"notes → {path.name}")
            speak_ack("Notes saved.")
            return True
        if memory.is_forget_all(text):
            memory.forget_all()
            with mem_lock:
                llm.reset()
            ui.line("vox", "memory cleared")
            return True
        if memory.is_forget_last(text):
            memory.forget_last()
            ui.line("vox", "forgot last fact")
            return True
        notes.add("user", text)
        return True

    def schedule_hold_expire(seq: int, delay: float):
        timer = threading.Timer(delay, lambda: evq.put(("hold_expire", seq)))
        timer.daemon = True
        timer.start()

    def finish_turn(turn: Turn):
        if state["turn"] is turn:
            state["mode"] = "listening"
        ui.set(mode="live", hint="live — just talk")

    def on_confirmed_text(turn: Turn) -> bool:
        """Runs once per turn when it is both confirmed and transcribed. False = quit."""
        if turn.handled:
            return True
        turn.handled = True
        if not turn.text:
            finish_turn(turn)
            return True
        if turn.hold:
            schedule_hold_expire(hold.hold(turn.text), hold_s)
            ui.set(hint="…go on, I'm listening")
            finish_turn(turn)
            return True
        if turn.merged:
            hold.clear()
        ui.line("you", turn.text)
        if is_command(turn.text):
            keep_going = handle_transcript(turn.text)
            finish_turn(turn)
            return keep_going
        notes.add("user", turn.text)
        if not turn.llm_started:
            start_llm(turn)
        return True

    def confirm(turn: Turn) -> bool:
        turn.confirmed.set()
        turn.t_confirm = time.time()
        state["mode"] = "responding"
        state["barged"] = False
        ui.set(mode="assistant", voicing=False, hint=tts.display_name)
        for after_s, which in ((acks.after_s if acks.enabled else 0.0, "ack"), (filler_after_s, "filler")):
            if after_s > 0:
                wait = max(0.0, after_s - (time.time() - turn.t_user_stop))
                timer = threading.Timer(wait, lambda which=which: evq.put(("deadline", turn.id, which)))
                timer.daemon = True
                timer.start()
        if turn.text is not None:
            return on_confirmed_text(turn)
        return True

    def new_turn(speculative: bool, silence_s: float) -> Turn:
        old = state["turn"]
        if old is not None and not old.confirmed.is_set():
            old.cancel.set()
        turn = Turn(speculative, time.time() - silence_s)
        state["turn"] = turn
        return turn

    def log_latency(turn: Turn):
        def ms(t):
            return (t - turn.t_user_stop) * 1000 if t else -1
        logging.info(
            "latency (from end of speech): stt=%.0fms confirm=%.0fms llm_token=%.0fms "
            "first_audio=%.0fms filler=%.0fms speculative=%s route=%s",
            ms(turn.t_stt), ms(turn.t_confirm), ms(turn.t_token), ms(turn.t_audio),
            ms(turn.t_filler), turn.speculative, turn.route,
        )
        first = ms(turn.t_audio)
        if first > slow_turn_ms:
            ui.line("vox", f"slow turn: first audio {first:.0f}ms after you stopped (budget {slow_turn_ms})")

    ui.line("vox", f"mic {mic_name}  ·  speaker {speaker}" + (f" #{sink}" if sink else ""))
    if mic_note:
        logging.warning(mic_note)
        ui.line("vox", f"NOTE: {mic_note}")
    if not src:
        logging.warning("Mic %r not found in PipeWire sources; capturing from the default input", mic_name)
        ui.line("vox", f"WARNING: mic '{mic_name}' not found — plugged in? Using the default input instead.")
    if not sink:
        logging.warning("Speaker %r not found in PipeWire sinks; playing to the default output", speaker)
        ui.line("vox", f"WARNING: speaker '{speaker}' not found — using the default output instead.")
    ui.line("vox", f"v{version()}  ·  stt {stt.device}  ·  tts {tts.device}  ·  voice {profile_id}  ·  brain {llm.model}")
    try:
        speak_ack(random.choice(STARTUP_LINES))
        if tools is not None:
            tools.resume_jobs()
        stt_worker.start()
        threading.Thread(target=tts_worker, name="tts", daemon=True).start()
        capture.start()
        if (cfg.get("audio") or {}).get("device_watch", True):
            watcher = DeviceWatcher(mic_prefs, src, "preferred" if not mic_note else "fallback")

            def watch_devices():
                while not stop.wait(3.0):
                    try:
                        msg = watcher.step()
                    except Exception:
                        logging.exception("device watch failed")
                        continue
                    if msg:
                        logging.warning(msg)
                        ui.line("vox", msg)

            threading.Thread(target=watch_devices, name="device-watch", daemon=True).start()
        ui.set(mode="live", hint="live — just talk")

        last_halt_check = 0.0
        while not stop.is_set():
            if time.monotonic() - last_halt_check >= 1.0:  # second line of defence behind the kill
                last_halt_check = time.monotonic()
                if is_halted():
                    logging.warning("halt flag present: stopping")
                    ui.line("vox", "HALT engaged: stopping")
                    break
            announcer.poll()
            if tools and getattr(tools, "reminders", None) and floor_is_free() and not tools.has_announcement():
                line = tools.poll_voice_reminders()
                if line:
                    say_unprompted(line)
            maybe_aside()
            try:
                ev = evq.get(timeout=0.2)
            except queue.Empty:
                if (can_end and state["mode"] == "listening" and llm.has_session()
                        and llm.idle_s() > idle_end_s and not hold.pending):
                    threading.Thread(target=lambda: llm.end() and ui.line("vox", "conversation saved"),
                                     name="idle-end", daemon=True).start()
                    llm.last_activity = time.monotonic()  # one attempt per quiet spell
                continue
            kind = ev[0]

            if kind == "level":
                _peak, in_speech = ev[1], ev[2]
                state["in_speech"] = in_speech
                if state["cap_fail"] and time.monotonic() - state["cap_restart_at"] > 10:
                    state["cap_fail"] = 0  # it has been running steadily since the reopen
                level_box["v"] = level_box["v"] * 0.55 + _peak * 0.45
                ui.set(
                    level=min(1.0, level_box["v"] * 4.0),
                    voicing=in_speech and state["mode"] == "listening",
                )
            elif kind == "speech_start":
                if state["mode"] == "listening":
                    ui.set(mode="live", voicing=True, hint="got you — pause when done")
                # while responding: wait for sustained speech (partial) before barge-in
            elif kind == "partial":
                pcm = ev[1]
                if state["mode"] == "responding":
                    dur_ms = len(pcm) / 16.0
                    if not state["barged"] and dur_ms >= barge_in_ms:
                        barge_in()
                elif live_captions:
                    stt_worker.submit_partial(pcm)
            elif kind == "interrupt_partial":
                if state["mode"] == "responding" and not state["barged"]:
                    stt_worker.submit_partial(ev[1])
            elif kind == "partial_text":
                text = ev[1]
                if state["mode"] == "listening":
                    ui.set(hint=f"hearing: {text[:60]}")
                elif state["mode"] == "responding" and not state["barged"] and is_interrupt_command(text):
                    logging.info("interrupt keyword heard: %r", text)
                    barge_in()
            elif kind == "pause":
                if state["mode"] == "responding" and not state["barged"]:
                    continue  # assistant echo off the speaker — not a user turn
                turn = new_turn(True, ev[3])
                stt_worker.submit_final(ev[1], ev[2], tag=turn.id)
            elif kind == "resume":
                turn = state["turn"]
                if turn is not None and not turn.confirmed.is_set():
                    turn.cancel.set()
                    state["turn"] = None
                    logging.info("speculation %d cancelled: speech resumed", turn.id)
            elif kind == "speech_end":
                if state["mode"] == "responding" and not state["barged"]:
                    continue  # assistant echo off the speaker — not a user turn
                turn = state["turn"]
                if not (turn is not None and turn.speculative
                        and not turn.confirmed.is_set() and not turn.cancel.is_set()):
                    turn = new_turn(False, ev[3])
                    stt_worker.submit_final(wav_bytes_to_f32(ev[1]), ev[2], tag=turn.id)
                if not confirm(turn):
                    break
            elif kind == "final_text":
                text, tag = ev[1], ev[3]
                turn = state["turn"]
                if turn is None or turn.id != tag or turn.cancel.is_set():
                    continue  # stale speculation
                turn.t_stt = time.time()
                if text and echo_gate is not None and echo_gate.is_echo(text):
                    logging.info("echo gate: dropped %.0f%% overlap with own voice: %r",
                                 echo_gate.overlap(text) * 100, text[:80])
                    text = ""
                if text and hold.pending:
                    text, turn.merged = hold.peek_merge(text), True  # continues what was left hanging
                elif text and is_filler_only(text):
                    text = ""  # "Um..." is not a turn
                turn.text = text
                turn.hold = bool(text) and ends_dangling(text) and not is_command(text)
                if text and not turn.hold and not is_command(text):
                    start_llm(turn)  # may run ahead of confirmation; output is gated
                if turn.confirmed.is_set() and not on_confirmed_text(turn):
                    break
            elif kind == "deadline":
                turn = state["turn"]
                # only when the brain is still silent: a reply already streaming
                # would otherwise queue behind the filler and arrive later
                if (turn is not None and turn.id == ev[1] and turn.llm_started
                        and not turn.cancel.is_set() and not turn.failed
                        and not turn.t_token and not turn.t_audio and not turn.t_filler):
                    line = acks.pick(turn.text) if ev[2] == "ack" else filler_line
                    if line is not None:
                        turn.t_filler = time.time()
                        pcm, sr = tts.phrase(line)
                        speaker_out.write_pcm(pcm, sr, on_start=lambda line=line: heard_line(line))
            elif kind == "aside":
                if floor_is_free() and llm.last_activity == ev[2]:  # nothing happened while the model was thinking
                    logging.info("initiative: speaking %r", ev[1])
                    initiative.spoke(ev[1])
                    llm.note_aside(ev[1])
                    say_unprompted(ev[1])
                else:
                    logging.info("initiative: dropped, the moment passed")
                    initiative.nothing_to_say()
            elif kind == "hold_expire":
                if not hold.pending or hold.seq != ev[1]:
                    continue
                cur = state["turn"]
                if state["mode"] == "responding" or (
                        cur is not None and not cur.confirmed.is_set() and not cur.cancel.is_set()):
                    schedule_hold_expire(hold.seq, 0.7)  # the user is mid-sentence again
                    continue
                turn = new_turn(False, 0.0)
                turn.text, turn.t_stt = hold.take(), time.time()
                if not confirm(turn):
                    break
            elif kind == "turn_done":
                turn = state["turn"]
                if turn is None or turn.id != ev[1]:
                    continue
                reply = "".join(turn.reply)
                if reply:
                    spoken.record(reply)
                    notes.add("assistant", reply)
                    if not owns_memory:
                        remember_later(turn.text, reply)
                log_latency(turn)
                finish_turn(turn)
            elif kind == "announce_done":
                if state.get("announce_cancel") is ev[1]:
                    state["announce_cancel"] = None
                    if state["mode"] == "responding":
                        state["mode"] = "listening"
                        ui.set(mode="live", hint="live — just talk")
            elif kind == "capture_error":
                state["cap_fail"] += 1
                if state["cap_fail"] > 5:
                    ui.line("vox", "mic keeps failing — giving up; start vox again once it is plugged in")
                    break
                ui.line("vox", f"mic stream lost: reopening ({state['cap_fail']}/5)")
                logging.warning("capture lost; reopening (%d/5)", state["cap_fail"])
                time.sleep(min(2 ** state["cap_fail"], 15))
                choice = choose_mic(read_status(), mic_prefs)
                if choice:
                    subprocess.run(["wpctl", "set-default", choice.id], check=False)
                state["cap_restart_at"] = time.monotonic()
                capture.start()
    finally:
        stop.set()
        stt_worker.stop()
        tts_q.put(None)
        speaker_out.close()
        if can_end and not is_halted():
            with suppress(Exception):
                llm.end()  # summary, facts, tasks and the session file are written from here
        with suppress(Exception):
            getattr(llm, "close", lambda: None)()  # the Composer agent process must not outlive Apollo


def main():
    from voice.presets import list_profiles

    parser = argparse.ArgumentParser(prog="vox", description="Live voice assistant.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="cmd")
    p_talk = sub.add_parser("talk", help="live mic + assistant on the speaker (default)")
    p_talk.add_argument(
        "--voice",
        choices=list_profiles(),
        help="voice profile (default: tts.profile in config)",
    )
    args = parser.parse_args()
    cfg = _load_cfg(Path(args.config))
    if args.cmd is None or args.cmd == "talk":
        cmd_talk(cfg, voice=getattr(args, "voice", None))
        return


if __name__ == "__main__":
    main()
