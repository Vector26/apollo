#!/usr/bin/env python3
"""Mic tools: live level range, delayed echo, or push-to-play.

PipeWire defaults: wireless USB mic in, system default out (aux).
Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import select
import signal
import sys
import termios
import threading
import time
import tty

import numpy as np
import sounddevice as sd

ESC = "\033"
GREEN = f"{ESC}[32m"
RED = f"{ESC}[31m"
YELLOW = f"{ESC}[33m"
CYAN = f"{ESC}[36m"
DIM = f"{ESC}[2m"
RESET = f"{ESC}[0m"
BOLD = f"{ESC}[1m"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def vu_bar(level: float, width: int = 32) -> str:
    level = max(0.0, min(1.0, level))
    filled = int(round(level * width))
    return "█" * filled + "░" * (width - filled)


def to_db(peak: float) -> float:
    return 20.0 * np.log10(max(peak, 1e-9))


class LevelTracker:
    """Peak, smoothed level, slow peak-hold, and session max (0..1)."""

    def __init__(self, hold_decay: float = 0.96):
        self.hold_decay = hold_decay
        self.smooth = 0.0
        self.peak = 0.0
        self.hold = 0.0
        self.session_max = 0.0
        self._lock = threading.Lock()

    def push_chunk(self, chunk: np.ndarray) -> None:
        peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        with self._lock:
            self.peak = peak
            self.smooth = self.smooth * 0.55 + peak * 0.45
            self.hold = max(peak, self.hold * self.hold_decay)
            self.session_max = max(self.session_max, peak)

    def snapshot(self) -> tuple[float, float, float, float]:
        with self._lock:
            return self.peak, self.smooth, self.hold, self.session_max


def format_range_line(
    peak: float,
    smooth: float,
    hold: float,
    session_max: float,
    threshold: float,
    extra: str = "",
) -> str:
    speaking = smooth >= threshold
    spin = SPINNER[int(time.time() * 12) % len(SPINNER)] if speaking else " "
    label = f"{RED}{BOLD}SPEAK{RESET}" if speaking else f"{GREEN}listen{RESET}"
    bar = vu_bar(min(1.0, smooth * 5.0))
    pct = int(smooth * 100)
    pk = int(peak * 100)
    hd = int(hold * 100)
    mx = int(session_max * 100)
    db = to_db(smooth)
    line = (
        f"\r {spin} {label} {GREEN}{bar}{RESET} "
        f"{CYAN}{pct:3d}%{RESET} "
        f"{DIM}pk{pk:3d} hold{hd:3d} max{mx:3d} ({db:+.0f}dB){RESET}"
    )
    if extra:
        line += f"  {DIM}{extra}{RESET}"
    return line + "   "


class DelayRing:
    def __init__(self, sample_rate: int, delay_sec: float, blocksize: int, levels: LevelTracker):
        self.delay_samples = max(1, int(sample_rate * delay_sec))
        self.buf = np.zeros(self.delay_samples + blocksize * 8, dtype=np.float32)
        self.write_pos = 0
        self.levels = levels
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self.levels.push_chunk(chunk)
        n = len(chunk)
        with self._lock:
            blen = len(self.buf)
            for i, s in enumerate(chunk):
                self.buf[(self.write_pos + i) % blen] = s
            self.write_pos += n
            start = self.write_pos - self.delay_samples - n
            out = np.empty(n, dtype=np.float32)
            for i in range(n):
                out[i] = self.buf[(start + i) % blen]
            return out


class RecordRing:
    """Rolling buffer of recent mic audio for push-to-play."""

    def __init__(self, sample_rate: int, max_sec: float):
        self.max_samples = max(1, int(sample_rate * max_sec))
        self.buf = np.zeros(self.max_samples, dtype=np.float32)
        self.pos = 0
        self.full = False
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        with self._lock:
            for s in chunk:
                self.buf[self.pos] = s
                self.pos = (self.pos + 1) % self.max_samples
                if self.pos == 0:
                    self.full = True

    def take_all(self) -> np.ndarray:
        with self._lock:
            if not self.full and self.pos == 0:
                return np.zeros(0, dtype=np.float32)
            if self.full:
                out = np.concatenate([self.buf[self.pos :], self.buf[: self.pos]])
            else:
                out = self.buf[: self.pos].copy()
            self.pos = 0
            self.full = False
            return out


def start_key_listener(stop: threading.Event, play_event: threading.Event, keys: set[str]) -> threading.Thread:
    """Background thread: fire play_event when user presses one of keys (cbreak stdin)."""

    def run():
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch in keys:
                    play_event.set()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    t = threading.Thread(target=run, name="keys", daemon=True)
    t.start()
    return t


def run_echo(args: argparse.Namespace) -> int:
    levels = LevelTracker()
    ring = DelayRing(args.rate, args.delay, args.blocksize, levels)
    stop = threading.Event()
    pending = np.zeros(0, dtype=np.float32)
    pend_lock = threading.Lock()

    def on_sig(*_):
        stop.set()

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    in_dev = sd.query_devices(kind="input")
    out_dev = sd.query_devices(kind="output")
    print(
        f"{DIM}ECHO  in: {in_dev['name']}  ·  out: {out_dev['name']}  ·  "
        f"delay {args.delay}s — Ctrl-C stop{RESET}",
        flush=True,
    )

    def in_cb(indata, frames, time_info, status):
        if stop.is_set():
            raise sd.CallbackStop()
        delayed = ring.push(indata[:, 0])
        with pend_lock:
            nonlocal pending
            pending = np.concatenate([pending, delayed]) if pending.size else delayed.copy()

    def out_cb(outdata, frames, time_info, status):
        if stop.is_set():
            outdata.fill(0)
            raise sd.CallbackStop()
        need = frames
        with pend_lock:
            nonlocal pending
            if pending.size < need:
                out = np.zeros(need, dtype=np.float32)
                if pending.size:
                    out[: pending.size] = pending
                    pending = np.zeros(0, dtype=np.float32)
            else:
                out = pending[:need].copy()
                pending = pending[need:]
        outdata[:, 0] = out

    in_stream = out_stream = None
    try:
        in_stream = sd.InputStream(
            samplerate=args.rate,
            channels=1,
            dtype="float32",
            blocksize=args.blocksize,
            latency="high",
            callback=in_cb,
        )
        out_stream = sd.OutputStream(
            samplerate=args.rate,
            channels=1,
            dtype="float32",
            blocksize=args.blocksize,
            latency="high",
            callback=out_cb,
        )
        in_stream.start()
        out_stream.start()
        while not stop.is_set():
            peak, smooth, hold, mx = levels.snapshot()
            sys.stdout.write(
                format_range_line(
                    peak, smooth, hold, mx, args.threshold, extra=f"echo {args.delay:.1f}s"
                )
            )
            sys.stdout.flush()
            time.sleep(0.05)
    except sd.CallbackStop:
        pass
    finally:
        _close_streams(in_stream, out_stream)
    return 0


def run_ptt(args: argparse.Namespace) -> int:
    levels = LevelTracker()
    recorder = RecordRing(args.rate, args.buffer_sec)
    stop = threading.Event()
    play_event = threading.Event()
    playing = threading.Event()

    key = args.key
    if len(key) != 1:
        print("error: --key must be a single character", file=sys.stderr)
        return 2
    keys = {key, key.lower(), key.upper()} if key.isalpha() else {key}

    def on_sig(*_):
        stop.set()

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    in_dev = sd.query_devices(kind="input")
    out_dev = sd.query_devices(kind="output")
    key_label = repr(key) if key == " " else key
    print(
        f"{DIM}PTT   in: {in_dev['name']}  ·  out: {out_dev['name']}  ·  "
        f"buffer {args.buffer_sec:.0f}s  ·  press {YELLOW}{key_label}{RESET}{DIM} to play — Ctrl-C stop{RESET}",
        flush=True,
    )

    def in_cb(indata, frames, time_info, status):
        if stop.is_set():
            raise sd.CallbackStop()
        chunk = indata[:, 0]
        levels.push_chunk(chunk)
        if not playing.is_set():
            recorder.push(chunk)

    def play_worker():
        while not stop.is_set():
            if not play_event.wait(0.1):
                continue
            play_event.clear()
            audio = recorder.take_all()
            if audio.size == 0:
                continue
            playing.set()
            try:
                sd.play(audio, args.rate, blocking=True)
            except Exception as e:
                print(f"\nplay failed: {e}", file=sys.stderr)
            finally:
                playing.clear()

    start_key_listener(stop, play_event, keys)
    threading.Thread(target=play_worker, name="play", daemon=True).start()

    in_stream = None
    try:
        in_stream = sd.InputStream(
            samplerate=args.rate,
            channels=1,
            dtype="float32",
            blocksize=args.blocksize,
            latency="high",
            callback=in_cb,
        )
        in_stream.start()
        while not stop.is_set():
            peak, smooth, hold, mx = levels.snapshot()
            hint = f"press {key_label} to play" if not playing.is_set() else "playing…"
            sys.stdout.write(format_range_line(peak, smooth, hold, mx, args.threshold, extra=hint))
            sys.stdout.flush()
            time.sleep(0.05)
    except sd.CallbackStop:
        pass
    finally:
        _close_streams(in_stream, None)
    return 0


def run_monitor(args: argparse.Namespace) -> int:
    """Level meter only — no speaker output."""
    levels = LevelTracker()
    stop = threading.Event()

    def on_sig(*_):
        stop.set()

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    in_dev = sd.query_devices(kind="input")
    print(
        f"{DIM}MONITOR  in: {in_dev['name']}  ·  range only — Ctrl-C stop{RESET}",
        flush=True,
    )

    def in_cb(indata, frames, time_info, status):
        if stop.is_set():
            raise sd.CallbackStop()
        levels.push_chunk(indata[:, 0])

    in_stream = None
    try:
        in_stream = sd.InputStream(
            samplerate=args.rate,
            channels=1,
            dtype="float32",
            blocksize=args.blocksize,
            latency="high",
            callback=in_cb,
        )
        in_stream.start()
        while not stop.is_set():
            peak, smooth, hold, mx = levels.snapshot()
            sys.stdout.write(format_range_line(peak, smooth, hold, mx, args.threshold, extra="monitor"))
            sys.stdout.flush()
            time.sleep(0.05)
    except sd.CallbackStop:
        pass
    finally:
        _close_streams(in_stream, None)
    return 0


def _close_streams(in_stream, out_stream) -> None:
    sys.stdout.write(f"\r{RESET}\n")
    sys.stdout.flush()
    for s in (in_stream, out_stream):
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Mic level meter, echo, or push-to-play.")
    parser.add_argument("--rate", type=int, default=48000)
    parser.add_argument("--blocksize", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.015, help="SPEAK indicator threshold")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--ptt",
        action="store_true",
        help="record mic continuously; play buffer when you press --key (no live echo)",
    )
    mode.add_argument(
        "--monitor",
        action="store_true",
        help="show mic range only; no playback",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="echo mode: delay seconds")
    parser.add_argument(
        "--key",
        default=" ",
        help="PTT mode: character to play recording (default: space)",
    )
    parser.add_argument(
        "--buffer-sec",
        type=float,
        default=15.0,
        help="PTT mode: max seconds kept before each play (default 15)",
    )
    args = parser.parse_args()

    if args.ptt:
        return run_ptt(args)
    if args.monitor:
        return run_monitor(args)
    return run_echo(args)


if __name__ == "__main__":
    raise SystemExit(main())
