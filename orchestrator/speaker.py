"""Keep a Bluetooth speaker stream warm so A2DP does not eat the first syllables."""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from audio_io import wav_to_pcm

logger = logging.getLogger(__name__)

PLAY_RATE = 48000


def silence_s16(ms: int, rate: int = PLAY_RATE) -> np.ndarray:
    n = max(0, int(rate * ms / 1000))
    return np.zeros(n, dtype=np.int16)


def resample_s16(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or pcm.size == 0:
        return pcm.astype(np.int16, copy=False)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(src_rate, dst_rate)
    y = resample_poly(pcm.astype(np.float32), dst_rate // g, src_rate // g)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


SLICE_MS = 40
LEAD_S = 0.2  # max audio written ahead of the playback clock (bounds barge-in lag)

# (pw-cat latency, write-ahead, idle keepalive margin). Bluetooth sinks pull audio in
# large, jittery quanta and drift against the system clock; wired margins underrun there.
BUFFERING = {
    "wired": (60, LEAD_S, 0.03),
    "bluetooth": (200, 0.6, 0.25),
}


def _inspect(target: str | None) -> str:
    if not target:
        return ""
    try:
        return subprocess.run(["wpctl", "inspect", str(target)], capture_output=True,
                              text=True, timeout=2).stdout
    except Exception:
        return ""


def sink_is_bluetooth(target: str | None) -> bool:
    return "bluez" in _inspect(target)


def sink_node_name(target: str | None) -> str | None:
    """node.name for a wpctl id. pw-cat --target matches names and serials, not the
    ids wpctl prints; an unmatched id silently plays on the default sink."""
    m = re.search(r'node\.name = "([^"]+)"', _inspect(target))
    return m.group(1) if m else None


GAP_MIN_MS = 50      # shorter silences are slice jitter, not a stall
GAP_MAX_MS = 2500    # longer silences are a new utterance, not a gap inside one


def voice_gap_ms(now: float, voice_until: float) -> int | None:
    """Silence (ms) between the end of the last real audio and a chunk starting at ``now``.
    None when it is not a gap inside one stretch of speech (first audio, jitter, or a new utterance)."""
    if voice_until <= 0.0:
        return None
    gap = int((now - voice_until) * 1000)
    return gap if GAP_MIN_MS <= gap <= GAP_MAX_MS else None


class WarmSpeaker:
    """Persistent pw-cat playback. Preroll once, then keep the sink fed with silence.

    write_pcm() only queues; a writer thread paces slices into pw-cat so the
    caller can synthesize the next clause while this one plays.
    """

    RESPAWN_EVERY_S = 3.0

    def __init__(self, target: str | None, preroll_ms: int = 500, buffering: str | None = None,
                 resolve_target: Callable[[], str | None] | None = None,
                 on_change: Callable[[str | None], None] | None = None):
        self.target = target
        self.resolve_target = resolve_target  # called when the output vanishes: where to play now (None = default output)
        self.on_change = on_change
        self._next_respawn = 0.0
        self.preroll_ms = preroll_ms
        if buffering is None:
            buffering = "bluetooth" if sink_is_bluetooth(target) else "wired"
        self.buffering = buffering
        self.latency_ms, self.lead_s, self.keep_ahead_s = BUFFERING[buffering]
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._keeper: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._cv = threading.Condition()
        self._pending: deque[tuple[np.ndarray, Callable[[], None] | None]] = deque()
        self._busy = False
        self._gen = 0
        self._play_until = 0.0  # monotonic time the written audio finishes playing
        self._voice_until = 0.0  # same, for real audio only (not keepalive silence)
        self.gaps: deque[int] = deque(maxlen=200)  # silent gaps (ms) between consecutive chunks of speech

    def _spawn(self):
        cmd = [
            "pw-cat", "-p",
            "--rate", str(PLAY_RATE),
            "--channels", "1",
            "--format", "s16",
            "--latency", f"{self.latency_ms}ms",
        ]
        if self.target:
            name = sink_node_name(self.target) if str(self.target).isdigit() else str(self.target)
            if not name:
                logger.warning("Sink %s has no node.name; playing on the default output", self.target)
            cmd += ["--target", name or str(self.target)]
        cmd.append("-")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def start(self):
        self._spawn()
        with self._lock:
            self._write_unlocked(silence_s16(self.preroll_ms))
        self._stop.clear()
        self._keeper = threading.Thread(target=self._keepalive, daemon=True)
        self._keeper.start()
        self._writer = threading.Thread(target=self._write_loop, name="speaker", daemon=True)
        self._writer.start()
        logger.info("WarmSpeaker started target=%s preroll=%dms buffering=%s", self.target,
                    self.preroll_ms, self.buffering)

    def write_wav(self, wav_bytes: bytes):
        pcm, sr = wav_to_pcm(wav_bytes)
        self.write_pcm(pcm, sr)

    def write_pcm(self, pcm: np.ndarray, src_rate: int,
                  on_start: Callable[[], None] | None = None):
        """Queue raw PCM (int16 or float32) for playback; returns immediately.

        ``on_start`` runs on the writer thread when this chunk starts playing;
        it never runs if the chunk is flushed first.
        """
        if pcm.dtype != np.int16:
            pcm = np.clip(pcm, -1.0, 1.0)
            pcm = (pcm * 32767).astype(np.int16)
        pcm = resample_s16(pcm, src_rate, PLAY_RATE)
        with self._cv:
            self._pending.append((pcm, on_start))
            self._cv.notify_all()

    def after_playback(self, cb: Callable[[], None]):
        """Run ``cb`` once everything queued before it has played (dropped by flush)."""
        with self._cv:
            self._pending.append((np.zeros(0, dtype=np.int16), cb))
            self._cv.notify_all()

    def voice_active(self, tail_s: float = 0.0) -> bool:
        """True while queued/written audio is playing, plus ``tail_s`` afterwards."""
        with self._cv:
            if self._pending or self._busy:
                return True
        return time.monotonic() < self._voice_until + tail_s

    def flush(self):
        """Drop queued audio (barge-in). At most LEAD_S already sent still plays."""
        with self._cv:
            self._pending.clear()
            self._gen += 1
            self._cv.notify_all()

    def drain(self, timeout: float = 30.0) -> bool:
        """Block until everything queued has been played. False on timeout."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._pending or self._busy:
                left = deadline - time.monotonic()
                if left <= 0 or self._stop.is_set():
                    return False
                self._cv.wait(min(left, 0.05))
        with self._lock:
            tail = self._play_until - time.monotonic()
        if tail > 0:
            time.sleep(min(tail, max(0.0, deadline - time.monotonic())))
        return time.monotonic() <= deadline

    def close(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._writer:
            self._writer.join(timeout=1)
        if self._keeper:
            self._keeper.join(timeout=1)
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        if self.proc:
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def _output_gone(self) -> bool:
        return self.proc is None or self.proc.poll() is not None

    def _respawn_unlocked(self) -> None:
        """The player died (its output vanished: Bluetooth dropped, a USB sink unplugged). Start it again on
        wherever audio can go now, so the voice comes back instead of staying silent for the rest of the session."""
        now = time.monotonic()
        if now < self._next_respawn:
            return
        self._next_respawn = now + self.RESPAWN_EVERY_S
        old, self.proc = self.proc, None
        if old is not None:
            try:
                old.kill()
            except OSError:
                pass
        try:
            if self.resolve_target is not None:
                self.target = self.resolve_target()
            self.buffering = "bluetooth" if sink_is_bluetooth(self.target) else "wired"
            self.latency_ms, self.lead_s, self.keep_ahead_s = BUFFERING[self.buffering]
            self._spawn()
            self._play_until = time.monotonic()
            self._write_unlocked(silence_s16(self.preroll_ms))
        except Exception:
            logger.exception("WarmSpeaker could not restart; will retry")
            return
        logger.warning("WarmSpeaker restarted on target=%s", self.target)
        if self.on_change is not None:
            try:
                self.on_change(self.target)
            except Exception:
                logger.exception("on_change callback failed")

    def _keepalive(self):
        pad = silence_s16(SLICE_MS)
        while not self._stop.is_set():
            time.sleep(0.02)
            with self._lock:
                if self._output_gone():
                    self._respawn_unlocked()
                if self._busy:
                    continue
                if self._play_until - time.monotonic() < self.keep_ahead_s:
                    self._write_unlocked(pad)

    def _fire(self, cb: Callable[[], None], gen: int):
        if self._gen != gen or self._stop.is_set():
            return
        try:
            cb()
        except Exception:
            logger.exception("on_start callback failed")

    def _write_loop(self):
        step = PLAY_RATE * SLICE_MS // 1000
        while not self._stop.is_set():
            with self._cv:
                while not self._pending and not self._stop.is_set():
                    self._cv.wait(0.1)
                if self._stop.is_set():
                    return
                pcm, on_start = self._pending.popleft()
                gen = self._gen
                self._busy = True
            try:
                if pcm.size == 0:
                    if on_start is not None:
                        with self._lock:
                            starts_in = max(0.0, self._play_until - time.monotonic())
                        timer = threading.Timer(starts_in, self._fire, args=(on_start, gen))
                        timer.daemon = True
                        timer.start()
                    continue
                for i in range(0, len(pcm), step):
                    if self._stop.is_set() or self._gen != gen:
                        break
                    ahead = self._play_until - time.monotonic() - self.lead_s
                    if ahead > 0:
                        time.sleep(ahead)
                    with self._lock:
                        now = time.monotonic()
                        starts_in = max(0.0, self._play_until - now)
                        if i == 0:
                            self._note_gap(voice_gap_ms(now, self._voice_until))
                        self._write_unlocked(pcm[i:i + step])
                        self._voice_until = self._play_until
                    if i == 0 and on_start is not None:
                        timer = threading.Timer(starts_in, self._fire, args=(on_start, gen))
                        timer.daemon = True
                        timer.start()
            finally:
                with self._cv:
                    self._busy = bool(self._pending)
                    self._cv.notify_all()

    def _note_gap(self, gap_ms: int | None) -> None:
        if gap_ms is None:
            return
        self.gaps.append(gap_ms)
        logger.info("speaker gap %dms between chunks", gap_ms)

    def _write_unlocked(self, pcm: np.ndarray):
        if not self.proc or not self.proc.stdin:
            return
        try:
            self.proc.stdin.write(pcm.tobytes())
            self.proc.stdin.flush()
        except BrokenPipeError:
            logger.warning("WarmSpeaker pipe broke")
            self.proc = None
            return
        now = time.monotonic()
        self._play_until = max(self._play_until, now) + len(pcm) / PLAY_RATE
