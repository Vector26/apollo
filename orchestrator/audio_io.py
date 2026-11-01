"""Audio capture (VAD-segmented) and playback with barge-in."""

import collections
import io
import logging
import queue
import threading
import time
import wave
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import torch

logger = logging.getLogger(__name__)

VAD_WINDOW = 512  # Silero at 16 kHz


def pcm_to_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32/int16 PCM to WAV bytes."""
    if pcm.dtype != np.int16:
        pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_to_pcm(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes to int16 PCM + sample rate."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return pcm, sr


class AudioPlayer:
    """Plays audio with interrupt support."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._playing = threading.Event()

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def stop(self):
        self._stop_event.set()
        sd.stop()

    def play(self, wav_bytes: bytes, device: Optional[int] = None):
        """Play WAV bytes; returns early if stop() called."""
        pcm, sr = wav_to_pcm(wav_bytes)
        self._stop_event.clear()
        self._playing.set()
        try:
            audio = pcm.astype(np.float32) / 32768.0
            sd.play(audio, sr, device=device, blocking=False)
            while sd.get_stream().active:
                if self._stop_event.is_set():
                    sd.stop()
                    return
                sd.sleep(50)
        finally:
            self._playing.clear()


class VADCapture:
    """Continuous mic capture with Silero VAD utterance segmentation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        capture_rate: int = 48000,
        vad_threshold: float = 0.5,
        silence_ms: int = 800,
        min_speech_ms: int = 300,
        device: Optional[int] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_level: Optional[Callable[[float, bool], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        self.sample_rate = sample_rate
        self.capture_rate = capture_rate
        self.vad_threshold = vad_threshold
        self.silence_samples = int(sample_rate * silence_ms / 1000)
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.device = device
        self.on_speech_start = on_speech_start
        self.on_level = on_level
        self.stop_event = stop_event

        self._model, self._utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self._get_speech_timestamps = self._utils[0]

    def _resample(self, chunk: np.ndarray) -> np.ndarray:
        """Capture-rate chunk -> 16kHz for VAD."""
        if self.capture_rate == self.sample_rate:
            return chunk
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(self.capture_rate, self.sample_rate)
        return resample_poly(chunk, self.sample_rate // g, self.capture_rate // g).astype(np.float32)

    def listen_once(self) -> Optional[bytes]:
        """Record one utterance; return WAV bytes or None.

        Capture callback only copies samples. VAD runs off-callback so the
        USB dongle does not overrun (DWM-123 / Jieli full-speed).
        """
        logger.debug("Listening…")
        buffer: list[np.ndarray] = []
        speech_started = False
        silence_count = 0
        speech_samples = 0
        leftover = np.zeros(0, dtype=np.float32)
        chunks = queue.Queue(maxsize=48)
        done = threading.Event()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio status: %s", status)
            raw = indata[:, 0].copy()
            if self.on_level:
                peak = float(np.max(np.abs(raw))) if raw.size else 0.0
                self.on_level(peak, speech_started)
            if self.stop_event and self.stop_event.is_set():
                raise sd.CallbackStop()
            try:
                chunks.put_nowait(raw)
            except queue.Full:
                try:
                    chunks.get_nowait()
                except queue.Empty:
                    pass
                try:
                    chunks.put_nowait(raw)
                except queue.Full:
                    pass

        try:
            with sd.InputStream(
                samplerate=self.capture_rate,
                channels=1,
                dtype="float32",
                blocksize=2048,
                latency="high",
                device=self.device,
                callback=callback,
            ):
                while not done.is_set():
                    if self.stop_event and self.stop_event.is_set():
                        break
                    try:
                        raw = chunks.get(timeout=0.15)
                    except queue.Empty:
                        continue
                    chunk = self._resample(raw)
                    leftover = np.concatenate([leftover, chunk]) if leftover.size else chunk
                    while leftover.size >= VAD_WINDOW:
                        win = leftover[:VAD_WINDOW]
                        leftover = leftover[VAD_WINDOW:]
                        prob = self._model(
                            torch.from_numpy(win.copy()), self.sample_rate
                        ).item()
                        if prob >= self.vad_threshold:
                            if not speech_started:
                                speech_started = True
                                silence_count = 0
                                if self.on_speech_start:
                                    self.on_speech_start()
                            buffer.append(win)
                            speech_samples += len(win)
                            silence_count = 0
                        elif speech_started:
                            buffer.append(win)
                            speech_samples += len(win)
                            silence_count += len(win)
                            if silence_count >= self.silence_samples:
                                done.set()
                                break
        except sd.CallbackStop:
            pass

        if speech_samples < self.min_speech_samples:
            return None

        audio = np.concatenate(buffer)
        pcm = (audio * 32767).astype(np.int16)
        logger.info("Captured utterance: %.1fs", len(pcm) / self.sample_rate)
        return pcm_to_wav(pcm, self.sample_rate)


class ContinuousCapture:
    """Always-on mic capture with Silero VAD. Emits events; never blocks.

    Events are put on ``event_queue`` as tuples:

    - ``("level", peak: float, in_speech: bool)``
    - ``("speech_start",)``
    - ``("partial", pcm_f32_16k)`` — growing utterance buffer, every partial_ms
    - ``("pause", pcm_f32_16k, duration_s, silence_s)`` — silence reached
      speculate_ms: likely end of turn, not yet committed
    - ``("resume",)`` — speech came back after a pause event
    - ``("speech_end", wav_bytes, duration_s, silence_s)``

    The PortAudio callback only copies samples (DWM-123 / Jieli full-speed
    dongle overruns if we do work in the callback). VAD runs in the consumer
    loop of the capture thread.
    """

    PREROLL_MS = 300

    def __init__(
        self,
        sample_rate: int = 16000,
        capture_rate: int = 48000,
        vad_threshold: float = 0.5,
        silence_ms: int = 200,
        min_speech_ms: int = 250,
        partial_ms: int = 600,
        device: Optional[int] = None,
        event_queue: Optional[queue.Queue] = None,
        stop_event: Optional[threading.Event] = None,
        latency: str = "high",
        muted: Optional[Callable[[], bool]] = None,
        interrupt_listen: Optional[Callable[[], bool]] = None,
        interrupt_partial_ms: int = 450,
        interrupt_min_speech_ms: int = 220,
        interrupt_silence_ms: int = 280,
        speculate_ms: int = 0,
    ):
        self.sample_rate = sample_rate
        self.capture_rate = capture_rate
        self.vad_threshold = vad_threshold
        self.silence_samples = int(sample_rate * silence_ms / 1000)
        spec = int(sample_rate * speculate_ms / 1000)
        self.speculate_samples = spec if 0 < spec < self.silence_samples else 0
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.partial_samples = int(sample_rate * partial_ms / 1000)
        self.device = device
        self.events = event_queue if event_queue is not None else queue.Queue()
        self.stop_event = stop_event or threading.Event()
        self.latency = latency
        self.muted = muted  # True → ignore the mic (assistant audible: echo)
        self.interrupt_listen = interrupt_listen  # while muted: still VAD for keyword interrupt only
        self.interrupt_partial_samples = int(sample_rate * interrupt_partial_ms / 1000)
        self.interrupt_min_speech_samples = int(sample_rate * interrupt_min_speech_ms / 1000)
        self.interrupt_silence_samples = int(sample_rate * interrupt_silence_ms / 1000)

        self._model, self._utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        self._thread: Optional[threading.Thread] = None

    STALL_S = 10.0  # an open stream that delivers nothing this long has lost its device: report it, don't sit deaf

    def start(self):
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def _emit(self, *event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            pass

    def _resample(self, chunk: np.ndarray) -> np.ndarray:
        if self.capture_rate == self.sample_rate:
            return chunk
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(self.capture_rate, self.sample_rate)
        return resample_poly(chunk, self.sample_rate // g, self.capture_rate // g).astype(np.float32)

    def _run(self):
        buffer: list[np.ndarray] = []
        preroll: collections.deque = collections.deque(
            maxlen=max(1, int(self.sample_rate * self.PREROLL_MS / 1000 / VAD_WINDOW))
        )
        in_speech = False
        paused = False  # a "pause" event is outstanding for this utterance
        silence_count = 0
        speech_samples = 0
        since_partial = 0
        leftover = np.zeros(0, dtype=np.float32)
        ichunk: list[np.ndarray] = []
        i_in_speech = False
        i_silence = 0
        i_speech_samples = 0
        i_since_partial = 0
        chunks: queue.Queue = queue.Queue(maxsize=48)

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio status: %s", status)
            raw = indata[:, 0].copy()
            if self.stop_event.is_set():
                raise sd.CallbackStop()
            try:
                chunks.put_nowait(raw)
            except queue.Full:
                try:
                    chunks.get_nowait()
                except queue.Empty:
                    pass
                try:
                    chunks.put_nowait(raw)
                except queue.Full:
                    pass

        try:
            with sd.InputStream(
                samplerate=self.capture_rate,
                channels=1,
                dtype="float32",
                blocksize=2048,
                latency=self.latency,
                device=self.device,
                callback=callback,
            ):
                last_chunk = time.monotonic()
                while not self.stop_event.is_set():
                    try:
                        raw = chunks.get(timeout=0.15)
                    except queue.Empty:
                        if time.monotonic() - last_chunk > self.STALL_S:
                            raise RuntimeError(f"no audio from the mic for {self.STALL_S:.0f}s")
                        continue
                    last_chunk = time.monotonic()
                    chunk = self._resample(raw)
                    leftover = np.concatenate([leftover, chunk]) if leftover.size else chunk
                    while leftover.size >= VAD_WINDOW:
                        win = leftover[:VAD_WINDOW]
                        leftover = leftover[VAD_WINDOW:]
                        peak = float(np.max(np.abs(win))) if win.size else 0.0
                        if self.muted is not None and self.muted():
                            listen_interrupt = (
                                self.interrupt_listen is not None and self.interrupt_listen()
                            )
                            if not listen_interrupt:
                                if in_speech or buffer or preroll:
                                    if paused:
                                        self._emit("resume")
                                    in_speech = False
                                    paused = False
                                    buffer = []
                                    preroll.clear()
                                    self._model.reset_states()
                                ichunk, i_in_speech, i_silence, i_speech_samples, i_since_partial = [], False, 0, 0, 0
                                self._emit("level", peak, False)
                                continue
                            # Keyword interrupt path: short utterances only, never commit a full turn.
                            if in_speech or buffer or preroll:
                                if paused:
                                    self._emit("resume")
                                in_speech = False
                                paused = False
                                buffer = []
                                preroll.clear()
                                self._model.reset_states()
                            prob = self._model(
                                torch.from_numpy(win.copy()), self.sample_rate
                            ).item()
                            self._emit("level", peak, i_in_speech)
                            if prob >= self.vad_threshold:
                                if not i_in_speech:
                                    i_in_speech = True
                                    i_silence = 0
                                    i_speech_samples = 0
                                    i_since_partial = 0
                                    ichunk = []
                                    self._model.reset_states()
                                ichunk.append(win)
                                i_speech_samples += len(win)
                                i_since_partial += len(win)
                                i_silence = 0
                                if (
                                    i_speech_samples >= self.interrupt_min_speech_samples
                                    and i_since_partial >= self.interrupt_partial_samples
                                ):
                                    i_since_partial = 0
                                    self._emit("interrupt_partial", np.concatenate(ichunk).copy())
                            elif i_in_speech:
                                ichunk.append(win)
                                i_speech_samples += len(win)
                                i_silence += len(win)
                                if (
                                    i_speech_samples >= self.interrupt_min_speech_samples
                                    and i_silence >= self.interrupt_silence_samples
                                ):
                                    self._emit("interrupt_partial", np.concatenate(ichunk).copy())
                                    ichunk, i_in_speech, i_silence, i_speech_samples, i_since_partial = [], False, 0, 0, 0
                            continue
                        prob = self._model(
                            torch.from_numpy(win.copy()), self.sample_rate
                        ).item()
                        self._emit("level", peak, in_speech)
                        if prob >= self.vad_threshold:
                            if not in_speech:
                                in_speech = True
                                silence_count = 0
                                speech_samples = 0
                                since_partial = 0
                                buffer = list(preroll)
                                self._model.reset_states()
                                self._emit("speech_start")
                            if paused:
                                paused = False
                                self._emit("resume")
                            buffer.append(win)
                            speech_samples += len(win)
                            since_partial += len(win)
                            silence_count = 0
                            if since_partial >= self.partial_samples:
                                since_partial = 0
                                self._emit("partial", np.concatenate(buffer).copy())
                        elif in_speech:
                            buffer.append(win)
                            speech_samples += len(win)
                            since_partial += len(win)
                            silence_count += len(win)
                            if (
                                self.speculate_samples
                                and not paused
                                and silence_count >= self.speculate_samples
                                and speech_samples >= self.min_speech_samples
                            ):
                                paused = True
                                audio = np.concatenate(buffer)
                                self._emit(
                                    "pause", audio.copy(), len(audio) / self.sample_rate,
                                    silence_count / self.sample_rate,
                                )
                            if silence_count >= self.silence_samples:
                                in_speech = False
                                paused = False
                                if speech_samples >= self.min_speech_samples:
                                    audio = np.concatenate(buffer)
                                    pcm = (audio * 32767).astype(np.int16)
                                    self._emit(
                                        "speech_end",
                                        pcm_to_wav(pcm, self.sample_rate),
                                        len(pcm) / self.sample_rate,
                                        silence_count / self.sample_rate,
                                    )
                                buffer = []
                        else:
                            preroll.append(win)
        except sd.CallbackStop:
            pass
        except Exception:
            logger.exception("Capture thread died")
            self._emit("capture_error")
