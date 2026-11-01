"""Host-side streaming STT for the live voice path.

Engines (``stt.engine`` in config.yaml):
  parakeet — NeMo Parakeet TDT 0.6b v3 via onnx-asr, int8 on CPU (frees GPU VRAM for TTS/RVC)
  whisper  — faster-whisper large-v3-turbo (CTranslate2), cuda when headroom allows else cpu

Partials transcribe the growing utterance buffer while the user speaks;
one final pass on speech end produces the committed transcript.
"""

from apollo_home import apollo_path
import collections
import logging
import os
import queue
import re
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = apollo_path("models/faster-whisper-large-v3-turbo")
NO_SPEECH_PROB = 0.6
NO_SPEECH_LOGPROB = -1.0
FINAL_BEAM_SIZE = 5
# What Whisper "hears" in breath, clicks and room tone when that is the whole utterance.
NOISE_PHRASES = {"you", "thanks for watching", "thank you for watching", "please subscribe",
                 "subscribe", "bye bye"}


def _too_quiet(pcm_f32_16k: np.ndarray, rms_floor: float = 1e-3) -> bool:
    """Skip the model on silence/clicks so Parakeet/Whisper do not hallucinate 'Thank you.'"""
    if pcm_f32_16k.size < 800:  # under ~50 ms at 16 kHz
        return True
    rms = float(np.sqrt(np.mean(np.square(pcm_f32_16k, dtype=np.float64))))
    return rms < rms_floor


def _filter_transcript(text: str) -> str:
    """Drop common hallucinations on silence/noise (Whisper and Parakeet both do this sometimes)."""
    t = text.strip()
    if re.sub(r"[^a-z ]", "", t.lower()).strip() in NOISE_PHRASES:
        return ""
    return t


def pick_device(preferred: str = "auto", min_free_vram_mb: int = 4000) -> str:
    """Choose cuda when there is headroom beside the LLM, else cpu."""
    if preferred in ("cuda", "cpu"):
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            if free / 1e6 >= min_free_vram_mb:
                return "cuda"
            logger.warning("Only %.0f MB VRAM free — STT on CPU", free / 1e6)
    except Exception as e:
        logger.warning("CUDA probe failed (%s) — STT on CPU", e)
    return "cpu"


class StreamingSTT:
    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "auto",
                 compute_type: Optional[str] = None, language: str = "en"):
        from faster_whisper import WhisperModel

        self.device = pick_device(device)
        self.compute_type = compute_type or ("int8" if self.device == "cuda" else "int8")
        self.language = language
        logger.info("Loading faster-whisper %s on %s (%s)", model_path, self.device, self.compute_type)
        self.model = WhisperModel(
            model_path,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, pcm_f32_16k: np.ndarray, beam_size: int = 1) -> str:
        """Transcribe float32 16 kHz mono PCM. Silero already did VAD."""
        if pcm_f32_16k.size == 0 or _too_quiet(pcm_f32_16k):
            return ""
        segments, _info = self.model.transcribe(
            pcm_f32_16k,
            language=self.language,
            vad_filter=False,
            beam_size=beam_size,
            without_timestamps=True,
        )
        # Whisper's own no-speech rule: noise and room tone otherwise come back as
        # "Thank you." / "you" and would be answered as if spoken.
        text = " ".join(
            s.text for s in segments
            if not (s.no_speech_prob > NO_SPEECH_PROB and s.avg_logprob < NO_SPEECH_LOGPROB)
        ).strip()
        return _filter_transcript(text)


class ParakeetSTT:
    """Parakeet TDT 0.6b v3 (int8) on CPU only — keeps Whisper off the GPU during voice sessions."""

    def __init__(
        self,
        model: str = "nemo-parakeet-tdt-0.6b-v3",
        quantization: str = "int8",
        threads: int = 4,
    ):
        import onnxruntime as ort
        import onnx_asr

        self.device = "cpu"
        self.model_id = model
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        logger.info("Loading Parakeet %s on cpu (%s, %d threads)", model, quantization, so.intra_op_num_threads)
        self.model = onnx_asr.load_model(
            model,
            quantization=quantization or None,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

    def transcribe(self, pcm_f32_16k: np.ndarray, beam_size: int = 1) -> str:
        del beam_size  # greedy decoder inside onnx-asr
        if pcm_f32_16k.size == 0 or _too_quiet(pcm_f32_16k):
            return ""
        audio = np.ascontiguousarray(pcm_f32_16k, dtype=np.float32)
        return _filter_transcript(str(self.model.recognize(audio, sample_rate=16000)))


def build_stt(cfg: dict):
    """Construct the configured STT engine (Parakeet or Whisper)."""
    engine = str(cfg.get("engine") or "whisper").lower()
    if engine == "parakeet":
        return ParakeetSTT(
            model=cfg.get("model", "nemo-parakeet-tdt-0.6b-v3"),
            quantization=str(cfg.get("quantization", "int8")),
            threads=int(cfg.get("threads", 4)),
        )
    if engine != "whisper":
        raise ValueError(f"unknown stt.engine {engine!r}; use parakeet or whisper")
    return StreamingSTT(
        model_path=cfg.get("model", DEFAULT_MODEL),
        device=cfg.get("device", "auto"),
        compute_type=cfg.get("compute_type"),
    )


class STTWorker:
    """Dedicated STT thread. Partials are latest-wins; finals are never dropped."""

    def __init__(self, stt: StreamingSTT, event_queue: queue.Queue):
        self.stt = stt
        self.events = event_queue
        self._cv = threading.Condition()
        self._finals: collections.deque = collections.deque()
        self._partial: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stt", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def submit_partial(self, pcm: np.ndarray):
        """Latest-wins; only runs when no final is waiting."""
        with self._cv:
            self._partial = pcm
            self._cv.notify_all()

    def submit_final(self, pcm: np.ndarray, duration_s: float, tag=None):
        """Never dropped; emits ("final_text", text, duration_s, tag)."""
        with self._cv:
            self._finals.append((pcm, duration_s, tag))
            self._partial = None
            self._cv.notify_all()

    def _run(self):
        while not self._stop.is_set():
            with self._cv:
                while not self._finals and self._partial is None and not self._stop.is_set():
                    self._cv.wait()
                if self._stop.is_set():
                    return
                if self._finals:
                    job, is_final = self._finals.popleft(), True
                else:
                    job, is_final = self._partial, False
                    self._partial = None
            if is_final:
                pcm, duration_s, tag = job
                try:
                    # the committed transcript is what gets answered: greedy decoding
                    # garbles short utterances ("hey hi what's up" → "you know, it's all")
                    text = self.stt.transcribe(pcm, beam_size=FINAL_BEAM_SIZE)
                except Exception as e:
                    logger.error("Final STT failed: %s", e)
                    text = ""
                self.events.put(("final_text", text, duration_s, tag))
                continue
            try:
                text = self.stt.transcribe(job)
            except Exception as e:
                logger.error("Partial STT failed: %s", e)
                continue
            if text:
                self.events.put(("partial_text", text))


def wav_bytes_to_f32(wav_bytes: bytes) -> np.ndarray:
    from audio_io import wav_to_pcm

    pcm, _sr = wav_to_pcm(wav_bytes)
    return pcm.astype(np.float32) / 32768.0
