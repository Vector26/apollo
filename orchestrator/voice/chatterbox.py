"""Chatterbox Turbo loader and synthesis."""

from __future__ import annotations

import logging
import threading
import types
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def patch_chatterbox_f32(model) -> None:
    """float64 -> float32 fixes for upstream chatterbox (from Voicebox / vox tts_engine)."""
    tokzr = model.s3gen.tokenizer
    orig_log_mel = tokzr.log_mel_spectrogram.__func__

    def _f32_log_mel(self_tokzr, audio, padding=0):
        import torch

        if torch.is_tensor(audio):
            audio = audio.float()
        return orig_log_mel(self_tokzr, audio, padding)

    tokzr.log_mel_spectrogram = types.MethodType(_f32_log_mel, tokzr)

    ve = model.ve
    orig_ve_forward = ve.forward.__func__

    def _f32_ve_forward(self_ve, mels):
        return orig_ve_forward(self_ve, mels.float())

    ve.forward = types.MethodType(_f32_ve_forward, ve)


class ChatterboxSynth:
    def __init__(self, model_dir: str | Path, device: str = "auto"):
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model_dir = Path(model_dir)
        logger.info("Loading Chatterbox Turbo from %s on %s", model_dir, device)
        self.model = ChatterboxTurboTTS.from_local(str(model_dir), device)
        patch_chatterbox_f32(self.model)
        self.sample_rate = getattr(self.model, "sr", None) or 24000
        self._lock = threading.Lock()
        self._cond_key: Optional[tuple[str, float]] = None

    def _ensure_conditionals(self, ref_audio: str | Path, exaggeration: float) -> None:
        key = (str(ref_audio), exaggeration)
        if self._cond_key == key:
            return
        self.model.prepare_conditionals(str(ref_audio), exaggeration=exaggeration, norm_loudness=True)
        self._cond_key = key

    def generate(
        self,
        text: str,
        ref_audio: str | Path,
        *,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        exaggeration: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }

        with self._lock:
            # Turbo only uses exaggeration while building the voice conditionals.
            self._ensure_conditionals(ref_audio, exaggeration or 0.0)
            wav = self.model.generate(text, **kwargs)

        import torch

        if isinstance(wav, torch.Tensor):
            audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        else:
            audio = np.asarray(wav, dtype=np.float32).squeeze()
        return audio, self.sample_rate
